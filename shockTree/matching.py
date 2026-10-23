"""Indexed physical candidates, heuristic confidence, and sparse global assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from .config import MatchOptions
from .model import KMS_TO_KPC_GYR, Snapshot, minimum_image


EDGE_DTYPE = np.dtype([
    ("parent", "<i8"), ("child", "<i8"), ("cost", "<f8"),
    ("distance", "<f8"), ("allowed", "<f8"), ("normal_cost", "<f8"),
    ("mach_cost", "<f8"), ("dissipation_cost", "<f8"), ("available_weight", "<f8"),
])


@dataclass
class PairResult:
    edges: np.ndarray
    scores: np.ndarray
    accepted: np.ndarray  # Indices into edges.
    stats: dict


def _increment(stats, reason, mask):
    stats["rejected"][reason] = stats["rejected"].get(reason, 0) + int(np.count_nonzero(mask))


def _query_batches(tree, prediction, radii, chunk, limit):
    """Count first; subdivide queries before materializing neighbor lists."""
    pending = [(lo, min(len(prediction), lo + chunk)) for lo in range(0, len(prediction), chunk)]
    while pending:
        lo, hi = pending.pop()
        counts = tree.query_ball_point(prediction[lo:hi], radii[lo:hi], return_length=True)
        total = int(np.sum(counts))
        if total > limit:
            if hi - lo == 1:
                raise ValueError(f"one spatial query has {total:,} neighbors; tighten physical search settings or raise max_query_candidates")
            middle = (lo + hi) // 2
            pending.extend([(middle, hi), (lo, middle)])
            continue
        if not total:
            continue
        neighbors = tree.query_ball_point(prediction[lo:hi], radii[lo:hi], return_sorted=True)
        parent = np.repeat(np.arange(lo, hi, dtype=np.int64), counts)
        child = np.concatenate([np.asarray(x, dtype=np.int64) for x in neighbors if len(x)])
        yield parent, child


def candidates(parent: Snapshot, child: Snapshot, opts: MatchOptions, box=None):
    opts.validate()
    dt = child.time_gyr - parent.time_gyr
    if dt <= 0 or child.timestep <= parent.timestep:
        raise ValueError("matching requires increasing snapshot numbers and physical time")
    stats = {"parent_snapshot": parent.timestep, "child_snapshot": child.timestep,
             "elapsed_gyr": dt, "parent_count": len(parent), "child_count": len(child),
             "spatial_candidates": 0, "eligible_candidates": 0, "history_predictions": 0,
             "rejected": {}, "normal_comparison": "signed dot product", "prediction": "bounded nearest compatible position / established branch velocity"}
    if not len(parent) or not len(child) or dt > opts.max_interval_gyr:
        if dt > opts.max_interval_gyr:
            stats["rejected"]["interval_exceeds_limit"] = len(parent)
        return np.empty(0, dtype=EDGE_DTYPE), stats
    box = None if box is None else np.asarray(box, dtype=float)
    # Conservative upper bound on integral(v_phys/a dt), using the smaller a.
    a_min = min(parent.aexp, child.aexp)
    base = max(opts.minimum_displacement_kpc, opts.max_speed_kms * KMS_TO_KPC_GYR * dt) / a_min
    prediction = parent.pos.copy()
    history = np.zeros(len(parent), dtype=bool)
    if parent.velocity is not None:
        history = np.all(np.isfinite(parent.velocity), axis=1)
        speed = np.linalg.norm(parent.velocity, axis=1) * parent.aexp / KMS_TO_KPC_GYR
        history &= speed <= opts.max_speed_kms
        if parent.velocity_score is not None:
            history &= parent.velocity_score >= opts.history_min_score
        prediction[history] += parent.velocity[history] * dt
    stats["history_predictions"] = int(history.sum())
    if box is not None:
        prediction %= box
        maximum = base + opts.cell_slack * max(float(parent.cell_size.max()), float(child.cell_size.max()))
        if maximum >= float(box.min()) / 2:
            raise ValueError("allowed displacement reaches half the periodic box; winding is ambiguous; shorten the interval or tighten the physical bound")
    residual_base = np.full(len(parent), base)
    residual_base[history] *= opts.prediction_uncertainty_fraction
    # Resolution bins avoid using the coarsest descendant cell for every query.
    levels = np.floor(np.log2(child.cell_size / child.cell_size.min()) + 1e-10).astype(np.int64)
    chunks = []
    stored = 0
    for level in np.unique(levels):
        rows = np.flatnonzero(levels == level)
        positions = child.pos[rows].copy()
        if box is not None:
            positions %= box
        index = cKDTree(positions, boxsize=box)
        radii = residual_base + opts.cell_slack * np.maximum(parent.cell_size, child.cell_size[rows].max())
        for p, local_child in _query_batches(index, prediction, radii, opts.query_chunk, opts.max_query_candidates):
            q = rows[local_child]
            stats["spatial_candidates"] += len(p)
            cell_allowance = opts.cell_slack * np.maximum(parent.cell_size[p], child.cell_size[q])
            allowed = residual_base[p] + cell_allowance
            displacement = minimum_image(child.pos[q] - prediction[p], box)
            distance = np.linalg.norm(displacement, axis=1)
            traveled = np.linalg.norm(minimum_image(child.pos[q] - parent.pos[p], box), axis=1)
            cosine = np.clip(np.einsum("ij,ij->i", parent.normal[p], child.normal[q]), -1, 1)
            mach_difference = np.abs(np.log(child.mach[q]) - np.log(parent.mach[p]))
            available_energy = (np.isfinite(parent.dissipation[p]) & (parent.dissipation[p] > 0)
                                & np.isfinite(child.dissipation[q]) & (child.dissipation[q] > 0))
            energy_difference = np.zeros(len(p))
            energy_difference[available_energy] = np.abs(np.log(child.dissipation[q[available_energy]])
                                                        - np.log(parent.dissipation[p[available_energy]]))
            valid = np.ones(len(p), dtype=bool)
            gates = (
                ("prediction_distance", distance > allowed),
                ("propagation_distance", traveled > base + cell_allowance),
                ("normal_orientation", cosine < opts.normal_min_cosine),
                ("mach_evolution", mach_difference > np.log(opts.max_mach_ratio)),
                ("dissipation_evolution", available_energy & (energy_difference > np.log(opts.max_dissipation_ratio))),
            )
            for name, rejected in gates:
                _increment(stats, name, valid & rejected)
                valid &= ~rejected
            cn = 1 - cosine
            cm = mach_difference / opts.mach_log_scale
            ce = energy_difference / opts.dissipation_log_scale
            weights = opts.w_pos + opts.w_n + opts.w_mach + opts.w_dissipation * available_energy
            cost = (opts.w_pos * distance / allowed + opts.w_n * cn + opts.w_mach * cm
                    + opts.w_dissipation * ce) / weights
            _increment(stats, "total_cost", valid & (cost > opts.max_cost))
            valid &= np.isfinite(cost) & (cost <= opts.max_cost)
            count = int(valid.sum())
            if not count:
                continue
            stored += count
            if stored > opts.max_candidate_edges:
                raise ValueError(f"eligible candidate edges exceed {opts.max_candidate_edges:,}; memory guard stopped this pair without approximating assignment")
            edges = np.empty(count, dtype=EDGE_DTYPE)
            for name, value in (("parent", p), ("child", q), ("cost", cost), ("distance", distance),
                                ("allowed", allowed), ("normal_cost", cn), ("mach_cost", cm),
                                ("dissipation_cost", ce), ("available_weight", weights)):
                edges[name] = value[valid]
            chunks.append(edges)
        del index, positions
    edges = np.concatenate(chunks) if chunks else np.empty(0, dtype=EDGE_DTYPE)
    stats["eligible_candidates"] = len(edges)
    return edges, stats


def _best_two(edges, field, count):
    costs = edges["cost"]
    groups = edges[field]
    best_cost = np.full(count, np.inf)
    np.minimum.at(best_cost, groups, costs)
    edge_number = np.arange(len(edges), dtype=np.int64)
    best_edge = np.full(count, len(edges), dtype=np.int64)
    tied = costs == best_cost[groups]
    np.minimum.at(best_edge, groups[tied], edge_number[tied])
    second_cost = np.full(count, np.inf)
    other = edge_number != best_edge[groups]
    np.minimum.at(second_cost, groups[other], costs[other])
    return best_cost, best_edge, second_cost


def association_scores(edges, np_parent, np_child, opts):
    """Bounded association quality, not a calibrated probability.

    score = exp(-C/T) * [(1-w_margin)+w_margin*margin] * mutual_factor
            * (available_feature_weight / total_configured_weight).
    margin = clip(min(alternative_parent-C, alternative_child-C)/scale,0,1).
    An endpoint with no alternative contributes margin=1. The alternative is
    the second-best cost for its best edge and the best cost otherwise.
    Forward/backward consistency means independently ranking descendants for
    each parent and progenitors for each child, using the same pair costs.
    Mutual first choices receive factor 1; other choices get the configured
    nonmutual factor. Equal-cost alternatives have zero margin. This is an
    explicit ranking diagnostic, not a measurement of physical identity.
    """
    if not len(edges):
        return np.empty(0)
    index = np.arange(len(edges))
    pa, pb, ps = _best_two(edges, "parent", np_parent)
    ca, cb, cs = _best_two(edges, "child", np_child)
    p, c, cost = edges["parent"], edges["child"], edges["cost"]
    parent_alternative = np.where(index == pb[p], ps[p], pa[p])
    child_alternative = np.where(index == cb[c], cs[c], ca[c])
    margin = np.clip(np.minimum(parent_alternative - cost, child_alternative - cost) / opts.margin_scale, 0, 1)
    mutual = (index == pb[p]) & (index == cb[c])
    mutual_factor = np.where(mutual, 1.0, opts.nonmutual_score_factor)
    configured_weight = opts.w_pos + opts.w_n + opts.w_mach + opts.w_dissipation
    score = (np.exp(-cost / opts.score_temperature) * (1 - opts.margin_weight + opts.margin_weight * margin)
             * mutual_factor * edges["available_weight"] / configured_weight)
    return np.clip(score, 0, 1)


def assign_primary(edges, scores, n_parent, n_child, opts):
    """Global rectangular sparse assignment with private unmatched columns.

    The sparse shortest-augmenting-path assignment has the same one-to-one
    constraints as Hungarian assignment; no N_parent by N_child dense matrix
    is formed. Missing/gated edges do not exist in the graph. Each active
    parent gets its own unmatched dummy. Real edge cost is 1-score+epsilon.
    Dummy cost is 1-min_score+2*epsilon, so unsupported pairs remain unmatched.
    """
    eligible = scores >= opts.min_score
    if not np.any(eligible):
        return np.empty(0, dtype=np.int64)
    p, c = edges["parent"][eligible], edges["child"][eligible]
    active_p, active_c = np.unique(p), np.unique(c)
    local_p, local_c = np.searchsorted(active_p, p), np.searchsorted(active_c, c)
    epsilon = 1e-10  # Sparse solver requires positive stored costs, even for score=1.
    rows = np.concatenate([local_p, np.arange(len(active_p))])
    cols = np.concatenate([local_c, len(active_c) + np.arange(len(active_p))])
    weights = np.concatenate([1 - scores[eligible] + epsilon,
                              np.full(len(active_p), 1 - opts.min_score + 2 * epsilon)])
    graph = coo_matrix((weights, (rows, cols)), shape=(len(active_p), len(active_c) + len(active_p))).tocsr()
    assigned_p, assigned_c = min_weight_full_bipartite_matching(graph)
    real = assigned_c < len(active_c)
    chosen_child = np.full(n_parent, -1, dtype=np.int64)
    chosen_child[active_p[assigned_p[real]]] = active_c[assigned_c[real]]
    selected = np.flatnonzero(eligible & (chosen_child[edges["parent"]] == edges["child"]))
    if np.unique(edges["parent"][selected]).size != selected.size or np.unique(edges["child"][selected]).size != selected.size:
        raise AssertionError("assignment produced duplicate primary endpoints")
    return selected


def match_snapshots(parent, child, opts, box=None):
    edges, stats = candidates(parent, child, opts, box)
    scores = association_scores(edges, len(parent), len(child), opts)
    stats["rejected"]["confidence"] = int(np.count_nonzero(scores < opts.min_score))
    accepted = assign_primary(edges, scores, len(parent), len(child), opts)
    stats.update(primary_links=len(accepted),
                 matched_parent_fraction=len(accepted) / len(parent) if len(parent) else 0.0,
                 matched_child_fraction=len(accepted) / len(child) if len(child) else 0.0,
                 discarded_secondary_links=int(np.count_nonzero(scores >= opts.min_score)) - len(accepted))
    # Motion for the next pair is established solely from accepted detections.
    child.velocity = np.full((len(child), 3), np.nan)
    child.velocity_score = np.zeros(len(child))
    if len(accepted):
        p, c = edges["parent"][accepted], edges["child"][accepted]
        measured = minimum_image(child.pos[c] - parent.pos[p], box) / (child.time_gyr - parent.time_gyr)
        if parent.velocity is not None:
            prior_valid = np.all(np.isfinite(parent.velocity[p]), axis=1)
            measured[prior_valid] = (opts.velocity_smoothing * measured[prior_valid]
                                     + (1 - opts.velocity_smoothing) * parent.velocity[p[prior_valid]])
        child.velocity[c] = measured
        child.velocity_score[c] = scores[accepted]
    return PairResult(edges, scores, accepted, stats)
