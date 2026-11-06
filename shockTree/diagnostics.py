"""Bounded sidecar tables and scientific plots, separate from the minimal tree."""

import csv
import json
from pathlib import Path

import numpy as np

from .model import decode_key, minimum_image, node_key


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _distribution(values, bins):
    values = np.asarray(values)
    counts, edges = np.histogram(values, bins=bins)
    output = {"count": len(values), "histogram_counts": counts.tolist(), "histogram_edges": edges.tolist()}
    if len(values):
        output.update(min=float(values.min()), max=float(values.max()), mean=float(values.mean()),
                      median=float(np.median(values)), p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))
    return output


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def pair_diagnostics(directory, parent, child, result, cfg, box=None, plot=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stats = result.stats
    edge, score = result.edges, result.scores
    accepted = result.accepted
    feasible = score >= cfg.matching.min_score
    parent_degree = np.bincount(edge["parent"][feasible], minlength=len(parent))
    child_degree = np.bincount(edge["child"][feasible], minlength=len(child))
    stats.update(possible_split_parents=int(np.count_nonzero(parent_degree > 1)),
                 possible_merge_children=int(np.count_nonzero(child_degree > 1)))
    stats["score_distribution"] = _distribution(score[accepted], np.linspace(0, 1, 21))
    distance = edge["distance"][accepted]
    upper = max(float(distance.max()) if len(distance) else 0, 1e-12)
    stats["prediction_residual_comoving_kpc"] = _distribution(distance, np.linspace(0, upper, 21))
    stats["prediction_residual_physical_kpc_at_child"] = _distribution(distance * child.aexp, np.linspace(0, upper * child.aexp, 21))
    primary = np.zeros(len(edge), dtype=bool)
    primary[accepted] = True
    secondary = np.flatnonzero(feasible & ~primary)
    total_secondary = len(secondary)
    if len(secondary) > cfg.secondary_limit:
        if cfg.secondary_limit:
            secondary = secondary[np.argpartition(score[secondary], -cfg.secondary_limit)[-cfg.secondary_limit:]]
        else:
            secondary = secondary[:0]
    secondary = secondary[np.argsort(-score[secondary], kind="stable")]
    table_fields = ["parent_timestep", "parent_shock_id", "parent_key", "child_timestep", "child_shock_id", "child_key",
                    "score", "cost", "distance_comoving_kpc", "allowed_comoving_kpc", "normal_cost", "mach_cost",
                    "dissipation_cost", "possible_split", "possible_merge"]
    with (directory / "secondary_links.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=table_fields)
        writer.writeheader()
        for index in secondary:
            item = edge[index]
            p, c = item["parent"], item["child"]
            writer.writerow(dict(zip(table_fields, [parent.timestep, int(parent.ids[p]), int(node_key(parent.timestep, parent.ids[p])),
                child.timestep, int(child.ids[c]), int(node_key(child.timestep, child.ids[c])), float(score[index]),
                float(item["cost"]), float(item["distance"]), float(item["allowed"]), float(item["normal_cost"]),
                float(item["mach_cost"]), float(item["dissipation_cost"]), bool(parent_degree[p] > 1), bool(child_degree[c] > 1)])))
    stats["secondary_rows_written"] = len(secondary)
    stats["secondary_table_truncated"] = len(secondary) < total_secondary
    write_json(directory / "matching_statistics.json", stats)
    if not plot:
        return
    plt = _pyplot()
    from matplotlib.collections import LineCollection
    selected = accepted[np.linspace(0, len(accepted) - 1, min(len(accepted), cfg.plot_samples), dtype=np.int64)] if len(accepted) else accepted
    p, c = edge["parent"][selected], edge["child"][selected]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    if len(selected):
        a = parent.pos[p]
        b = a + minimum_image(child.pos[c] - a, box)
        origin = np.median(a, axis=0)
        segments = np.stack([a[:, :2] - origin[:2], b[:, :2] - origin[:2]], axis=1)
        collection = LineCollection(segments, cmap="viridis", linewidths=.7, alpha=.6)
        collection.set_array(score[selected])
        collection.set_clim(0, 1)
        axes[0].add_collection(collection)
        axes[0].scatter(a[:, 0] - origin[0], a[:, 1] - origin[1], s=3, color="steelblue", label="parent")
        axes[0].scatter(b[:, 0] - origin[0], b[:, 1] - origin[1], s=3, color="tomato", label="child")
        fig.colorbar(collection, ax=axes[0], label="association score")
        axes[0].legend(fontsize=8)
        axes[0].autoscale()
    else:
        axes[0].text(.5, .5, "No accepted links", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set(xlabel="x offset (comoving kpc)", ylabel="y offset (comoving kpc)", title=f"{cfg.diagnostic_label}: {parent.timestep} → {child.timestep}")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].hist(score[accepted], bins=np.linspace(0, 1, 21), color="steelblue")
    axes[1].set(xlabel="association score", ylabel="link count", title="All accepted links")
    axes[2].hist(distance, bins=20, color="tomato")
    axes[2].set(xlabel="prediction residual (comoving kpc)", ylabel="link count", title="All accepted links")
    fig.savefig(directory / "matched_overlay.png", dpi=160)
    plt.close(fig)


def branch_diagnostics(directory, tree, state, segments, metadata, cfg):
    timesteps = list(segments)
    times = {t: metadata[t].time_gyr for t in timesteps}
    max_lifetime = max(times.values()) - min(times.values()) if times else 0
    length_edges = np.arange(.5, len(timesteps) + 1.5, 1.)
    lifetime_edges = np.linspace(0, max(max_lifetime, 1e-12), 21)
    score_edges = np.linspace(0, 1, 21)
    length_hist = np.zeros(len(length_edges) - 1, dtype=np.int64)
    lifetime_hist = np.zeros(20, dtype=np.int64)
    score_hist = np.zeros(20, dtype=np.int64)
    count = low = written = low_written = 0
    with (Path(directory) / "branch_samples.csv").open("w", newline="") as stream, \
            (Path(directory) / "low_confidence_branches.csv").open("w", newline="") as low_stream:
        names = ["first", "last", "root_timestep", "terminal_timestep", "length", "lifetime_gyr",
                 "mean_link_score", "min_link_score", "low_confidence", "touches_selected_boundary"]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        low_writer = csv.DictWriter(low_stream, fieldnames=names)
        low_writer.writeheader()
        for lo in range(0, len(tree), cfg.chunk_rows):
            hi = min(len(tree), lo + cfg.chunk_rows)
            terminal = np.flatnonzero(tree["son"][lo:hi] == -1) + lo
            if not len(terminal):
                continue
            lengths = state["length"][terminal]
            root_time, _ = decode_key(tree["first"][terminal])
            terminal_time = tree["timestep"][terminal]
            life = np.array([times[int(last)] - times[int(first)] for first, last in zip(root_time, terminal_time)])
            minimum = state["min_score"][terminal]
            linked_branch = lengths > 1
            low_mask = linked_branch & (minimum < cfg.low_confidence_score)
            count += len(terminal)
            low += int(low_mask.sum())
            length_hist += np.histogram(lengths, length_edges)[0]
            lifetime_hist += np.histogram(life, lifetime_edges)[0]
            score_hist += np.histogram(minimum[linked_branch], score_edges)[0]
            # Table is explicitly capped; all histogram/count statistics are exhaustive.
            order = np.concatenate([np.flatnonzero(low_mask), np.flatnonzero(~low_mask)])
            order = order[:max(0, cfg.branch_diagnostic_limit - written)]
            low_order = np.flatnonzero(low_mask)[:max(0, cfg.branch_diagnostic_limit - low_written)]
            targets = [(index, writer) for index in order] + [(index, low_writer) for index in low_order]
            for index, target_writer in targets:
                row = terminal[index]
                mean = float(state["score_sum"][row] / (lengths[index] - 1)) if lengths[index] > 1 else ""
                target_writer.writerow(dict(zip(names, [int(tree["first"][row]), int(tree["last"][row]), int(root_time[index]),
                    int(terminal_time[index]), int(lengths[index]), float(life[index]), mean,
                    float(minimum[index]) if lengths[index] > 1 else "", bool(low_mask[index]),
                    bool(root_time[index] == timesteps[0] or terminal_time[index] == timesteps[-1])])))
            written += len(order)
            low_written += len(low_order)
    output = {"branches": count, "low_confidence_branches": low, "table_rows_written": written,
              "table_truncated": written < count, "isolated_branch_score": "undefined; excluded from link-score histogram",
              "low_confidence_table_rows": low_written, "low_confidence_table_truncated": low_written < low,
              "length_histogram": {"edges": length_edges.tolist(), "counts": length_hist.tolist()},
              "lifetime_gyr_histogram": {"edges": lifetime_edges.tolist(), "counts": lifetime_hist.tolist()},
              "minimum_link_score_histogram": {"edges": score_edges.tolist(), "counts": score_hist.tolist()}}
    write_json(Path(directory) / "branch_statistics.json", output)
    if cfg.max_plot_pairs:
        plt = _pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        fig.suptitle(cfg.diagnostic_label)
        for axis, edges, counts, label in zip(axes, (length_edges, lifetime_edges, score_edges),
                                             (length_hist, lifetime_hist, score_hist),
                                             ("branch length (snapshots)", "lifetime (Gyr)", "minimum link score")):
            axis.stairs(counts, edges, fill=True, color="steelblue", alpha=.8)
            axis.set(xlabel=label, ylabel="branch count")
        fig.savefig(Path(directory) / "branch_distributions.png", dpi=160)
        plt.close(fig)
    return output
