module shockfinder_kernel
  use iso_fortran_env, only: output_unit
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  private
  public :: find_shocks, build_neighbor_index, fill_fine_neighbors

contains

  ! Return the ideal-gas entropy proxy T/rho^(gamma-1).
  pure real(8) function entropy_value(temp, rho, gamma) result(s)
    real(8), intent(in) :: temp, rho, gamma

    if (rho > 0.0_dp .and. temp > 0.0_dp) then
      s = temp / rho**(gamma - 1.0_dp)
    else
      s = 0.0_dp
    end if
  end function entropy_value

  ! Invert the ideal-gas Rankine-Hugoniot temperature jump for general gamma.
  ! Ratios below unity are treated as non-shocks with Mach 1.
  pure real(8) function mach_from_temperature_jump(t_ratio, gamma) result(mach)
    real(8), intent(in) :: t_ratio, gamma
    real(8) :: a, b, c, disc, m2

    if (t_ratio <= 1.0_dp) then
      mach = 1.0_dp
      return
    end if

    a = 2.0_dp * gamma * (gamma - 1.0_dp)
    b = 4.0_dp * gamma - (gamma - 1.0_dp)**2 - t_ratio * (gamma + 1.0_dp)**2
    c = -2.0_dp * (gamma - 1.0_dp)
    disc = max(b*b - 4.0_dp*a*c, 0.0_dp)
    m2 = (-b + sqrt(disc)) / (2.0_dp*a)
    mach = sqrt(max(m2, 1.0_dp))
  end function mach_from_temperature_jump

  ! Map a coordinate axis to the positive face slot in the six-neighbor table.
  pure integer function plus_face(axis) result(face)
    integer, intent(in) :: axis
    face = axis * 2
  end function plus_face

  ! Map a coordinate axis to the negative face slot in the six-neighbor table.
  pure integer function minus_face(axis) result(face)
    integer, intent(in) :: axis
    face = axis * 2 - 1
  end function minus_face

  ! Sample a face state from a same/coarser neighbor or from finer face cells.
  ! The returned values are used for local gradients and divergence.
  pure subroutine face_sample(pos, vel, dx, temp, rho, neighbors, fine_face_index, &
       fine_neighbors, n, nfine, &
       i, face, axis, gamma, value_pos, value_vel, value_temp, value_entropy, ok)
    integer, intent(in) :: n, nfine, i, face, axis
    integer, intent(in) :: neighbors(n, 6), fine_face_index(n, 6), fine_neighbors(nfine, 4)
    real(8), intent(in) :: pos(n, 3), vel(n, 3), dx(n), temp(n), rho(n), gamma
    real(8), intent(out) :: value_pos, value_vel, value_temp, value_entropy
    logical, intent(out) :: ok

    integer :: nb, k, count, group
    real(8) :: sign

    ok = .false.
    value_pos = 0.0_dp
    value_vel = 0.0_dp
    value_temp = 0.0_dp
    value_entropy = 0.0_dp

    nb = neighbors(i, face)
    if (nb > 0) then
      value_pos = pos(nb, axis)
      value_vel = vel(nb, axis)
      value_temp = temp(nb)
      value_entropy = entropy_value(temp(nb), rho(nb), gamma)
      ok = temp(nb) > 0.0_dp .and. rho(nb) > 0.0_dp
      return
    end if

    group = fine_face_index(i, face)
    if (group <= 0 .or. group > nfine) return
    count = 0
    do k = 1, 4
      nb = fine_neighbors(group, k)
      if (nb <= 0) cycle
      if (temp(nb) <= 0.0_dp .or. rho(nb) <= 0.0_dp) cycle
      count = count + 1
      value_vel = value_vel + vel(nb, axis)
      value_temp = value_temp + temp(nb)
      value_entropy = value_entropy + entropy_value(temp(nb), rho(nb), gamma)
    end do
    if (count <= 0) return

    if (mod(face, 2) == 0) then
      sign = 1.0_dp
    else
      sign = -1.0_dp
    end if
    value_pos = pos(i, axis) + sign * 0.5_dp * dx(i)
    value_vel = value_vel / real(count, 8)
    value_temp = value_temp / real(count, 8)
    value_entropy = value_entropy / real(count, 8)
    ok = .true.
  end subroutine face_sample

  ! Compute local velocity divergence plus temperature and entropy gradients.
  ! Invalid cells or missing face pairs simply leave valid set to false.
  pure subroutine local_quantities(pos, vel, dx, temp, rho, neighbors, fine_face_index, &
       fine_neighbors, n, nfine, i, &
       gamma, divv, grad_t, grad_s, valid)
    integer, intent(in) :: n, nfine, i
    integer, intent(in) :: neighbors(n, 6), fine_face_index(n, 6), fine_neighbors(nfine, 4)
    real(8), intent(in) :: pos(n, 3), vel(n, 3), dx(n), temp(n), rho(n), gamma
    real(8), intent(out) :: divv, grad_t(3), grad_s(3)
    logical, intent(out) :: valid

    integer :: axis
    real(8) :: dist, xm, xp, vm, vp, tm, tp, sm, sp
    logical :: okm, okp

    divv = 0.0_dp
    grad_t = 0.0_dp
    grad_s = 0.0_dp
    valid = .false.

    if (i < 1 .or. i > n) return
    if (rho(i) <= 0.0_dp .or. temp(i) <= 0.0_dp .or. dx(i) <= 0.0_dp) return

    do axis = 1, 3
      call face_sample(pos, vel, dx, temp, rho, neighbors, fine_face_index, fine_neighbors, n, nfine, &
           i, minus_face(axis), axis, gamma, xm, vm, tm, sm, okm)
      call face_sample(pos, vel, dx, temp, rho, neighbors, fine_face_index, fine_neighbors, n, nfine, &
           i, plus_face(axis), axis, gamma, xp, vp, tp, sp, okp)
      if (.not. okm .or. .not. okp) cycle
      dist = xp - xm
      if (abs(dist) <= 0.0_dp) cycle

      grad_t(axis) = (tp - tm) / dist
      grad_s(axis) = (sp - sm) / dist
      divv = divv + (vp - vm) / dist
      valid = .true.
    end do
  end subroutine local_quantities

  ! Normalize a vector and report whether its magnitude was non-zero.
  pure subroutine normalize_vector(vector, unit_vector, ok)
    real(8), intent(in) :: vector(3)
    real(8), intent(out) :: unit_vector(3)
    logical, intent(out) :: ok
    real(8) :: norm

    norm = sqrt(dot_product(vector, vector))
    ok = norm > 0.0_dp
    if (ok) then
      unit_vector = vector / norm
    else
      unit_vector = 0.0_dp
    end if
  end subroutine normalize_vector

  ! Choose the neighbor crossed by a gradient walk through one AMR face.
  ! Same/coarser neighbors are preferred; finer face cells use the nearest center.
  pure integer function choose_face_neighbor(pos, neighbors, fine_face_index, fine_neighbors, &
       n, nfine, i, face, xpoint) result(nb)
    integer, intent(in) :: n, nfine, i, face
    integer, intent(in) :: neighbors(n, 6), fine_face_index(n, 6), fine_neighbors(nfine, 4)
    real(8), intent(in) :: pos(n, 3), xpoint(3)

    integer :: k, trial, group
    real(8) :: best, dist2

    nb = neighbors(i, face)
    if (nb > 0) return

    group = fine_face_index(i, face)
    if (group <= 0 .or. group > nfine) return
    best = huge(1.0_dp)
    do k = 1, 4
      trial = fine_neighbors(group, k)
      if (trial <= 0) cycle
      dist2 = (pos(trial, 1) - xpoint(1))**2 + &
              (pos(trial, 2) - xpoint(2))**2 + &
              (pos(trial, 3) - xpoint(3))**2
      if (dist2 < best) then
        best = dist2
        nb = trial
      end if
    end do
  end function choose_face_neighbor

  ! Advance one step along the shock-normal direction through the AMR mesh.
  ! The output is the crossed neighbor and the new point just inside that cell.
  pure subroutine next_along_gradient(pos, dx, neighbors, fine_face_index, fine_neighbors, &
       n, nfine, i, xold, direction, &
       next_cell, xnew)
    integer, intent(in) :: n, nfine, i
    integer, intent(in) :: neighbors(n, 6), fine_face_index(n, 6), fine_neighbors(nfine, 4)
    real(8), intent(in) :: pos(n, 3), dx(n), xold(3), direction(3)
    integer, intent(out) :: next_cell
    real(8), intent(out) :: xnew(3)

    integer :: axis, face
    real(8) :: t, tbest, half_width, eps

    next_cell = 0
    face = 0
    xnew = xold
    tbest = huge(1.0_dp)
    half_width = 0.5_dp * dx(i)

    do axis = 1, 3
      if (direction(axis) > 1.0e-14_dp) then
        t = (pos(i, axis) + half_width - xold(axis)) / direction(axis)
        if (t > 1.0e-14_dp .and. t < tbest) then
          tbest = t
          face = plus_face(axis)
        end if
      else if (direction(axis) < -1.0e-14_dp) then
        t = (pos(i, axis) - half_width - xold(axis)) / direction(axis)
        if (t > 1.0e-14_dp .and. t < tbest) then
          tbest = t
          face = minus_face(axis)
        end if
      end if
    end do

    if (tbest >= huge(1.0_dp) * 0.5_dp) return

    eps = max(dx(i), 1.0_dp) * 1.0e-10_dp
    xnew = xold + direction * (tbest + eps)
    next_cell = choose_face_neighbor(pos, neighbors, fine_face_index, fine_neighbors, &
         n, nfine, i, face, xnew)
  end subroutine next_along_gradient

  ! Resolve every candidate to a deterministic local convergence minimum along
  ! the shock normal. Previously resolved paths are reused (path compression),
  ! avoiding repeated walks through the same thick shock zone.
  subroutine resolve_shock_centers(pos, dx, neighbors, fine_face_index, fine_neighbors, &
       n, nfine, candidate, &
       divv_arr, grad_t_arr, max_center_steps, normal_cosine, plateau_tolerance, &
       resolved_center, limit_count)
    integer, intent(in) :: n, nfine, max_center_steps
    integer, intent(in) :: neighbors(n, 6), fine_face_index(n, 6), fine_neighbors(nfine, 4)
    logical, intent(in) :: candidate(n)
    real(8), intent(in) :: pos(n, 3), dx(n), divv_arr(n), grad_t_arr(n, 3)
    real(8), intent(in) :: normal_cosine, plateau_tolerance
    integer, intent(out) :: resolved_center(n)
    integer(8), intent(out) :: limit_count

    integer :: i, center, next_center, trial, step_count, path_length, j
    integer, allocatable :: path(:)
    real(8) :: best_divv, scale, tolerance, dirvec(3), trial_dir(3)
    real(8) :: xwalk(3), xnext(3)
    logical :: ok_direction, ok_trial_direction, center_changed, finished

    resolved_center = 0
    limit_count = 0_8
    allocate(path(n))

    do i = 1, n
      if (.not. candidate(i) .or. resolved_center(i) > 0) cycle

      center = i
      path_length = 0
      finished = .false.
      ! Evaluate the starting cell plus at most max_center_steps moves. This
      ! lets a center reached on the final permitted move be accepted.
      do step_count = 0, max_center_steps
        if (resolved_center(center) > 0) then
          center = resolved_center(center)
          finished = .true.
          exit
        end if

        path_length = path_length + 1
        path(path_length) = center
        call normalize_vector(grad_t_arr(center, :), dirvec, ok_direction)
        if (.not. ok_direction) exit

        best_divv = divv_arr(center)
        next_center = center
        center_changed = .false.
        xwalk = pos(center, :)

        call next_along_gradient(pos, dx, neighbors, fine_face_index, fine_neighbors, &
             n, nfine, center, xwalk, dirvec, &
             trial, xnext)
        if (trial > 0 .and. candidate(trial)) then
          call normalize_vector(grad_t_arr(trial, :), trial_dir, ok_trial_direction)
          scale = max(1.0_dp, abs(best_divv), abs(divv_arr(trial)))
          tolerance = plateau_tolerance * scale
          if (ok_trial_direction .and. abs(dot_product(dirvec, trial_dir)) >= normal_cosine) then
            if (divv_arr(trial) < best_divv - tolerance .or. &
                 (abs(divv_arr(trial) - best_divv) <= tolerance .and. trial < next_center)) then
              next_center = trial
              best_divv = divv_arr(trial)
              center_changed = .true.
            end if
          end if
        end if

        call next_along_gradient(pos, dx, neighbors, fine_face_index, fine_neighbors, &
             n, nfine, center, xwalk, -dirvec, &
             trial, xnext)
        if (trial > 0 .and. candidate(trial)) then
          call normalize_vector(grad_t_arr(trial, :), trial_dir, ok_trial_direction)
          scale = max(1.0_dp, abs(best_divv), abs(divv_arr(trial)))
          tolerance = plateau_tolerance * scale
          if (ok_trial_direction .and. abs(dot_product(dirvec, trial_dir)) >= normal_cosine) then
            if (divv_arr(trial) < best_divv - tolerance .or. &
                 (abs(divv_arr(trial) - best_divv) <= tolerance .and. trial < next_center)) then
              next_center = trial
              best_divv = divv_arr(trial)
              center_changed = .true.
            end if
          end if
        end if

        if (.not. center_changed) then
          finished = .true.
          exit
        end if
        center = next_center
      end do

      if (finished) then
        resolved_center(center) = center
        do j = 1, path_length
          resolved_center(path(j)) = center
        end do
      else
        limit_count = limit_count + 1_8
      end if
    end do

    deallocate(path)
  end subroutine resolve_shock_centers

  pure integer(8) function packed_cell_key(ix, iy, iz, cell_level, spans, &
       spatial_size, level_min) result(key)
    integer(8), intent(in) :: ix, iy, iz, spans(3), spatial_size
    integer, intent(in) :: cell_level, level_min
    key = ix + spans(1) * (iy + spans(2) * iz) + &
         int(cell_level - level_min, 8) * spatial_size
  end function packed_cell_key

  pure integer function hash_lookup(key, hash_keys, hash_rows, table_size) result(row)
    integer(8), intent(in) :: key, hash_keys(table_size)
    integer, intent(in) :: table_size, hash_rows(table_size)
    integer(8) :: mixed
    integer :: slot, probes

    mixed = ieor(key, ishft(key, -33)) * 6364136223846793005_8
    slot = 1 + int(iand(mixed, int(table_size - 1, 8)))
    row = 0
    do probes = 1, table_size
      if (hash_rows(slot) == 0) return
      if (hash_keys(slot) == key) then
        row = hash_rows(slot)
        return
      end if
      slot = slot + 1
      if (slot > table_size) slot = 1
    end do
  end function hash_lookup

  subroutine make_integer_geometry(pos, dx, level, n, lo, widths, spans, &
       spatial_size, level_min)
    integer, intent(in) :: n, level(n)
    real(8), intent(in) :: pos(n, 3), dx(n)
    integer(8), intent(out) :: lo(n, 3), widths(n), spans(3), spatial_size
    integer, intent(out) :: level_min
    integer :: i, axis
    real(8) :: finest_dx, origin(3)

    finest_dx = minval(dx)
    do axis = 1, 3
      origin(axis) = minval(pos(:, axis) - 0.5_dp * dx)
    end do
    do i = 1, n
      widths(i) = nint(dx(i) / finest_dx, 8)
      do axis = 1, 3
        lo(i, axis) = nint((pos(i, axis) - 0.5_dp * dx(i) - origin(axis)) / finest_dx, 8)
      end do
    end do
    do axis = 1, 3
      spans(axis) = maxval(lo(:, axis) + widths) + 1_8
    end do
    spatial_size = spans(1) * spans(2) * spans(3)
    level_min = minval(level)
  end subroutine make_integer_geometry

  subroutine make_hash(lo, level, n, spans, spatial_size, level_min, &
       hash_keys, hash_rows, table_size)
    integer, intent(in) :: n, level(n), level_min, table_size
    integer(8), intent(in) :: lo(n, 3), spans(3), spatial_size
    integer(8), intent(out) :: hash_keys(table_size)
    integer, intent(out) :: hash_rows(table_size)
    integer :: i, slot
    integer(8) :: key, mixed

    hash_keys = 0_8
    hash_rows = 0
    do i = 1, n
      key = packed_cell_key(lo(i, 1), lo(i, 2), lo(i, 3), level(i), &
           spans, spatial_size, level_min)
      mixed = ieor(key, ishft(key, -33)) * 6364136223846793005_8
      slot = 1 + int(iand(mixed, int(table_size - 1, 8)))
      do while (hash_rows(slot) /= 0)
        slot = slot + 1
        if (slot > table_size) slot = 1
      end do
      hash_keys(slot) = key
      hash_rows(slot) = i
    end do
  end subroutine make_hash

  ! Build same/coarser face links and a sparse index for finer faces with an
  ! O(1)-average open-addressing hash lookup. Fine-neighbor values are filled
  ! by fill_fine_neighbors after Python allocates the exact sparse size.
  subroutine build_neighbor_index(pos, dx, level, n, neighbors, fine_face_index, nfine)
    !f2py intent(in) pos, dx, level, n
    !f2py intent(out) neighbors, fine_face_index, nfine
    integer, intent(in) :: n, level(n)
    real(8), intent(in) :: pos(n, 3), dx(n)
    integer, intent(out) :: neighbors(n, 6), fine_face_index(n, 6), nfine
    integer(8), allocatable :: lo(:, :), widths(:), hash_keys(:)
    integer, allocatable :: hash_rows(:)
    integer(8) :: spans(3), spatial_size, target(3), coarse_lo(3), key, fw, cw
    integer :: level_min, table_size, i, axis, face, direction, found, k0, k1, count

    allocate(lo(n, 3), widths(n))
    call make_integer_geometry(pos, dx, level, n, lo, widths, spans, spatial_size, level_min)
    table_size = 1
    do while (table_size < 2 * n)
      table_size = table_size * 2
    end do
    allocate(hash_keys(table_size), hash_rows(table_size))
    call make_hash(lo, level, n, spans, spatial_size, level_min, &
         hash_keys, hash_rows, table_size)

    neighbors = 0
    fine_face_index = 0
    !$omp parallel do schedule(static) private(i, axis, face, direction, target, coarse_lo, &
    !$omp& key, found, fw, cw, k0, k1, count)
    do i = 1, n
      do axis = 1, 3
        do direction = -1, 1, 2
          face = 2 * axis - merge(1, 0, direction < 0)
          target = lo(i, :)
          target(axis) = target(axis) + int(direction, 8) * widths(i)
          key = packed_cell_key(target(1), target(2), target(3), level(i), &
               spans, spatial_size, level_min)
          found = hash_lookup(key, hash_keys, hash_rows, table_size)
          if (found > 0 .and. widths(found) == widths(i)) then
            neighbors(i, face) = found
            cycle
          end if

          cw = 2_8 * widths(i)
          target = lo(i, :)
          if (direction < 0) then
            target(axis) = lo(i, axis) - 1_8
          else
            target(axis) = lo(i, axis) + widths(i)
          end if
          coarse_lo = (target / cw) * cw
          key = packed_cell_key(coarse_lo(1), coarse_lo(2), coarse_lo(3), level(i) - 1, &
               spans, spatial_size, level_min)
          found = hash_lookup(key, hash_keys, hash_rows, table_size)
          if (found > 0 .and. widths(found) == cw) then
            neighbors(i, face) = found
            cycle
          end if

          if (mod(widths(i), 2_8) /= 0_8) cycle
          fw = widths(i) / 2_8
          count = 0
          do k0 = 0, 1
            do k1 = 0, 1
              target = lo(i, :)
              if (direction < 0) then
                target(axis) = lo(i, axis) - fw
              else
                target(axis) = lo(i, axis) + widths(i)
              end if
              if (axis == 1) then
                target(2) = lo(i, 2) + int(k0, 8) * fw
                target(3) = lo(i, 3) + int(k1, 8) * fw
              else if (axis == 2) then
                target(1) = lo(i, 1) + int(k0, 8) * fw
                target(3) = lo(i, 3) + int(k1, 8) * fw
              else
                target(1) = lo(i, 1) + int(k0, 8) * fw
                target(2) = lo(i, 2) + int(k1, 8) * fw
              end if
              key = packed_cell_key(target(1), target(2), target(3), level(i) + 1, &
                   spans, spatial_size, level_min)
              if (hash_lookup(key, hash_keys, hash_rows, table_size) > 0) count = count + 1
            end do
          end do
          if (count > 0) fine_face_index(i, face) = 1
        end do
      end do
    end do
    !$omp end parallel do

    nfine = 0
    do face = 1, 6
      do i = 1, n
        if (fine_face_index(i, face) /= 0) then
          nfine = nfine + 1
          fine_face_index(i, face) = nfine
        end if
      end do
    end do
    deallocate(lo, widths, hash_keys, hash_rows)
  end subroutine build_neighbor_index

  subroutine fill_fine_neighbors(pos, dx, level, fine_face_index, n, nfine, fine_neighbors)
    !f2py intent(in) pos, dx, level, fine_face_index, n, nfine
    !f2py intent(out) fine_neighbors
    integer, intent(in) :: n, nfine, level(n), fine_face_index(n, 6)
    real(8), intent(in) :: pos(n, 3), dx(n)
    integer, intent(out) :: fine_neighbors(nfine, 4)
    integer(8), allocatable :: lo(:, :), widths(:), hash_keys(:)
    integer, allocatable :: hash_rows(:)
    integer(8) :: spans(3), spatial_size, target(3), key, fw
    integer :: level_min, table_size, i, axis, face, direction, group, slot, k0, k1

    allocate(lo(n, 3), widths(n))
    call make_integer_geometry(pos, dx, level, n, lo, widths, spans, spatial_size, level_min)
    table_size = 1
    do while (table_size < 2 * n)
      table_size = table_size * 2
    end do
    allocate(hash_keys(table_size), hash_rows(table_size))
    call make_hash(lo, level, n, spans, spatial_size, level_min, &
         hash_keys, hash_rows, table_size)
    fine_neighbors = 0

    !$omp parallel do schedule(static) private(i, axis, face, direction, group, slot, &
    !$omp& k0, k1, target, key, fw)
    do i = 1, n
      do axis = 1, 3
        do direction = -1, 1, 2
          face = 2 * axis - merge(1, 0, direction < 0)
          group = fine_face_index(i, face)
          if (group <= 0) cycle
          fw = widths(i) / 2_8
          slot = 0
          do k0 = 0, 1
            do k1 = 0, 1
              slot = slot + 1
              target = lo(i, :)
              if (direction < 0) then
                target(axis) = lo(i, axis) - fw
              else
                target(axis) = lo(i, axis) + widths(i)
              end if
              if (axis == 1) then
                target(2) = lo(i, 2) + int(k0, 8) * fw
                target(3) = lo(i, 3) + int(k1, 8) * fw
              else if (axis == 2) then
                target(1) = lo(i, 1) + int(k0, 8) * fw
                target(3) = lo(i, 3) + int(k1, 8) * fw
              else
                target(1) = lo(i, 1) + int(k0, 8) * fw
                target(2) = lo(i, 2) + int(k1, 8) * fw
              end if
              key = packed_cell_key(target(1), target(2), target(3), level(i) + 1, &
                   spans, spatial_size, level_min)
              fine_neighbors(group, slot) = hash_lookup(key, hash_keys, hash_rows, table_size)
            end do
          end do
        end do
      end do
    end do
    !$omp end parallel do
    deallocate(lo, widths, hash_keys, hash_rows)
  end subroutine fill_fine_neighbors

  ! Scan AMR cells for Skillman-style shock zones and assign center Mach numbers.
  ! Neighbor tables are supplied by Python/Numba and the cell loop is OpenMP-ready.
  subroutine find_shocks(pos, vel, dx, temp, rho, level, detection_mask, neighbors, &
       fine_face_index, fine_neighbors, n, nfine, &
       gamma, temp_floor, min_mach, max_steps, max_center_steps, center_normal_cosine, &
       center_plateau_tolerance, show_progress, progress_interval, mach, shock, center_index, &
       upstream_index, downstream_index, diagnostics)
    !f2py intent(in) pos, vel, dx, temp, rho, level, detection_mask, neighbors
    !f2py intent(in) fine_face_index, fine_neighbors, n, nfine
    !f2py intent(in) gamma, temp_floor, min_mach, max_steps, max_center_steps
    !f2py intent(in) center_normal_cosine, center_plateau_tolerance, show_progress, progress_interval
    !f2py intent(out) mach, shock, center_index, upstream_index, downstream_index, diagnostics
    integer, intent(in) :: n, nfine, max_steps, max_center_steps, show_progress, progress_interval
    integer, intent(in) :: level(n), detection_mask(n), neighbors(n, 6), fine_face_index(n, 6)
    integer, intent(in) :: fine_neighbors(nfine, 4)
    real(8), intent(in) :: pos(n, 3), vel(n, 3), dx(n), temp(n), rho(n)
    real(8), intent(in) :: gamma, temp_floor, min_mach
    real(8), intent(in) :: center_normal_cosine, center_plateau_tolerance
    real(8), intent(out) :: mach(n)
    integer, intent(out) :: shock(n), center_index(n), upstream_index(n), downstream_index(n)
    integer(8), intent(out) :: diagnostics(10)

    integer :: i, step_count, done
    integer :: center, trial, upstream, downstream
    integer :: progress_count, next_progress
    integer :: clock_rate, clock_now, pre_start, scan_start
    real(8) :: grad_t(3), grad_s(3), dirvec(3), xwalk(3), xnext(3)
    real(8) :: t_pre, t_post, rho_pre, rho_post, ratio, m, elapsed
    real(8), allocatable :: divv_arr(:), grad_t_arr(:, :)
    logical, allocatable :: candidate(:)
    integer, allocatable :: resolved_center(:)
    logical :: valid, ok_direction, endpoint_found

    mach = 0.0_dp
    shock = 0
    center_index = 0
    upstream_index = 0
    downstream_index = 0
    diagnostics = 0_8

    allocate(divv_arr(n), grad_t_arr(n, 3), candidate(n), resolved_center(n))

    ! Precompute all local shock diagnostics once. This loop is independent for
    ! each AMR cell and is therefore a good OpenMP target.
    progress_count = 0
    next_progress = progress_interval
    call system_clock(count_rate=clock_rate)
    call system_clock(pre_start)

    !$omp parallel do schedule(static) private(i, done, grad_s, valid, clock_now, elapsed)
    do i = 1, n
      call local_quantities(pos, vel, dx, temp, rho, neighbors, fine_face_index, &
           fine_neighbors, n, nfine, i, &
           gamma, divv_arr(i), grad_t_arr(i, :), grad_s, valid)
      candidate(i) = detection_mask(i) /= 0 .and. valid .and. divv_arr(i) < 0.0_dp .and. &
           dot_product(grad_t_arr(i, :), grad_s) > 0.0_dp
      if (show_progress /= 0 .and. progress_interval > 0) then
        !$omp atomic capture
        progress_count = progress_count + 1
        done = progress_count
        !$omp end atomic
        if (done >= next_progress) then
          !$omp critical(progress_write)
          if (done >= next_progress) then
            call system_clock(clock_now)
            elapsed = real(clock_now - pre_start, 8) / real(max(clock_rate, 1), 8)
            write(output_unit, '(A,I0,A,I0,A,F5.1,A,F10.1,A)') "ShockFinder Fortran precompute: ", &
                 done, "/", n, " (", 100.0_dp * real(done, 8) / real(max(n, 1), 8), "%) elapsed=", &
                 elapsed, " s"
            flush(output_unit)
            do while (next_progress <= done)
              next_progress = next_progress + progress_interval
            end do
          end if
          !$omp end critical(progress_write)
        end if
      end if
    end do
    !$omp end parallel do
    if (show_progress /= 0 .and. (progress_interval <= 0 .or. mod(n, progress_interval) /= 0)) then
      call system_clock(clock_now)
      elapsed = real(clock_now - pre_start, 8) / real(max(clock_rate, 1), 8)
      write(output_unit, '(A,I0,A,I0,A,F10.1,A)') "ShockFinder Fortran precompute: ", &
           n, "/", n, " (100.0%) elapsed=", elapsed, " s"
      flush(output_unit)
    end if

    call resolve_shock_centers(pos, dx, neighbors, fine_face_index, fine_neighbors, &
         n, nfine, candidate, divv_arr, &
         grad_t_arr, max_center_steps, center_normal_cosine, center_plateau_tolerance, &
         resolved_center, diagnostics(1))

    ! Candidate paths are independent after center resolution. Several paths
    ! can share a center, so the small critical section protects that write.
    progress_count = 0
    next_progress = progress_interval
    call system_clock(scan_start)

    !$omp parallel do schedule(dynamic, 256) private(i, step_count, &
    !$omp& center, trial, upstream, downstream, grad_t, t_pre, t_post, &
    !$omp& rho_pre, rho_post, ratio, m, dirvec, xwalk, xnext, ok_direction, &
    !$omp& endpoint_found, done, clock_now, elapsed)
    do i = 1, n
      if (show_progress /= 0 .and. progress_interval > 0) then
        !$omp atomic capture
        progress_count = progress_count + 1
        done = progress_count
        !$omp end atomic
        if (done >= next_progress) then
          !$omp critical(progress_write)
          if (done >= next_progress) then
            call system_clock(clock_now)
            elapsed = real(clock_now - scan_start, 8) / real(max(clock_rate, 1), 8)
            write(output_unit, '(A,I0,A,I0,A,F5.1,A,F10.1,A)') "ShockFinder Fortran scan: ", &
                 done, "/", n, " (", 100.0_dp * real(done, 8) / real(max(n, 1), 8), "%) elapsed=", &
                 elapsed, " s"
            flush(output_unit)
            do while (next_progress <= done)
              next_progress = next_progress + progress_interval
            end do
          end if
          !$omp end critical(progress_write)
        end if
      end if
      if (.not. candidate(i)) cycle

      center = resolved_center(i)
      if (center <= 0) cycle
      if (center /= i) cycle

      grad_t = grad_t_arr(center, :)
      call normalize_vector(grad_t, dirvec, ok_direction)
      if (.not. ok_direction) cycle

      upstream = center
      xwalk = pos(center, :)
      endpoint_found = .false.
      trial = -1
      do step_count = 1, max_steps
        call next_along_gradient(pos, dx, neighbors, fine_face_index, fine_neighbors, &
             n, nfine, upstream, xwalk, -dirvec, &
             trial, xnext)
        if (trial <= 0) then
          !$omp atomic update
          diagnostics(2) = diagnostics(2) + 1_8
          exit
        end if
        if (divv_arr(trial) >= 0.0_dp) then
          !$omp atomic update
          diagnostics(8) = diagnostics(8) + 1_8
          upstream = trial
          endpoint_found = .true.
          exit
        else if ((temp(trial) - temp(upstream)) * (rho(trial) - rho(upstream)) <= 0.0_dp) then
          !$omp atomic update
          diagnostics(7) = diagnostics(7) + 1_8
          upstream = trial
          endpoint_found = .true.
          exit
        else if (.not. candidate(trial)) then
          !$omp atomic update
          diagnostics(6) = diagnostics(6) + 1_8
          upstream = trial
          endpoint_found = .true.
          exit
        end if
        upstream = trial
        xwalk = xnext
      end do
      ! A valid endpoint is the first cell outside the shock zone. If the walk
      ! reaches its safety cap while still inside the zone, reject this center.
      if (.not. endpoint_found) then
        if (trial /= 0) then
          !$omp atomic update
          diagnostics(3) = diagnostics(3) + 1_8
        end if
        cycle
      end if

      downstream = center
      xwalk = pos(center, :)
      endpoint_found = .false.
      trial = -1
      do step_count = 1, max_steps
        call next_along_gradient(pos, dx, neighbors, fine_face_index, fine_neighbors, &
             n, nfine, downstream, xwalk, dirvec, &
             trial, xnext)
        if (trial <= 0) then
          !$omp atomic update
          diagnostics(4) = diagnostics(4) + 1_8
          exit
        end if
        if (divv_arr(trial) >= 0.0_dp) then
          !$omp atomic update
          diagnostics(8) = diagnostics(8) + 1_8
          downstream = trial
          endpoint_found = .true.
          exit
        else if ((temp(trial) - temp(downstream)) * (rho(trial) - rho(downstream)) <= 0.0_dp) then
          !$omp atomic update
          diagnostics(7) = diagnostics(7) + 1_8
          downstream = trial
          endpoint_found = .true.
          exit
        else if (.not. candidate(trial)) then
          !$omp atomic update
          diagnostics(6) = diagnostics(6) + 1_8
          downstream = trial
          endpoint_found = .true.
          exit
        end if
        downstream = trial
        xwalk = xnext
      end do
      if (.not. endpoint_found) then
        if (trial /= 0) then
          !$omp atomic update
          diagnostics(5) = diagnostics(5) + 1_8
        end if
        cycle
      end if

      t_pre = max(temp(upstream), temp_floor)
      t_post = temp(downstream)
      rho_pre = rho(upstream)
      rho_post = rho(downstream)

      if (t_post <= t_pre .or. rho_post <= rho_pre) then
        !$omp atomic update
        diagnostics(9) = diagnostics(9) + 1_8
        cycle
      end if

      ratio = t_post / t_pre
      m = mach_from_temperature_jump(ratio, gamma)
      if (m < min_mach * (1.0_dp - 1.0e-12_dp)) then
        !$omp atomic update
        diagnostics(10) = diagnostics(10) + 1_8
        cycle
      end if

      !$omp critical(shock_update)
      if (m > mach(center)) then
        mach(center) = m
        shock(center) = 1
        center_index(center) = center
        upstream_index(center) = upstream
        downstream_index(center) = downstream
      end if
      !$omp end critical(shock_update)
    end do
    !$omp end parallel do
    if (show_progress /= 0 .and. (progress_interval <= 0 .or. mod(n, progress_interval) /= 0)) then
      call system_clock(clock_now)
      elapsed = real(clock_now - scan_start, 8) / real(max(clock_rate, 1), 8)
      write(output_unit, '(A,I0,A,I0,A,F10.1,A)') "ShockFinder Fortran scan: ", &
           n, "/", n, " (100.0%) elapsed=", elapsed, " s"
      flush(output_unit)
    end if

    deallocate(divv_arr, grad_t_arr, candidate, resolved_center)
  end subroutine find_shocks

end module shockfinder_kernel
