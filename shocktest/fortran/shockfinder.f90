module shockfinder_kernel
  use iso_fortran_env, only: output_unit
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  real(8), parameter :: entropy_exponent = 2.0_dp / 3.0_dp
  private
  public :: find_shocks

contains

  ! Return the entropy proxy T/rho^(2/3) used by the Skillman shock criteria.
  ! The exponent assumes an ideal monatomic gas with gamma = 5/3.
  pure real(8) function entropy_value(temp, rho) result(s)
    real(8), intent(in) :: temp, rho

    if (rho > 0.0_dp .and. temp > 0.0_dp) then
      s = temp / rho**entropy_exponent
    else
      s = 0.0_dp
    end if
  end function entropy_value

  ! Invert the Rankine-Hugoniot temperature jump for gamma = 5/3.
  ! Ratios below unity are treated as non-shocks with Mach 1.
  pure real(8) function mach_from_temperature_jump(t_ratio) result(mach)
    real(8), intent(in) :: t_ratio
    real(8) :: disc, m2

    if (t_ratio <= 1.0_dp) then
      mach = 1.0_dp
      return
    end if

    disc = (14.0_dp - 16.0_dp * t_ratio)**2 + 60.0_dp
    m2 = ((16.0_dp * t_ratio - 14.0_dp) + sqrt(disc)) / 10.0_dp
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
  pure subroutine face_sample(pos, vel, dx, temp, rho, neighbors, fine_neighbors, n, &
       i, face, axis, value_pos, value_vel, value_temp, value_entropy, ok)
    integer, intent(in) :: n, i, face, axis
    integer, intent(in) :: neighbors(n, 6), fine_neighbors(n, 6, 4)
    real(8), intent(in) :: pos(n, 3), vel(n, 3), dx(n), temp(n), rho(n)
    real(8), intent(out) :: value_pos, value_vel, value_temp, value_entropy
    logical, intent(out) :: ok

    integer :: nb, k, count
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
      value_entropy = entropy_value(temp(nb), rho(nb))
      ok = temp(nb) > 0.0_dp .and. rho(nb) > 0.0_dp
      return
    end if

    count = 0
    do k = 1, 4
      nb = fine_neighbors(i, face, k)
      if (nb <= 0) cycle
      if (temp(nb) <= 0.0_dp .or. rho(nb) <= 0.0_dp) cycle
      count = count + 1
      value_vel = value_vel + vel(nb, axis)
      value_temp = value_temp + temp(nb)
      value_entropy = value_entropy + entropy_value(temp(nb), rho(nb))
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
  pure subroutine local_quantities(pos, vel, dx, temp, rho, neighbors, fine_neighbors, n, i, &
       divv, grad_t, grad_s, valid)
    integer, intent(in) :: n, i
    integer, intent(in) :: neighbors(n, 6), fine_neighbors(n, 6, 4)
    real(8), intent(in) :: pos(n, 3), vel(n, 3), dx(n), temp(n), rho(n)
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
      call face_sample(pos, vel, dx, temp, rho, neighbors, fine_neighbors, n, &
           i, minus_face(axis), axis, xm, vm, tm, sm, okm)
      call face_sample(pos, vel, dx, temp, rho, neighbors, fine_neighbors, n, &
           i, plus_face(axis), axis, xp, vp, tp, sp, okp)
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
  pure integer function choose_face_neighbor(pos, neighbors, fine_neighbors, n, i, face, xpoint) result(nb)
    integer, intent(in) :: n, i, face
    integer, intent(in) :: neighbors(n, 6), fine_neighbors(n, 6, 4)
    real(8), intent(in) :: pos(n, 3), xpoint(3)

    integer :: k, trial
    real(8) :: best, dist2

    nb = neighbors(i, face)
    if (nb > 0) return

    best = huge(1.0_dp)
    do k = 1, 4
      trial = fine_neighbors(i, face, k)
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
  pure subroutine next_along_gradient(pos, dx, neighbors, fine_neighbors, n, i, xold, direction, &
       next_cell, xnew)
    integer, intent(in) :: n, i
    integer, intent(in) :: neighbors(n, 6), fine_neighbors(n, 6, 4)
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
    next_cell = choose_face_neighbor(pos, neighbors, fine_neighbors, n, i, face, xnew)
  end subroutine next_along_gradient

  ! Scan AMR cells for Skillman-style shock zones and assign center Mach numbers.
  ! Neighbor tables are supplied by Python/Numba and the cell loop is OpenMP-ready.
  subroutine find_shocks(pos, vel, dx, temp, rho, level, neighbors, fine_neighbors, n, &
       temp_floor, min_mach, max_steps, show_progress, progress_interval, mach, &
       shock, center_index, upstream_index, downstream_index)
    !f2py intent(in) pos, vel, dx, temp, rho, level, neighbors, fine_neighbors, n
    !f2py intent(in) temp_floor, min_mach, max_steps, show_progress, progress_interval
    !f2py intent(out) mach, shock, center_index, upstream_index, downstream_index
    integer, intent(in) :: n, max_steps, show_progress, progress_interval
    integer, intent(in) :: level(n), neighbors(n, 6), fine_neighbors(n, 6, 4)
    real(8), intent(in) :: pos(n, 3), vel(n, 3), dx(n), temp(n), rho(n)
    real(8), intent(in) :: temp_floor, min_mach
    real(8), intent(out) :: mach(n)
    integer, intent(out) :: shock(n), center_index(n), upstream_index(n), downstream_index(n)

    integer :: i, face, nb, k, step_count, done
    integer :: center, trial, upstream, downstream
    integer :: progress_count, next_progress
    integer :: clock_rate, clock_now, pre_start, scan_start
    real(8) :: best_divv, grad_t(3), grad_s(3), dirvec(3), xwalk(3), xnext(3)
    real(8) :: t_pre, t_post, rho_pre, rho_post, ratio, m, elapsed
    real(8), allocatable :: divv_arr(:), grad_t_arr(:, :)
    logical, allocatable :: candidate(:)
    logical :: valid, ok_direction

    mach = 0.0_dp
    shock = 0
    center_index = 0
    upstream_index = 0
    downstream_index = 0

    allocate(divv_arr(n), grad_t_arr(n, 3), candidate(n))

    ! Precompute all local shock diagnostics once. This loop is independent for
    ! each AMR cell and is therefore a good OpenMP target.
    progress_count = 0
    next_progress = progress_interval
    call system_clock(count_rate=clock_rate)
    call system_clock(pre_start)

    !$omp parallel do schedule(static) private(i, done, grad_s, valid, clock_now, elapsed)
    do i = 1, n
      call local_quantities(pos, vel, dx, temp, rho, neighbors, fine_neighbors, n, i, &
           divv_arr(i), grad_t_arr(i, :), grad_s, valid)
      candidate(i) = valid .and. divv_arr(i) < 0.0_dp .and. &
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

    ! Candidate cells are independent except when several cells choose the same
    ! maximum-convergence center. The small critical section protects that write.
    progress_count = 0
    next_progress = progress_interval
    call system_clock(scan_start)

    !$omp parallel do schedule(dynamic, 256) private(i, face, nb, k, step_count, &
    !$omp& center, trial, upstream, downstream, best_divv, grad_t, t_pre, t_post, &
    !$omp& rho_pre, rho_post, ratio, m, dirvec, xwalk, xnext, ok_direction, done, clock_now, elapsed)
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

      center = i
      best_divv = divv_arr(i)
      grad_t = grad_t_arr(i, :)
      do face = 1, 6
        nb = neighbors(i, face)
        if (nb > 0) then
          if (candidate(nb) .and. divv_arr(nb) < best_divv) then
            center = nb
            best_divv = divv_arr(nb)
            grad_t = grad_t_arr(nb, :)
          end if
        end if

        do k = 1, 4
          nb = fine_neighbors(i, face, k)
          if (nb <= 0) cycle
          if (candidate(nb) .and. divv_arr(nb) < best_divv) then
            center = nb
            best_divv = divv_arr(nb)
            grad_t = grad_t_arr(nb, :)
          end if
        end do
      end do

      call normalize_vector(grad_t, dirvec, ok_direction)
      if (.not. ok_direction) cycle

      upstream = center
      xwalk = pos(center, :)
      do step_count = 1, max_steps
        call next_along_gradient(pos, dx, neighbors, fine_neighbors, n, upstream, xwalk, -dirvec, &
             trial, xnext)
        if (trial <= 0) exit
        if (.not. candidate(trial)) exit
        upstream = trial
        xwalk = xnext
      end do
      call next_along_gradient(pos, dx, neighbors, fine_neighbors, n, upstream, xwalk, -dirvec, &
           trial, xnext)
      upstream = trial
      if (upstream <= 0) cycle

      downstream = center
      xwalk = pos(center, :)
      do step_count = 1, max_steps
        call next_along_gradient(pos, dx, neighbors, fine_neighbors, n, downstream, xwalk, dirvec, &
             trial, xnext)
        if (trial <= 0) exit
        if (.not. candidate(trial)) exit
        downstream = trial
        xwalk = xnext
      end do
      call next_along_gradient(pos, dx, neighbors, fine_neighbors, n, downstream, xwalk, dirvec, &
           trial, xnext)
      downstream = trial
      if (downstream <= 0) cycle

      t_pre = max(temp(upstream), temp_floor)
      t_post = temp(downstream)
      rho_pre = rho(upstream)
      rho_post = rho(downstream)

      if (t_post <= t_pre) cycle
      if (rho_post <= rho_pre) cycle

      ratio = t_post / t_pre
      m = mach_from_temperature_jump(ratio)
      if (m < min_mach) cycle

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

    deallocate(divv_arr, grad_t_arr, candidate)
  end subroutine find_shocks

end module shockfinder_kernel
