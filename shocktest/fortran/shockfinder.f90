module shockfinder_kernel
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  private
  public :: find_shocks

contains

  pure real(8) function entropy_value(temp, rho, gamma) result(s)
    real(8), intent(in) :: temp, rho, gamma

    if (rho > 0.0_dp .and. temp > 0.0_dp) then
      s = temp / rho**(gamma - 1.0_dp)
    else
      s = 0.0_dp
    end if
  end function entropy_value

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

  pure integer function plus_face(axis) result(face)
    integer, intent(in) :: axis
    face = axis * 2
  end function plus_face

  pure integer function minus_face(axis) result(face)
    integer, intent(in) :: axis
    face = axis * 2 - 1
  end function minus_face

  pure integer function face_for_step(axis, direction) result(face)
    integer, intent(in) :: axis, direction
    if (direction >= 0) then
      face = plus_face(axis)
    else
      face = minus_face(axis)
    end if
  end function face_for_step

  pure subroutine local_quantities(pos, vel, dx, temp, rho, neighbors, n, gamma, i, &
       divv, grad_t, grad_s, valid)
    integer, intent(in) :: n, i
    integer, intent(in) :: neighbors(n, 6)
    real(8), intent(in) :: pos(n, 3), vel(n, 3), dx(n), temp(n), rho(n), gamma
    real(8), intent(out) :: divv, grad_t(3), grad_s(3)
    logical, intent(out) :: valid

    integer :: axis, im, ip
    real(8) :: dist, sm, sp

    divv = 0.0_dp
    grad_t = 0.0_dp
    grad_s = 0.0_dp
    valid = .false.

    if (i < 1 .or. i > n) return
    if (rho(i) <= 0.0_dp .or. temp(i) <= 0.0_dp .or. dx(i) <= 0.0_dp) return

    do axis = 1, 3
      im = neighbors(i, minus_face(axis))
      ip = neighbors(i, plus_face(axis))
      if (im <= 0 .or. ip <= 0) cycle
      dist = pos(ip, axis) - pos(im, axis)
      if (abs(dist) <= 0.0_dp) cycle

      sm = entropy_value(temp(im), rho(im), gamma)
      sp = entropy_value(temp(ip), rho(ip), gamma)
      grad_t(axis) = (temp(ip) - temp(im)) / dist
      grad_s(axis) = (sp - sm) / dist
      divv = divv + (vel(ip, axis) - vel(im, axis)) / dist
      valid = .true.
    end do
  end subroutine local_quantities

  pure logical function cell_is_candidate(pos, vel, dx, temp, rho, neighbors, n, gamma, i)
    integer, intent(in) :: n, i
    integer, intent(in) :: neighbors(n, 6)
    real(8), intent(in) :: pos(n, 3), vel(n, 3), dx(n), temp(n), rho(n), gamma
    real(8) :: divv, grad_t(3), grad_s(3)
    logical :: valid

    call local_quantities(pos, vel, dx, temp, rho, neighbors, n, gamma, i, &
         divv, grad_t, grad_s, valid)
    cell_is_candidate = valid .and. divv < 0.0_dp .and. dot_product(grad_t, grad_s) > 0.0_dp
  end function cell_is_candidate

  pure integer function dominant_axis(grad_t) result(axis)
    real(8), intent(in) :: grad_t(3)
    real(8) :: best
    integer :: a

    axis = 0
    best = 0.0_dp
    do a = 1, 3
      if (abs(grad_t(a)) > best) then
        best = abs(grad_t(a))
        axis = a
      end if
    end do
  end function dominant_axis

  subroutine find_shocks(pos, vel, dx, temp, rho, level, neighbors, n, gamma, &
       temp_floor, min_mach, max_steps, mach, shock, center_index, upstream_index, &
       downstream_index)
    !f2py intent(in) pos, vel, dx, temp, rho, level, neighbors, n
    !f2py intent(in) gamma, temp_floor, min_mach, max_steps
    !f2py intent(out) mach, shock, center_index, upstream_index, downstream_index
    integer, intent(in) :: n, max_steps
    integer, intent(in) :: level(n), neighbors(n, 6)
    real(8), intent(in) :: pos(n, 3), vel(n, 3), dx(n), temp(n), rho(n)
    real(8), intent(in) :: gamma, temp_floor, min_mach
    real(8), intent(out) :: mach(n)
    integer, intent(out) :: shock(n), center_index(n), upstream_index(n), downstream_index(n)

    integer :: i, face, nb, axis, direction, step_count
    integer :: center, trial, upstream, downstream
    real(8) :: best_divv, grad_t(3)
    real(8) :: t_pre, t_post, rho_pre, rho_post, ratio, m
    real(8), allocatable :: divv_arr(:), grad_t_arr(:, :), grad_s_arr(:, :)
    logical, allocatable :: valid_arr(:), candidate(:)

    mach = 0.0_dp
    shock = 0
    center_index = 0
    upstream_index = 0
    downstream_index = 0

    allocate(divv_arr(n), grad_t_arr(n, 3), grad_s_arr(n, 3), valid_arr(n), candidate(n))

    !$omp parallel do schedule(static) private(i)
    do i = 1, n
      call local_quantities(pos, vel, dx, temp, rho, neighbors, n, gamma, i, &
           divv_arr(i), grad_t_arr(i, :), grad_s_arr(i, :), valid_arr(i))
      candidate(i) = valid_arr(i) .and. divv_arr(i) < 0.0_dp .and. &
           dot_product(grad_t_arr(i, :), grad_s_arr(i, :)) > 0.0_dp
    end do
    !$omp end parallel do

    !$omp parallel do schedule(dynamic, 256) private(i, face, nb, axis, direction, step_count, &
    !$omp& center, trial, upstream, downstream, best_divv, grad_t, t_pre, t_post, &
    !$omp& rho_pre, rho_post, ratio, m)
    do i = 1, n
      if (.not. candidate(i)) cycle

      center = i
      best_divv = divv_arr(i)
      grad_t = grad_t_arr(i, :)
      do face = 1, 6
        nb = neighbors(i, face)
        if (nb <= 0) cycle
        if (candidate(nb) .and. divv_arr(nb) < best_divv) then
          center = nb
          best_divv = divv_arr(nb)
          grad_t = grad_t_arr(nb, :)
        end if
      end do

      axis = dominant_axis(grad_t)
      if (axis <= 0) cycle
      if (grad_t(axis) >= 0.0_dp) then
        direction = 1
      else
        direction = -1
      end if

      upstream = center
      do step_count = 1, max_steps
        trial = neighbors(upstream, face_for_step(axis, -direction))
        if (trial <= 0) exit
        if (.not. candidate(trial)) exit
        upstream = trial
      end do
      upstream = neighbors(upstream, face_for_step(axis, -direction))
      if (upstream <= 0) cycle

      downstream = center
      do step_count = 1, max_steps
        trial = neighbors(downstream, face_for_step(axis, direction))
        if (trial <= 0) exit
        if (.not. candidate(trial)) exit
        downstream = trial
      end do
      downstream = neighbors(downstream, face_for_step(axis, direction))
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

    deallocate(divv_arr, grad_t_arr, grad_s_arr, valid_arr, candidate)
  end subroutine find_shocks

end module shockfinder_kernel
