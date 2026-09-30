! Post-processing of SAVED ShockFinder cells only. No shock detection/physics.
! Build both extensions for the running environment from the repository root:
! PYTHON=/path/to/python ./f2py.sh
!
! Spatial bins find neighbors without retaining edge lists. SciPy supplies
! bounded candidate pairs for geometries outside the packed-bin range.
! Both paths apply identical AMR cube-contact and axial-normal cuts.
! pos/normal have shape (3,n) so a transpose view of canonical NumPy (n,3)
! arrays can be borrowed without copying their full geometry.
! All indices and parent values are ZERO-BASED, matching the Python wrapper.
module merger_neighbor_kernel
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: merge_neighbor_pairs, connect_bucket, merge_component_labels, measure_geometry
contains
  ! Open-boundary front reductions. Read compact fields through row indices;
  ! no (m,3) geometry, area, or weight temporaries are allocated for a front.
  subroutine measure_geometry(pos, dx, normal, area, rows, margin, n, m, &
       area_sum, center, normal_sum, lower, upper, guarded_lower, guarded_upper)
    integer, intent(in) :: n, m, rows(m)
    real(8), intent(in) :: pos(3,n), dx(n), normal(3,n), area(n), margin
    real(8), intent(out) :: area_sum, center(3), normal_sum(3), lower(3), upper(3)
    real(8), intent(out) :: guarded_lower(3), guarded_upper(3)
    integer :: k, row, reference_row
    real(8) :: reference(3), sign_normal, weight, half_width, guard
    area_sum = 0.0d0
    center = 0.0d0
    normal_sum = 0.0d0
    lower = huge(0.0d0)
    upper = -huge(0.0d0)
    guarded_lower = lower
    guarded_upper = upper
    if (m == 0) return
    reference_row = rows(1)+1
    do k = 1, m
      row = rows(k)+1
      area_sum = area_sum + area(row)
      if (area(row) > area(reference_row)) reference_row = row
    end do
    reference = normal(:,reference_row)
    do k = 1, m
      row = rows(k)+1
      weight = area(row)/area_sum
      center = center + pos(:,row)*weight
      sign_normal = 1.0d0
      if (dot_product(normal(:,row), reference) < 0.0d0) sign_normal = -1.0d0
      normal_sum = normal_sum + normal(:,row)*(sign_normal*area(row))
      half_width = 0.5d0*dx(row)
      guard = (0.5d0+margin)*dx(row)
      lower = min(lower, pos(:,row)-half_width)
      upper = max(upper, pos(:,row)+half_width)
      guarded_lower = min(guarded_lower, pos(:,row)-guard)
      guarded_upper = max(guarded_upper, pos(:,row)+guard)
    end do
  end subroutine measure_geometry

  ! Generic spatial bins, not an assumption that shocks fill an AMR lattice.
  ! Each cell is placed in a bin at least as wide as the largest permitted
  ! reach for this target AMR bucket. Query at most 27 bins and apply the same
  ! exact contact/normal cuts as the SciPy backend. No edge list is allocated.
  ! status=1 requests the existing tree/pair fallback when packed bin keys
  ! cannot represent the geometry or allocation is unavailable. It returns
  ! before changing parent in that case.
  subroutine connect_bucket(pos, dx, normal, rows, source, target, parent, &
       gap_factor, normal_cosine, box_size, same_bucket, n, nvalid, nsource, ntarget, status)
    integer, intent(in) :: n, nvalid, nsource, ntarget, same_bucket
    real(8), intent(in) :: pos(3, n), dx(n), normal(3, n)
    integer, intent(in) :: rows(nvalid), source(nsource), target(ntarget)
    integer, intent(inout) :: parent(nvalid)
    real(8), intent(in) :: gap_factor, normal_cosine, box_size
    integer, intent(out) :: status
    integer(8), allocatable :: keys(:)
    integer, allocatable :: heads(:), next(:)
    integer(8) :: spans(3), bins(3), choices(3, 3), key, product
    integer :: k, i, j, row, axis, slot, table_size, allocation_status
    integer :: count(3), a, b, c, linked
    real(8) :: lower(3), upper(3), point(3), scaled(3), width, maximum_dx

    status = 0
    if (nsource == 0 .or. ntarget == 0) return
    maximum_dx = 0.0d0
    lower = huge(0.0d0)
    upper = -huge(0.0d0)
    do k = 1, ntarget
      row = rows(target(k) + 1) + 1
      maximum_dx = max(maximum_dx, dx(row))
      point = pos(:, row)
      if (box_size > 0.0d0) point = modulo(point, box_size)
      lower = min(lower, point)
      upper = max(upper, point)
    end do
    ! Padding protects broad-phase bin boundaries from rounding. The final
    ! scientific contact threshold is NEVER widened by this padding.
    width = (1.0d0 + gap_factor) * maximum_dx * 1.000001d0
    status = 1
    if (.not. ieee_is_finite(width) .or. width <= 0.0d0) return
    if (box_size > 0.0d0) then
      if (box_size / width > 1.0d9) return
      spans = max(1_8, floor(box_size / width, kind=8))
      width = box_size / real(spans(1), 8)
      lower = 0.0d0
    else
      scaled = (upper - lower) / width
      if (.not. all(ieee_is_finite(scaled))) return
      if (any(scaled > 1.0d9)) return
      spans = floor(scaled, kind=8) + 1_8
    end if
    product = 1_8
    do axis = 1, 3
      if (spans(axis) > huge(product) / product) return
      product = product * spans(axis)
    end do
    table_size = 1
    do while (int(table_size, 8) < 2_8 * int(ntarget, 8))
      if (table_size > huge(table_size) / 2) return
      table_size = table_size * 2
    end do
    allocate(keys(table_size), heads(table_size), next(ntarget), stat=allocation_status)
    if (allocation_status /= 0) return
    heads = 0
    next = 0
    do k = 1, ntarget
      row = rows(target(k) + 1) + 1
      point = pos(:, row)
      if (box_size > 0.0d0) point = modulo(point, box_size)
      bins = min(spans - 1_8, floor((point - lower) / width, kind=8))
      key = bins(1) + spans(1) * (bins(2) + spans(2) * bins(3))
      slot = bucket_slot(key, table_size)
      do while (heads(slot) /= 0)
        if (keys(slot) == key) exit
        slot = 1 + modulo(slot, table_size)
      end do
      keys(slot) = key
      next(k) = heads(slot)
      heads(slot) = k
    end do
    status = 0
    do k = 1, nsource
      i = source(k)
      row = rows(i + 1) + 1
      point = pos(:, row)
      if (box_size > 0.0d0) point = modulo(point, box_size)
      scaled = (point - lower) / width
      if (box_size <= 0.0d0) then
        ! A source in the bin AFTER the last occupied target bin can still
        ! contact a target near its upper edge. Keep that entire extra bin.
        if (any(scaled < -1.0d0) .or. any(scaled >= real(spans, 8) + 1.0d0)) cycle
      end if
      bins = floor(scaled, kind=8)
      if (box_size > 0.0d0) bins = min(spans - 1_8, bins)
      do axis = 1, 3
        count(axis) = 0
        do a = -1, 1
          if (box_size > 0.0d0) then
            if (a + 1 >= min(3_8, spans(axis))) exit
            choices(a + 2, axis) = modulo(bins(axis) + int(a, 8), spans(axis))
            count(axis) = count(axis) + 1
          else
            if (bins(axis) + a < 0_8 .or. bins(axis) + a >= spans(axis)) cycle
            count(axis) = count(axis) + 1
            choices(count(axis), axis) = bins(axis) + a
          end if
        end do
      end do
      do c = 1, count(3)
        do b = 1, count(2)
          do a = 1, count(1)
            key = choices(a, 1) + spans(1) * (choices(b, 2) + spans(2) * choices(c, 3))
            slot = bucket_slot(key, table_size)
            do while (heads(slot) /= 0)
              if (keys(slot) == key) exit
              slot = 1 + modulo(slot, table_size)
            end do
            if (heads(slot) == 0) cycle
            linked = heads(slot)
            do while (linked /= 0)
              j = target(linked)
              if (same_bucket == 0 .or. j > i) then
                call accept_pair(pos, dx, normal, rows, i, j, parent, gap_factor, &
                                 normal_cosine, box_size, n, nvalid)
              end if
              linked = next(linked)
            end do
          end do
        end do
      end do
    end do
    deallocate(keys, heads, next)
  end subroutine connect_bucket

  pure integer function bucket_slot(key, table_size) result(slot)
    integer(8), intent(in) :: key
    integer, intent(in) :: table_size
    integer(8) :: mixed
    ! Bit intrinsics avoid relying on signed integer multiplication overflow.
    mixed = ieor(key, ishft(key, -32))
    mixed = ieor(mixed, ishftc(mixed, 17))
    mixed = ieor(mixed, ishft(mixed, -13))
    slot = 1 + int(iand(mixed, int(table_size - 1, 8)))
  end function bucket_slot

  subroutine merge_neighbor_pairs(pos, dx, normal, rows, left, right, parent, &
       gap_factor, normal_cosine, box_size, same_bucket, n, nvalid, npairs)
    integer, intent(in) :: n, nvalid, npairs, same_bucket
    real(8), intent(in) :: pos(3, n), dx(n), normal(3, n)
    integer, intent(in) :: rows(nvalid), left(npairs), right(npairs)
    integer, intent(inout) :: parent(nvalid)
    real(8), intent(in) :: gap_factor, normal_cosine, box_size
    integer :: k, i, j
    do k = 1, npairs
      i = left(k)
      j = right(k)
      if (same_bucket /= 0 .and. j <= i) cycle
      call accept_pair(pos, dx, normal, rows, i, j, parent, gap_factor, &
                       normal_cosine, box_size, n, nvalid)
    end do
  end subroutine merge_neighbor_pairs

  ! Reconcile completed local graphs by existing global detected-cell indices.
  ! ALL ghost memberships are supplied. No geometry is accepted here and no
  ! workers write this parent: the caller reduces already verified components.
  subroutine merge_component_labels(cells, labels, parent, n, m)
    integer, intent(in) :: n, m, cells(m), labels(m)
    integer, intent(inout) :: parent(n)
    integer :: k, a, b, following
    do k = 1, m
      a = cells(k)
      b = labels(k)
      do
        following = parent(a+1)
        if (following == a) exit
        parent(a+1) = parent(following+1)
        a = parent(a+1)
      end do
      do
        following = parent(b+1)
        if (following == b) exit
        parent(b+1) = parent(following+1)
        b = parent(b+1)
      end do
      if (a < b) parent(b+1) = a
      if (b < a) parent(a+1) = b
    end do
  end subroutine merge_component_labels

  subroutine accept_pair(pos, dx, normal, rows, i, j, parent, gap_factor, &
                         normal_cosine, box_size, n, nvalid)
    integer, intent(in) :: n, nvalid, i, j, rows(nvalid)
    real(8), intent(in) :: pos(3, n), dx(n), normal(3, n)
    integer, intent(inout) :: parent(nvalid)
    real(8), intent(in) :: gap_factor, normal_cosine, box_size
    integer :: a, b, axis, root_a, root_b, next_root
    real(8) :: reach, delta, alignment
    logical :: close

      if (parent(i + 1) == parent(j + 1)) return
      a = rows(i + 1) + 1
      b = rows(j + 1) + 1
      reach = 0.5d0 * (dx(a) + dx(b)) + gap_factor * max(dx(a), dx(b))
      close = .true.
      do axis = 1, 3
        delta = pos(axis, a) - pos(axis, b)
        if (box_size > 0.0d0) delta = delta - box_size * dnint(delta / box_size)
        if (abs(delta) > reach) then
          close = .false.
          exit
        end if
      end do
      if (.not. close) return
      alignment = abs(normal(1, a) * normal(1, b) + &
                      normal(2, a) * normal(2, b) + normal(3, a) * normal(3, b))
      if (alignment < normal_cosine) return

      ! Path halving and a minimum-index root make components deterministic
      ! across candidate batch sizes, AMR levels and query traversal orders.
      root_a = i
      do
        next_root = parent(root_a + 1)
        if (next_root == root_a) exit
        parent(root_a + 1) = parent(next_root + 1)
        root_a = parent(root_a + 1)
      end do
      root_b = j
      do
        next_root = parent(root_b + 1)
        if (next_root == root_b) exit
        parent(root_b + 1) = parent(next_root + 1)
        root_b = parent(root_b + 1)
      end do
      if (root_a == root_b) return
      if (root_a < root_b) then
        parent(root_b + 1) = root_a
      else
        parent(root_a + 1) = root_b
      end if
  end subroutine accept_pair
end module merger_neighbor_kernel
