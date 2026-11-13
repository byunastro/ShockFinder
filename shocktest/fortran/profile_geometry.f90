! Stateless, read-only AMR geometry kernels for the profile estimator.
! Density/momentum restriction, CIC stencil order and boundary rules mirror
! profile_geometry.py. No fast-math, global mutable state or periodic wrapping.
module profile_geometry_kernel
  use, intrinsic :: ieee_arithmetic
  implicit none
contains
  integer function slot(q,lev,mask) result(k)
    integer(8),intent(in)::q(3),mask
    integer,intent(in)::lev
    integer(8)::v
    v=iand(q(1),mask)*73856093_8+iand(q(2),mask)*19349663_8 &
      +iand(q(3),mask)*83492791_8+int(lev,8)*104729_8
    k=int(iand(v,mask))+1
  end function

  subroutine build_index(q,levels,nt,n,table,duplicate)
    integer,intent(in)::n,nt,levels(n)
    integer(8),intent(in)::q(n,3)
    integer,intent(out)::table(nt),duplicate
    integer::i,k,j
    table=0;duplicate=0
    do i=1,n
      k=slot(q(i,:),levels(i),int(nt-1,8))
      do
        j=table(k)
        if(j==0)exit
        if(levels(j)==levels(i).and.all(q(j,:)==q(i,:)))then
          duplicate=i;return
        endif
        k=mod(k,nt)+1
      enddo
      table(k)=i
    enddo
  end subroutine

  integer function locate_one(point,origin,widths,q,levels,table,n,nw,nt) result(id)
    integer,intent(in)::n,nw,nt,levels(n),table(nt)
    integer(8),intent(in)::q(n,3)
    real(8),intent(in)::point(3),origin(3),widths(nw)
    real(8)::c(3),nearest(3)
    integer(8)::wanted(3)
    integer::lev,k,j
    id=0
    if(.not.all(ieee_is_finite(point)))return
    do lev=1,nw
      c=(point-origin)/widths(lev);nearest=anint(c)
      where(abs(c-nearest)<2d-12)c=nearest
      if(any(abs(c)>9d18))return
      wanted=floor(c,kind=8);k=slot(wanted,lev,int(nt-1,8))
      do
        j=table(k)
        if(j==0)exit
        if(levels(j)==lev.and.all(q(j,:)==wanted))then
          id=j;return
        endif
        k=mod(k,nt)+1
      enddo
    enddo
  end function

  subroutine locate_batch(points,origin,widths,q,levels,table,n,nw,nt,ns,ids)
    integer,intent(in)::n,nw,nt,ns,levels(n),table(nt)
    integer(8),intent(in)::q(n,3)
    real(8),intent(in)::points(ns,3),origin(3),widths(nw)
    integer,intent(out)::ids(ns)
    integer::i
    ! Tiny final-center lookups do not amortize an OpenMP team launch.
    !$omp parallel do private(i) schedule(static) if(ns>=512)
    do i=1,ns
      ids(i)=locate_one(points(i,:),origin,widths,q,levels,table,n,nw,nt)
    enddo
    !$omp end parallel do
  end subroutine

  recursive subroutine virtual_one(point,h,origin,widths,dx,q,levels,table,values, &
                                  n,nw,nt,nv,depth,out)
    integer,intent(in)::n,nw,nt,nv,depth,levels(n),table(nt)
    integer(8),intent(in)::q(n,3)
    real(8),intent(in)::point(3),h,origin(3),widths(nw),dx(n),values(n,nv)
    real(8),intent(out)::out(nv)
    real(8)::child(3),v(nv)
    integer::id,a,b,c
    out=ieee_value(0d0,ieee_quiet_nan)
    if(depth>30)return
    id=locate_one(point,origin,widths,q,levels,table,n,nw,nt)
    if(id==0)return
    if(dx(id)>=h*(1d0-1d-12))then
      out=values(id,:);return
    endif
    out=0d0
    do a=-1,1,2
      do b=-1,1,2
        do c=-1,1,2
          child=point+.25d0*h*real([a,b,c],8)
          call virtual_one(child,h/2d0,origin,widths,dx,q,levels,table,values, &
                           n,nw,nt,nv,depth+1,v)
          out=out+v/8d0
        enddo
      enddo
    enddo
  end subroutine

  subroutine sample_batch(points,h,origin,widths,dx,q,levels,table,values, &
                          n,nw,nt,nv,ns,out)
    integer,intent(in)::n,nw,nt,nv,ns,levels(n),table(nt)
    integer(8),intent(in)::q(n,3)
    real(8),intent(in)::points(ns,3),h(ns),origin(3),widths(nw),dx(n),values(n,nv)
    real(8),intent(out)::out(ns,nv)
    real(8)::coordinate(3),near(3),low(3),fraction(3),node(3),w(3),weight,v(nv)
    integer::i,a,b,c,bit(3),id
    !$omp parallel do private(i,a,b,c,bit,id,coordinate,near,low,fraction,node,w,weight,v) schedule(static)
    do i=1,ns
      out(i,:)=0d0
      if(.not.all(ieee_is_finite(points(i,:))))then
        out(i,:)=ieee_value(0d0,ieee_quiet_nan);cycle
      endif
      coordinate=(points(i,:)-origin)/h(i)-.5d0;near=anint(coordinate)
      where(abs(coordinate-near)<2d-12)coordinate=near
      low=real(floor(coordinate,kind=8),8);fraction=coordinate-low
      do a=0,1
        do b=0,1
          do c=0,1
            bit=[a,b,c];w=1d0-fraction
            where(bit==1)w=fraction
            weight=(w(1)*w(2))*w(3)
            if(weight<=1d-14)cycle
            node=origin+(low+real(bit,8)+.5d0)*h(i)
            call virtual_one(node,h(i),origin,widths,dx,q,levels,table,values,n,nw,nt,nv,0,v)
            out(i,:)=out(i,:)+weight*v
          enddo
        enddo
      enddo
      id=locate_one(points(i,:),origin,widths,q,levels,table,n,nw,nt)
      if(id==0)out(i,:)=ieee_value(0d0,ieee_quiet_nan)
    enddo
    !$omp end parallel do
  end subroutine

  subroutine sample_batch_cached(points,h,origin,widths,dx,q,levels,table,values, &
                                 n,nw,nt,nv,ns,out,nunique,nreference)
    ! Reuse exact virtual-cell evaluations within one bounded sampling batch.
    ! Index creation is serial; evaluation and interpolation have disjoint writes.
    ! Keep the original corner/restriction accumulation order and boundary rules.
    integer,intent(in)::n,nw,nt,nv,ns,levels(n),table(nt)
    integer(8),intent(in)::q(n,3)
    real(8),intent(in)::points(ns,3),h(ns),origin(3),widths(nw),dx(n),values(n,nv)
    real(8),intent(out)::out(ns,nv)
    integer,intent(out)::nunique,nreference
    integer,allocatable::lookup(:),refs(:,:)
    integer(8),allocatable::keys(:,:)
    real(8),allocatable::nodes(:,:),scales(:),weights(:,:),cached(:,:)
    real(8)::coordinate(3),near(3),low(3),fraction(3),w(3),weight
    integer(8)::key(4),mask,hkey,hash
    integer::i,a,b,c,bit(3),id,k,j,corner,capacity
    nunique=0;nreference=0;out=0d0
    if(ns==0)return
    capacity=1
    do while(capacity<16*ns)
      capacity=capacity*2
    enddo
    mask=int(capacity-1,8)
    allocate(lookup(capacity),refs(8,ns),keys(4,8*ns),nodes(3,8*ns), &
             scales(8*ns),weights(8,ns),cached(nv,8*ns))
    lookup=0;refs=0;weights=0d0
    do i=1,ns
      if(.not.all(ieee_is_finite(points(i,:))))cycle
      coordinate=(points(i,:)-origin)/h(i)-.5d0;near=anint(coordinate)
      where(abs(coordinate-near)<2d-12)coordinate=near
      low=real(floor(coordinate,kind=8),8);fraction=coordinate-low
      corner=0
      do a=0,1
        do b=0,1
          do c=0,1
            corner=corner+1;bit=[a,b,c];w=1d0-fraction
            where(bit==1)w=fraction
            weight=(w(1)*w(2))*w(3)
            if(weight<=1d-14)cycle
            key(1:3)=int(low,8)+int(bit,8);key(4)=transfer(h(i),key(4))
            hkey=ieor(key(4),ieor(ishft(key(4),-32),ishft(key(4),-48)))
            hash=iand(key(1),mask)*73856093_8+iand(key(2),mask)*19349663_8 &
                +iand(key(3),mask)*83492791_8+iand(hkey,mask)*104729_8
            k=int(iand(hash,mask))+1
            do
              j=lookup(k)
              if(j==0)then
                nunique=nunique+1;j=nunique;lookup(k)=j;keys(:,j)=key
                nodes(:,j)=origin+(low+real(bit,8)+.5d0)*h(i);scales(j)=h(i)
                exit
              endif
              if(all(keys(:,j)==key))exit
              k=mod(k,capacity)+1
            enddo
            refs(corner,i)=j;weights(corner,i)=weight;nreference=nreference+1
          enddo
        enddo
      enddo
    enddo
    !$omp parallel do private(j) schedule(static) if(nunique>=256)
    do j=1,nunique
      call virtual_one(nodes(:,j),scales(j),origin,widths,dx,q,levels,table,values, &
                       n,nw,nt,nv,0,cached(:,j))
    enddo
    !$omp end parallel do
    !$omp parallel do private(i,corner,j,id) schedule(static) if(ns>=1024)
    do i=1,ns
      if(.not.all(ieee_is_finite(points(i,:))))then
        out(i,:)=ieee_value(0d0,ieee_quiet_nan);cycle
      endif
      do corner=1,8
        j=refs(corner,i)
        if(j/=0)out(i,:)=out(i,:)+weights(corner,i)*cached(:,j)
      enddo
      id=locate_one(points(i,:),origin,widths,q,levels,table,n,nw,nt)
      if(id==0)out(i,:)=ieee_value(0d0,ieee_quiet_nan)
    enddo
    !$omp end parallel do
    deallocate(lookup,refs,keys,nodes,scales,weights,cached)
  end subroutine
  subroutine sample_batch_reuse(points,h,origin,widths,dx,q,levels,table,values, &
                                 cache_keys,cache_valid,cache_values,nc, &
                                 n,nw,nt,nv,ns,out,nunique,nreference,nhit)
    ! Persist exact virtual cells across batches in a caller-owned bounded cache.
    ! Index creation is serial; evaluation and interpolation have disjoint writes.
    ! Keep the original corner/restriction accumulation order and boundary rules.
    integer,intent(in)::n,nw,nt,nv,ns,levels(n),table(nt)
    integer(8),intent(in)::q(n,3)
    real(8),intent(in)::points(ns,3),h(ns),origin(3),widths(nw),dx(n),values(n,nv)
    real(8),intent(out)::out(ns,nv)
    integer,intent(out)::nunique,nreference,nhit
    integer,intent(in)::nc
    integer(8),intent(inout)::cache_keys(4,nc)
    integer,intent(inout)::cache_valid(nc)
    real(8),intent(inout)::cache_values(nv,nc)
    integer,allocatable::persistent_slots(:)
    logical,allocatable::hit(:)
    integer(8)::pmask
    integer::pk
    integer,allocatable::lookup(:),refs(:,:)
    integer(8),allocatable::keys(:,:)
    real(8),allocatable::nodes(:,:),scales(:),weights(:,:),cached(:,:)
    real(8)::coordinate(3),near(3),low(3),fraction(3),w(3),weight
    integer(8)::key(4),mask,hkey,hash
    integer::i,a,b,c,bit(3),id,k,j,corner,capacity
    nunique=0;nreference=0;nhit=0;out=0d0
    if(ns==0)return
    capacity=1
    do while(capacity<16*ns)
      capacity=capacity*2
    enddo
    mask=int(capacity-1,8)
    allocate(lookup(capacity),refs(8,ns),keys(4,8*ns),nodes(3,8*ns), &
             scales(8*ns),weights(8,ns),cached(nv,8*ns))
    lookup=0;refs=0;weights=0d0
    do i=1,ns
      if(.not.all(ieee_is_finite(points(i,:))))cycle
      coordinate=(points(i,:)-origin)/h(i)-.5d0;near=anint(coordinate)
      where(abs(coordinate-near)<2d-12)coordinate=near
      low=real(floor(coordinate,kind=8),8);fraction=coordinate-low
      corner=0
      do a=0,1
        do b=0,1
          do c=0,1
            corner=corner+1;bit=[a,b,c];w=1d0-fraction
            where(bit==1)w=fraction
            weight=(w(1)*w(2))*w(3)
            if(weight<=1d-14)cycle
            key(1:3)=int(low,8)+int(bit,8);key(4)=transfer(h(i),key(4))
            hkey=ieor(key(4),ieor(ishft(key(4),-32),ishft(key(4),-48)))
            hash=iand(key(1),mask)*73856093_8+iand(key(2),mask)*19349663_8 &
                +iand(key(3),mask)*83492791_8+iand(hkey,mask)*104729_8
            k=int(iand(hash,mask))+1
            do
              j=lookup(k)
              if(j==0)then
                nunique=nunique+1;j=nunique;lookup(k)=j;keys(:,j)=key
                nodes(:,j)=origin+(low+real(bit,8)+.5d0)*h(i);scales(j)=h(i)
                exit
              endif
              if(all(keys(:,j)==key))exit
              k=mod(k,capacity)+1
            enddo
            refs(corner,i)=j;weights(corner,i)=weight;nreference=nreference+1
          enddo
        enddo
      enddo
    enddo
    allocate(persistent_slots(nunique),hit(nunique))
    pmask=int(nc-1,8)
    do j=1,nunique
      key=keys(:,j)
      hkey=ieor(key(4),ieor(ishft(key(4),-32),ishft(key(4),-48)))
      hash=iand(key(1),pmask)*73856093_8+iand(key(2),pmask)*19349663_8 &
          +iand(key(3),pmask)*83492791_8+iand(hkey,pmask)*104729_8
      pk=int(iand(hash,pmask))+1;persistent_slots(j)=pk
      hit(j)=cache_valid(pk)/=0
      if(hit(j))hit(j)=all(cache_keys(:,pk)==key)
      if(hit(j))then
        cached(:,j)=cache_values(:,pk);nhit=nhit+1
      endif
    enddo
    !$omp parallel do private(j) schedule(static) if(nunique>=256)
    do j=1,nunique
      if(hit(j))cycle
      call virtual_one(nodes(:,j),scales(j),origin,widths,dx,q,levels,table,values, &
                       n,nw,nt,nv,0,cached(:,j))
    enddo
    !$omp end parallel do
    ! Commit only after the parallel reads; direct-map collisions cannot race.
    do j=1,nunique
      pk=persistent_slots(j)
      cache_keys(:,pk)=keys(:,j);cache_values(:,pk)=cached(:,j);cache_valid(pk)=1
    enddo
    deallocate(persistent_slots,hit)
    !$omp parallel do private(i,corner,j,id) schedule(static) if(ns>=1024)
    do i=1,ns
      if(.not.all(ieee_is_finite(points(i,:))))then
        out(i,:)=ieee_value(0d0,ieee_quiet_nan);cycle
      endif
      do corner=1,8
        j=refs(corner,i)
        if(j/=0)out(i,:)=out(i,:)+weights(corner,i)*cached(:,j)
      enddo
      id=locate_one(points(i,:),origin,widths,q,levels,table,n,nw,nt)
      if(id==0)out(i,:)=ieee_value(0d0,ieee_quiet_nan)
    enddo
    !$omp end parallel do
    deallocate(lookup,refs,keys,nodes,scales,weights,cached)
  end subroutine

  subroutine derivatives(pos,dx,temperature,velocity,pressure,origin,widths,q,levels,table, &
                         n,nw,nt,div,gt,gp)
    integer,intent(in)::n,nw,nt,levels(n),table(nt)
    integer(8),intent(in)::q(n,3)
    real(8),intent(in)::pos(n,3),dx(n),temperature(n),velocity(n,3),pressure(n),origin(3),widths(nw)
    real(8),intent(out)::div(n),gt(n,3),gp(n,3)
    real(8)::sides(2,6),summed(6),probe(3),distance
    integer::i,axis,t1,t2,side,sign,a,b,id,count
    div=0d0;gt=0d0;gp=0d0
    !$omp parallel do private(i,axis,t1,t2,side,sign,a,b,id,count,sides,summed,probe,distance) schedule(static)
    do i=1,n
      do axis=1,3
        t1=1;t2=2
        if(axis==1)then
          t1=2;t2=3
        else if(axis==2)then
          t2=3
        endif
        do side=1,2
          sign=2*side-3;summed=0d0;count=0
          do a=-1,1,2
            do b=-1,1,2
              probe=pos(i,:);probe(axis)=probe(axis)+sign*(.5d0+1d-7)*dx(i)
              probe(t1)=probe(t1)+a*.25d0*dx(i);probe(t2)=probe(t2)+b*.25d0*dx(i)
              id=locate_one(probe,origin,widths,q,levels,table,n,nw,nt)
              if(id==0)cycle
              summed(1)=summed(1)+pos(id,axis)
              summed(2)=summed(2)+temperature(id);summed(3)=summed(3)+pressure(id)
              summed(4:6)=summed(4:6)+velocity(id,:);count=count+1
            enddo
          enddo
          sides(side,:)=ieee_value(0d0,ieee_quiet_nan)
          if(count>0)sides(side,:)=summed/count
        enddo
        distance=sides(2,1)-sides(1,1)
        if(.not.ieee_is_finite(distance).or.distance<=0d0)cycle
        gt(i,axis)=(sides(2,2)-sides(1,2))/distance
        gp(i,axis)=(sides(2,3)-sides(1,3))/distance
        div(i)=div(i)+(sides(2,3+axis)-sides(1,3+axis))/distance
      enddo
    enddo
    !$omp end parallel do
  end subroutine

  subroutine derivatives_at(pos,dx,temperature,velocity,pressure,origin,widths,q,levels,table, &
                         n,nw,nt,indices,ns,div,gt,gp)
    integer,intent(in)::n,nw,nt,ns,indices(ns),levels(n),table(nt)
    integer(8),intent(in)::q(n,3)
    real(8),intent(in)::pos(n,3),dx(n),temperature(n),velocity(n,3),pressure(n),origin(3),widths(nw)
    real(8),intent(out)::div(ns),gt(ns,3),gp(ns,3)
    real(8)::sides(2,6),summed(6),probe(3),distance
    integer::i,j,axis,t1,t2,side,sign,a,b,id,count
    div=0d0;gt=0d0;gp=0d0
    !$omp parallel do private(i,j,axis,t1,t2,side,sign,a,b,id,count,sides,summed,probe,distance) &
    !$omp schedule(static) if(ns>=128)
    do j=1,ns
      i=indices(j)
      do axis=1,3
        t1=1;t2=2
        if(axis==1)then
          t1=2;t2=3
        else if(axis==2)then
          t2=3
        endif
        do side=1,2
          sign=2*side-3;summed=0d0;count=0
          do a=-1,1,2
            do b=-1,1,2
              probe=pos(i,:);probe(axis)=probe(axis)+sign*(.5d0+1d-7)*dx(i)
              probe(t1)=probe(t1)+a*.25d0*dx(i);probe(t2)=probe(t2)+b*.25d0*dx(i)
              id=locate_one(probe,origin,widths,q,levels,table,n,nw,nt)
              if(id==0)cycle
              summed(1)=summed(1)+pos(id,axis)
              summed(2)=summed(2)+temperature(id);summed(3)=summed(3)+pressure(id)
              summed(4:6)=summed(4:6)+velocity(id,:);count=count+1
            enddo
          enddo
          sides(side,:)=ieee_value(0d0,ieee_quiet_nan)
          if(count>0)sides(side,:)=summed/count
        enddo
        distance=sides(2,1)-sides(1,1)
        if(.not.ieee_is_finite(distance).or.distance<=0d0)cycle
        gt(j,axis)=(sides(2,2)-sides(1,2))/distance
        gp(j,axis)=(sides(2,3)-sides(1,3))/distance
        div(j)=div(j)+(sides(2,3+axis)-sides(1,3+axis))/distance
      enddo
    enddo
    !$omp end parallel do
  end subroutine
end module

! Background fitting and conservative mass roots. Small SVD pseudo-inverses
! are supplied from the SAME cached NumPy basis as the Python reference.
! No relaxed tolerances, alternate physics or normal equations are used.
module profile_model_kernel
  use, intrinsic :: ieee_arithmetic
  implicit none
contains
  integer function upstream_policy_version() result(version)
    version=3
  end function

  recursive real(8) function pairwise_sum(a,n) result(v)
    ! Match NumPy's float64 contiguous pairwise reduction, including tail order.
    integer,intent(in)::n
    real(8),intent(in)::a(n)
    real(8)::r(8)
    integer::i,k,m
    if(n<8)then
      v=-0d0
      do i=1,n
        v=v+a(i)
      enddo
    else if(n<=128)then
      r=a(1:8);i=9
      do while(i<=n-mod(n,8))
        do k=1,8
          r(k)=r(k)+a(i+k-1)
        enddo
        i=i+8
      enddo
      v=((r(1)+r(2))+(r(3)+r(4)))+((r(5)+r(6))+(r(7)+r(8)))
      do while(i<=n)
        v=v+a(i);i=i+1
      enddo
    else
      m=n/2;m=m-mod(m,8)
      v=pairwise_sum(a(:m),m)+pairwise_sum(a(m+1:),n-m)
    endif
  end function

  real(8) function polynomial(c,x) result(v)
    real(8),intent(in)::c(3),x
    v=c(1)+x*c(2)+x*x*c(3)
  end function

  subroutine upstream_state_errors(s,states,first,last,orders,center,up,pinv,n,error)
    ! Sensitivity at the adopted surface, in log rho and relative P.
    ! L1 interpolation weights propagate a bounded measured window residual;
    ! the next polynomial order tests extrapolation beyond the fitted window.
    integer,intent(in)::n,first,last,orders(2)
    real(8),intent(in)::s(n),states(n,2),center,up(2),pinv(4,n,n,3)
    real(8),intent(out)::error(2)
    real(8)::raw(n),y(n),beta(4),higher(4),ctr,scale,target,t,avg
    real(8)::pred,res,weight,amplification,alternative,adopted,absolute,roundoff
    integer::nn,var,d,j,k
    nn=last-first+1
    error=ieee_value(0d0,ieee_positive_inf)
    if(nn<5.or.first<1.or.last>n.or.minval(up)<=0d0)return
    ctr=sum(s(first:last))/nn;scale=(s(last)-s(first))/2d0
    if(scale<=0d0)return
    target=(center-ctr)/scale
    do var=1,2
      raw(1:nn)=states(first:last,var);avg=sum(raw(1:nn))/nn
      d=orders(var)
      if(d==0)then
        error(var)=(maxval(raw(1:nn))-minval(raw(1:nn)))/avg
        cycle
      endif
      if(d<1.or.d>2)return
      y(1:nn)=raw(1:nn)
      if(var==1)y(1:nn)=log(raw(1:nn))
      beta=0;higher=0
      do k=1,d+1
        do j=1,nn
          beta(k)=beta(k)+pinv(k,j,nn,d)*y(j)
        enddo
      enddo
      do k=1,d+2
        do j=1,nn
          higher(k)=higher(k)+pinv(k,j,nn,d+1)*y(j)
        enddo
      enddo
      res=0;amplification=0
      do j=1,nn
        t=(s(first+j-1)-ctr)/scale;pred=0;weight=0
        do k=1,d+1
          pred=pred+beta(k)*t**(k-1)
          weight=weight+target**(k-1)*pinv(k,j,nn,d)
        enddo
        res=max(res,abs(y(j)-pred));amplification=amplification+abs(weight)
      enddo
      alternative=0
      do k=1,d+2
        alternative=alternative+higher(k)*target**(k-1)
      enddo
      adopted=up(var)
      if(var==1)adopted=log(up(var))
      roundoff=32d0*epsilon(1d0)*max(maxval(abs(y(1:nn))),abs(adopted),tiny(1d0)) &
               *max(amplification,1d0)
      absolute=max(abs(alternative-adopted),res*amplification)+roundoff
      error(var)=absolute
      if(var==2)error(var)=absolute/up(var)
    enddo
  end subroutine

  subroutine screen_one(s,states,u,widths,derivative,peak,slope,tol,seed,gamma,closure,pinv,xg,wg,recovery,n, &
                        coeff,vel,orders,errors,bounds,counts,roots,possible)
    ! This kernel is deliberately a conservative rejection screen. Python
    ! repeats all original physical checks and selection for possible models.
    integer,intent(in)::n,peak,recovery
    real(8),intent(in)::s(n),states(n,2),u(n),widths(n),derivative(n),slope,tol,seed,gamma,closure
    real(8),intent(in)::pinv(4,n,n,3),xg(8),wg(8)
    real(8),intent(out)::coeff(3,2,6,2),vel(3,6,2),errors(6,2),roots(6,6)
    integer,intent(out)::orders(2,6,2),bounds(2,6,2),counts(2),possible(6,6)
    real(8)::terms(n-1),total,center,a,b,up(2),down(2),support(2),ratio,rp,ent,m,re,sound,jump,vl,vr,dv,margin
    real(8),parameter::guard=1d-10
    integer::l,r,i,lo,hi,status
    coeff=0;vel=0;errors=0;roots=0;orders=0;bounds=0;counts=0;possible=0
    if(peak<1.or.peak>n)return
    call fit_windows(s,states,u,widths,peak,-1,slope,tol,seed,pinv,n, &
                     coeff(:,:,:,1),vel(:,:,1),orders(:,:,1),errors(:,1),bounds(:,:,1),counts(1),recovery)
    call fit_windows(s,states,u,widths,peak,1,slope,tol,seed,pinv,n, &
                     coeff(:,:,:,2),vel(:,:,2),orders(:,:,2),errors(:,2),bounds(:,:,2),counts(2),recovery)
    do i=1,n-1
      terms(i)=(s(i+1)-s(i))*(states(i+1,1)+states(i,1))/2d0
    enddo
    do l=1,counts(1)
      do r=1,counts(2)
        lo=bounds(2,l,1);hi=bounds(1,r,2)
        if(maxval(derivative(bounds(1,l,1):bounds(2,r,2)))>derivative(peak)*(1d0+1d-10))cycle
        a=s(lo);b=s(hi);total=pairwise_sum(terms(lo:hi-1),hi-lo)
        call mass_root(coeff(:,1,l,1),coeff(:,1,r,2),a,b,total,seed,xg,wg,center,status)
        if(status/=0)cycle
        if(abs(center-s(peak))>2d0*max(widths(peak),seed))cycle
        roots(l,r)=center
        ! Keep borderline/nonfinite arithmetic for the reference checks.
        possible(l,r)=1
        up=[exp(max(-700d0,min(700d0,polynomial(coeff(:,1,l,1),center)))), &
            polynomial(coeff(:,2,l,1),center)]
        down=[exp(max(-700d0,min(700d0,polynomial(coeff(:,1,r,2),center)))), &
              polynomial(coeff(:,2,r,2),center)]
        if(.not.all(ieee_is_finite(up)).or..not.all(ieee_is_finite(down)))cycle
        if(minval(up)<=0d0.or.minval(down)<=0d0)then
          possible(l,r)=0;cycle
        endif
        call upstream_state_errors(s,states,bounds(1,l,1),bounds(2,l,1), &
                                   orders(:,l,1),center,up,pinv,n,support)
        ! Borderline values remain for the authoritative surface checks.
        if(all(ieee_is_finite(support)).and.support(2)>1d0+guard)then
          possible(l,r)=0;cycle
        endif
        ratio=down(1)/up(1);rp=down(2)/up(2)
        if(ratio<1d0-guard.or.rp<1d0-guard)then
          possible(l,r)=0;cycle
        endif
        ent=rp/ratio**gamma
        if(.not.ieee_is_finite(ent).or.ent<=0d0)cycle
        if(log(ent)<1d-10-guard)then
          possible(l,r)=0;cycle
        endif
        m=sqrt(max(((gamma-1d0)+(gamma+1d0)*rp)/(2d0*gamma),1d0))
        re=(gamma+1d0)*m*m/((gamma-1d0)*m*m+2d0)
        sound=sqrt(gamma*up(2)/up(1));jump=m*sound*(1d0-1d0/re)
        vl=polynomial(vel(:,l,1),center);vr=polynomial(vel(:,r,2),center)
        margin=guard*max(abs(vl),abs(vr),sound,1d-300)
        if(vl<vr-margin)then
          possible(l,r)=0;cycle
        endif
        dv=abs((vl-vr)-jump)/max(abs(jump),1d-12*sound)
        if(dv>closure+margin/max(abs(jump),1d-12*sound)+guard)possible(l,r)=0
      enddo
    enddo
  end subroutine

  subroutine screen_batch(s,states,u,widths,derivative,peaks,slopes,tol,seed,gamma,closure,pinv,xg,wg,recovery,n,nb, &
                          coeff,vel,orders,errors,bounds,counts,roots,possible)
    integer,intent(in)::n,nb,peaks(nb),recovery
    real(8),intent(in)::s(n),states(n,2,nb),u(n,nb),widths(n,nb),derivative(n,nb),slopes(nb)
    real(8),intent(in)::tol,seed,gamma,closure,pinv(4,n,n,3),xg(8),wg(8)
    real(8),intent(out)::coeff(3,2,6,2,nb),vel(3,6,2,nb),errors(6,2,nb),roots(6,6,nb)
    integer,intent(out)::orders(2,6,2,nb),bounds(2,6,2,nb),counts(2,nb),possible(6,6,nb)
    integer::j
    ! Keep tiny batches serial; larger batches use the configured OpenMP team.
    !$omp parallel do private(j) schedule(dynamic,1) if(nb>=8)
    do j=1,nb
      call screen_one(s,states(:,:,j),u(:,j),widths(:,j),derivative(:,j),peaks(j),slopes(j), &
          tol,seed,gamma,closure,pinv,xg,wg,recovery,n,coeff(:,:,:,:,j),vel(:,:,:,j),orders(:,:,:,j), &
          errors(:,:,j),bounds(:,:,:,j),counts(:,j),roots(:,:,j),possible(:,:,j))
    enddo
    !$omp end parallel do
  end subroutine

  subroutine pair_support_batch(s,states,coeff,orders,bounds,roots,possible,pinv,downstream,n,nb,support)
    ! Batch surface sensitivities avoid a Python/F2PY crossing for every pair.
    integer,intent(in)::n,nb,orders(2,6,2,nb),bounds(2,6,2,nb),possible(6,6,nb),downstream
    real(8),intent(in)::s(n),states(n,2,nb),coeff(3,2,6,2,nb),roots(6,6,nb),pinv(4,n,n,3)
    real(8),intent(out)::support(2,2,6,6,nb)
    real(8)::state(2),center
    integer::j,l,r,side,k
    support=ieee_value(0d0,ieee_positive_inf)
    !$omp parallel do private(j,l,r,side,k,state,center) schedule(dynamic,1) if(nb>=8)
    do j=1,nb
      do l=1,6
        do r=1,6
          if(possible(l,r,j)==0)cycle
          center=roots(l,r,j)
          do side=1,1+downstream
            k=l
            if(side==2)k=r
            state=[exp(max(-700d0,min(700d0,polynomial(coeff(:,1,k,side,j),center)))), &
                   polynomial(coeff(:,2,k,side,j),center)]
            if(.not.all(ieee_is_finite(state)).or.minval(state)<=0d0)cycle
            call upstream_state_errors(s,states(:,:,j),bounds(1,k,side,j),bounds(2,k,side,j), &
                orders(:,k,side,j),center,state,pinv,n,support(:,side,l,r,j))
          enddo
        enddo
      enddo
    enddo
    !$omp end parallel do
  end subroutine

  subroutine physical(c,ctr,scale,out)
    real(8),intent(in)::c(3),ctr,scale
    real(8),intent(out)::out(3)
    out=[c(1)-ctr/scale*c(2)+ctr**2/scale**2*c(3), &
         c(2)/scale-2d0*ctr/scale**2*c(3),c(3)/scale**2]
  end subroutine

  subroutine fit_windows(s,states,u,widths,peak,side,slope,tol,seed,pinv,n, &
                         coeff,vel,orders,errors,bounds,nfit,recovery)
    integer,intent(in)::n,peak,side,recovery
    real(8),intent(in)::s(n),states(n,2),u(n),widths(n),slope,tol,seed,pinv(4,n,n,3)
    real(8),intent(out)::coeff(3,2,6),vel(3,6),errors(6)
    integer,intent(out)::orders(2,6),bounds(2,6),nfit
    real(8)::cs(3,2,n),us(3,n),errs(n),y(n),t(n),raw(n),beta(4),higher(4),b(3)
    real(8)::ctr,scale,target,den,avg,spread,res,unc,at,nxt,pred,e(2),last,step
    real(8)::up(2),support(2),score(n)
    integer::os(2,n),ib(2,n),edge,first,final,nn,k,j,d,var,count,flat,idx,m,chosen(6),nc
    integer::unstable(n),degree(n),best,attempt
    logical::ok,accepted,have_last,duplicate
    coeff=0;vel=0;errors=0;orders=0;bounds=0;nfit=0
    count=0;flat=0;have_last=.false.;step=s(2)-s(1)
    edge=peak+side
    do while(edge>=3.and.edge<=n-2)
      if(have_last)then
        if(abs(s(edge)-last)<max(widths(edge),seed))goto 90
      endif
      nn=max(5,ceiling(2d0*max(widths(edge),widths(peak))/step)+1)
      if(side<0)then
        first=edge-nn+1;final=edge
      else
        first=edge;final=edge+nn-1
      endif
      if(first<1.or.final>n)goto 90
      if(s(final)-s(first)+step/2d0<2d0*maxval(widths(first:final)))goto 90
      if(recovery==0.and.side>0.and.count>=4)then
        do var=1,2
          if((maxval(states(first:final,var))-minval(states(first:final,var))) &
              /(sum(states(first:final,var))/nn)>tol)goto 90
        enddo
      endif
      ctr=sum(s(first:final))/nn;scale=(s(final)-s(first))/2d0
      t(1:nn)=(s(first:final)-ctr)/scale;target=(s(peak)-ctr)/scale
      ok=.true.
      do var=1,2
        raw(1:nn)=states(first:final,var);avg=sum(raw(1:nn))/nn
        spread=(maxval(raw(1:nn))-minval(raw(1:nn)))/avg
        if(spread<=tol)then
          cs(:,var,count+1)=0
          if(var==1)then
            cs(1,var,count+1)=log(avg)
          else
            cs(1,var,count+1)=avg
          endif
          os(var,count+1)=0;e(var)=spread
          cycle
        endif
        y(1:nn)=raw(1:nn);den=avg
        if(var==1)then
          y(1:nn)=log(raw(1:nn));den=1d0
        endif
        accepted=.false.
        do d=1,2
          beta=0;higher=0
          do k=1,d+1
            do j=1,nn
              beta(k)=beta(k)+pinv(k,j,nn,d)*y(j)
            enddo
          enddo
          do k=1,d+2
            do j=1,nn
              higher(k)=higher(k)+pinv(k,j,nn,d+1)*y(j)
            enddo
          enddo
          res=0
          do j=1,nn
            pred=0
            do k=1,d+1
              pred=pred+beta(k)*t(j)**(k-1)
            enddo
            res=max(res,abs(y(j)-pred)/den)
          enddo
          at=0;nxt=0
          do k=1,d+1
            at=at+beta(k)*target**(k-1)
          enddo
          do k=1,d+2
            nxt=nxt+higher(k)*target**(k-1)
          enddo
          unc=abs(nxt-at)/den
          if(res<=tol.and.(recovery/=0.or.side<0.or.unc<=tol.or.d==2))then
            call physical(beta(1:3),ctr,scale,cs(:,var,count+1))
            os(var,count+1)=d;e(var)=min(max(res,unc),1d0)
            accepted=.true.;exit
          endif
        enddo
        if(.not.accepted)then
          ok=.false.;exit
        endif
      enddo
      if(.not.ok)goto 90
      if(cs(2,2,count+1)+2d0*s(peak)*cs(3,2,count+1)>.25d0*max(slope,1d-300))goto 90
      b=0
      do k=1,3
        do j=1,nn
          b(k)=b(k)+pinv(k,j,nn,2)*u(first+j-1)
        enddo
      enddo
      call physical(b,ctr,scale,us(:,count+1))
      errs(count+1)=sqrt(sum(e*e));ib(:,count+1)=[first,final]
      count=count+1;last=s(edge);have_last=.true.
      if(all(os(:,count)==0))flat=flat+1
      if(flat>=2)exit
90    edge=edge+side
    enddo
    nc=min(count,4)
    do j=1,nc
      chosen(j)=j
    enddo
    if(side<0.or.recovery/=0)then
      ! Rank already sampled alternatives by admissibility, then simplicity
      ! and extrapolation sensitivity. Retain two beyond the nearest four.
      do j=5,count
        degree(j)=maxval(os(:,j));unstable(j)=1;score(j)=huge(1d0)
        up=[exp(max(-700d0,min(700d0,polynomial(cs(:,1,j),s(peak))))), &
            polynomial(cs(:,2,j),s(peak))]
        if(all(ieee_is_finite(up)).and.minval(up)>0d0)then
          call upstream_state_errors(s,states,ib(1,j),ib(2,j),os(:,j),s(peak),up,pinv,n,support)
          if(all(ieee_is_finite(support)))then
            score(j)=sum(support)
            if(minval(support)>=0d0.and.support(2)<1d0)unstable(j)=0
          endif
        endif
      enddo
      do attempt=1,2
        best=0
        do j=5,count
          if(any(chosen(1:nc)==j))cycle
          if(best==0)then
            best=j
          else if(unstable(j)<unstable(best))then
            best=j
          else if(unstable(j)==unstable(best))then
            if(degree(j)<degree(best))then
              best=j
            else if(degree(j)==degree(best).and.score(j)<score(best))then
              best=j
            endif
          endif
        enddo
        if(best==0)exit
        nc=nc+1;chosen(nc)=best
      enddo
    else
      flat=0
      do j=1,count
        if(any(os(:,j)/=0))cycle
        flat=flat+1
        if(.not.any(chosen(1:nc)==j))then
          nc=nc+1;chosen(nc)=j
        endif
        if(flat==2)exit
      enddo
    endif
    nfit=nc
    do m=1,nc
      idx=chosen(m);coeff(:,:,m)=cs(:,:,idx);vel(:,m)=us(:,idx)
      errors(m)=errs(idx);orders(:,m)=os(:,idx);bounds(:,m)=ib(:,idx)
    enddo
  end subroutine

  real(8) function integral(c,a,b,xg,wg) result(v)
    real(8),intent(in)::c(3),a,b,xg(8),wg(8)
    real(8)::x,poly
    integer::i
    v=0
    do i=1,8
      x=(a+b)/2d0+(b-a)/2d0*xg(i)
      poly=c(1)+x*c(2)+x*x*c(3)
      v=v+wg(i)*exp(max(-700d0,min(700d0,poly)))
    enddo
    v=(b-a)/2d0*v
  end function

  subroutine mass_root(cl,cr,a,b,total,seed,xg,wg,root,status)
    real(8),intent(in)::cl(3),cr(3),a,b,total,seed,xg(8),wg(8)
    real(8),intent(out)::root
    integer,intent(out)::status
    real(8)::xp,xc,xb,fp,fc,fb,sp,sc,delta,half,trial,dpre,dblk
    integer::i
    xp=a;xc=b;xb=0;fb=0;sp=0;sc=0;root=0;status=1
    fp=integral(cl,a,xp,xg,wg)+integral(cr,xp,b,xg,wg)-total
    fc=integral(cl,a,xc,xg,wg)+integral(cr,xc,b,xg,wg)-total
    if(.not.ieee_is_finite(fp).or..not.ieee_is_finite(fc))return
    ! Match the reference's signbit pre-check, including signed zero.
    if((fp<0d0).eqv.(fc<0d0))return
    if(fp==0d0)then
      root=xp;status=0;return
    endif
    if(fc==0d0)then
      root=xc;status=0;return
    endif
    do i=1,60
      if(fp/=0d0.and.fc/=0d0.and.((fp<0d0).neqv.(fc<0d0)))then
        xb=xp;fb=fp;sp=xc-xp;sc=sp
      endif
      if(abs(fb)<abs(fc))then
        xp=xc;xc=xb;xb=xp
        fp=fc;fc=fb;fb=fp
      endif
      delta=(1d-10*seed+1d-12*abs(xc))/2d0;half=(xb-xc)/2d0
      if(fc==0d0.or.abs(half)<delta)then
        root=xc;status=0;return
      endif
      if(abs(sp)>delta.and.abs(fc)<abs(fp))then
        if(xp==xb)then
          trial=-fc*(xc-xp)/(fc-fp)
        else
          dpre=(fp-fc)/(xp-xc);dblk=(fb-fc)/(xb-xc)
          trial=-fc*(fb*dblk-fp*dpre)/(dblk*dpre*(fb-fp))
        endif
        if(2d0*abs(trial)<min(abs(sp),3d0*abs(half)-delta))then
          sp=sc;sc=trial
        else
          sp=half;sc=half
        endif
      else
        sp=half;sc=half
      endif
      xp=xc;fp=fc
      if(abs(sc)>delta)then
        xc=xc+sc
      else
        xc=xc+sign(delta,half)
      endif
      fc=integral(cl,a,xc,xg,wg)+integral(cr,xc,b,xg,wg)-total
      if(.not.ieee_is_finite(fc))return
    enddo
  end subroutine
end module
