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
    !$omp parallel do private(i) schedule(static)
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
    !$omp parallel do private(i,j,axis,t1,t2,side,sign,a,b,id,count,sides,summed,probe,distance) schedule(static)
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
  subroutine physical(c,ctr,scale,out)
    real(8),intent(in)::c(3),ctr,scale
    real(8),intent(out)::out(3)
    out=[c(1)-ctr/scale*c(2)+ctr**2/scale**2*c(3), &
         c(2)/scale-2d0*ctr/scale**2*c(3),c(3)/scale**2]
  end subroutine

  subroutine fit_windows(s,states,u,widths,peak,side,slope,tol,seed,pinv,n, &
                         coeff,vel,orders,errors,bounds,nfit)
    integer,intent(in)::n,peak,side
    real(8),intent(in)::s(n),states(n,2),u(n),widths(n),slope,tol,seed,pinv(4,n,n,3)
    real(8),intent(out)::coeff(3,2,6),vel(3,6),errors(6)
    integer,intent(out)::orders(2,6),bounds(2,6),nfit
    real(8)::cs(3,2,n),us(3,n),errs(n),y(n),t(n),raw(n),beta(4),higher(4),b(3)
    real(8)::ctr,scale,target,den,avg,spread,res,unc,at,nxt,pred,e(2),last,step
    integer::os(2,n),ib(2,n),edge,first,final,nn,k,j,d,var,count,flat,idx,m,chosen(6),nc
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
      if(count>=4)then
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
          if(res<=tol.and.(unc<=tol.or.d==2))then
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
    flat=0
    do j=1,count
      if(any(os(:,j)/=0))cycle
      flat=flat+1
      duplicate=any(chosen(1:nc)==j)
      if(.not.duplicate)then
        nc=nc+1;chosen(nc)=j
      endif
      if(flat==2)exit
    enddo
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
