void mwd_iso_ref_2space_1time( const int shape[3], const int xb, const int yb_r0, const int zb, const int xe, const int ye_r0, const int ze,
    const real_t * __restrict__ coef, real_t * __restrict__ u,
    real_t * __restrict__ v, const real_t * __restrict__ roc2, int t_dim, int b_inc, int e_inc, int NHALO, int tb, int te, stencil_CTX stencil_ctx, int mtid) {

  int *abs_tid;

#pragma omp parallel shared(abs_tid, shape, stencil_ctx, roc2, coef, mtid, tb, te, t_dim, NHALO) firstprivate(u, v, b_inc, e_inc) num_threads(stencil_ctx.thread_group_size) PROC_BIND(master)


  {
    int tgs, nwf, th_nwf, tid, gtid, zi, yb, ye, ib, ie, kt, t, k, j, i, q, r, err;
    double t_start, thb, the;

    const int nny =shape[1];
    const int nnx =shape[0];
    const unsigned long nnxy = 1UL * nnx * nny;
    uint64_t ln_domain = ((uint64_t) 1)* shape[0]*shape[1]*shape[2];

    tgs = stencil_ctx.thread_group_size;
    nwf = stencil_ctx.num_wf;

    tid = 0;
    gtid = 0;





    if(stencil_ctx.use_manual_cpu_bind == 1){
      err = sched_setaffinity(0, stencil_ctx.setsize, stencil_ctx.bind_masks[mtid*tgs+tid]);
      if(err==-1) printf("WARNING: Could not set CPU Affinity\n");
    }



    real_t * __restrict__ u_r = u;
    real_t * __restrict__ v_r = v;
    real_t *__restrict__ ux, *__restrict__ vx;

    int th_x = stencil_ctx.th_x;
    int th_y = stencil_ctx.th_y;
    int th_z = stencil_ctx.th_z;


    int tid_x = tid%th_x;
    int tid_y = tid/th_x;
    int tid_z = tid/(th_x*th_y);

    int yb_r = yb_r0;
    int ye_r = ye_r0;

    if(stencil_ctx.th_y>1 ){
      if(b_inc !=0 && e_inc!=0){
        if (tid_y%2 == 0){
          ye_r = (yb_r + ye_r)/2;
          e_inc = 0;
        } else{
          yb_r = (yb_r + ye_r)/2;
          b_inc = 0;
        }
      }else{
        th_z *= th_y;
        tid_z = tid/th_x;
        if (nwf < th_z) nwf = th_z;
      }
    }

    int nbx = (xe-xb)/th_x;
    q = (int)((xe-xb)/th_x);
    r = (xe-xb)%th_x;
    if(tid_x < r) {
      ib = xb + tid_x * (q+1);
      ie = ib + (q+1);
    }else {
      ib = xb + r * (q+1) + (tid_x - r) * q;
      ie = ib + q;
    }

    th_nwf = nwf/th_z;
    thb = th_nwf*tid_z;
    the = th_nwf*(tid_z+1);

    for(zi=zb; zi<ze; zi+=nwf) {

      if(ze-zi < nwf){
        q = (int)((ze-zi)/th_z);
        r = (ze-zi)%th_z;
        if(tid_z < r) {
          thb = tid_z * (q+1);
          the = thb + (q+1);
        }else {
          thb = r * (q+1) + (tid_z - r) * q;
          the =thb + q;
        }
      }

      yb = yb_r;
      ye = ye_r;

      kt = zi;
      for(t=tb; t< te; t++){
        if(t%2 == 0){
          u = v_r; v = u_r;
        } else{
          u = u_r; v = v_r;
        }

        for(k=kt+thb; k<kt+the; k++){
          for(j=yb; j<ye; j++) {
            ux = &(u[1ULL*k*nnxy + j*nnx]);
            vx = &(v[1ULL*k*nnxy + j*nnx]);
#pragma simd
            for(i=ib; i<ie; i++) {
              { ux[i] = coef[0]*vx[i] +coef[1]*(vx[i+1]+vx[i-1]) +coef[1]*(vx[i+nnx]+vx[i-nnx]) +coef[1]*(vx[-nnxy+i]+vx[+nnxy+i]); }
            }
          }
        }


        if(t< t_dim){
          yb -= b_inc;
          ye += e_inc;
        }else{
          yb += b_inc;
          ye -= e_inc;
        }

        kt -= NHALO;

        t_start = MPI_Wtime();
#pragma omp barrier
        stencil_ctx.t_wait[gtid] += MPI_Wtime() - t_start;

      }
    }
  }
}
