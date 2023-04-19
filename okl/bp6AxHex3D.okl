/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#if p_N<7
#define USE_3D_SHMEM 1
#else
#define USE_3D_SHMEM 0
#endif

#if !USE_3D_SHMEM


//This kernel processes 2D slices of the element in shmem and uses register arrays
// to store the element itself. May be slower for low order but allows us to run
// high degree efficiently

/* The NV compiler doesn't like the blocked kernel. Skip using it. */
#if OCCA_USE_CUDA==1 && p_N>=8

//padding for bank conflicts
#if p_Nq==16
#define p_pad 1
#else
#define p_pad 0
#endif

@kernel void bp6AxHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  wJ,
                        @restrict const  dfloat *  vgeo,
                        @restrict const  dfloat *  ggeo,
                        @restrict const  dfloat *  gllw,
                        @restrict const  dfloat *  D,
                        @restrict const  dfloat *  S,
                        @restrict const  dfloat *  MM,
                        const dfloat lambda,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

  for(dlong e=0; e<Nelements; e++; @outer(0)){

    @shared dfloat s_D[p_Nq][p_Nq+p_pad];
    @shared dfloat s_q[p_Nfields][p_Nq][p_Nq+p_pad];
    @shared dfloat s_v[p_Nfields][p_Nq][p_Nq+p_pad];
    @shared dfloat s_w[p_Nfields][p_Nq][p_Nq+p_pad];

    @exclusive dfloat r_GDut[p_Nfields], r_Auk[p_Nfields];

    // register array to hold u(i,j,0:N) private to thread
    @exclusive dfloat r_u[p_Nfields][p_Nq];
    // array for results Au(i,j,0:N)
    @exclusive dfloat r_Au[p_Nfields][p_Nq];

    @exclusive dlong element;

    for(int j=0;j<p_Nq;++j;@inner(1)){
      for(int i=0;i<p_Nq;++i;@inner(0)){

        //load D into local memory
        // s_D[i][j] = d \phi_i at node j
        s_D[j][i] = D[p_Nq*j+i];// D is column major

        element = elementList[e];

        const dlong base = i + j*p_Nq + element*p_Np*p_Nfields;

        // load pencil of u into register
        #pragma unroll p_Nq
        for (int k=0;k<p_Nq;k++) {
          const dlong id0 = GlobalToLocal[base + 0*p_Np + k*p_Nq*p_Nq];
          const dlong id1 = GlobalToLocal[base + 1*p_Np + k*p_Nq*p_Nq];
          const dlong id2 = GlobalToLocal[base + 2*p_Np + k*p_Nq*p_Nq];
          r_u[0][k] = (id0!=-1) ? q[id0] : 0.0;
          r_u[1][k] = (id1!=-1) ? q[id1] : 0.0;
          r_u[2][k] = (id2!=-1) ? q[id2] : 0.0;
        }

        #pragma unroll p_Nq
        for (int k=0;k<p_Nq;k++) {
          r_Au[0][k] = 0.0;
          r_Au[1][k] = 0.0;
          r_Au[2][k] = 0.0;
        }
      }
    }

    // Layer by layer
#if OCCA_USE_CUDA==1
    // only force some type of unrolling in CUDA mode
    #pragma unroll p_Nq
#endif
    for(int k = 0;k < p_Nq; k++){

      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          // share u(:,:,k)
          s_q[0][j][i] = r_u[0][k];
          s_q[1][j][i] = r_u[1][k];
          s_q[2][j][i] = r_u[2][k];
        }
      }

      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          // prefetch geometric factors
          const dlong gbase = element*p_Np + k*p_Nq*p_Nq + j*p_Nq + i;
          const dfloat r_GwJ = wJ[gbase];
          const dfloat r_G00 = ggeo[p_Nggeo*gbase+p_G00ID];
          const dfloat r_G01 = ggeo[p_Nggeo*gbase+p_G01ID];
          const dfloat r_G11 = ggeo[p_Nggeo*gbase+p_G11ID];
          const dfloat r_G12 = ggeo[p_Nggeo*gbase+p_G12ID];
          const dfloat r_G02 = ggeo[p_Nggeo*gbase+p_G02ID];
          const dfloat r_G22 = ggeo[p_Nggeo*gbase+p_G22ID];

          //start with du/dr
          dfloat du0 = 0.f, du1 = 0.f, du2 = 0.f;
          // #pragma unroll p_Nq
          for (int m=0;m<p_Nq;m++) {
            const dfloat Dr = s_D[i][m];
            du0 += Dr*s_q[0][j][m];
            du1 += Dr*s_q[1][j][m];
            du2 += Dr*s_q[2][j][m];
          }

          s_v[0][j][i] = r_G00*du0;
          s_v[1][j][i] = r_G00*du1;
          s_v[2][j][i] = r_G00*du2;
          s_w[0][j][i] = r_G01*du0;
          s_w[1][j][i] = r_G01*du1;
          s_w[2][j][i] = r_G01*du2;
          r_GDut[0]    = r_G02*du0;
          r_GDut[1]    = r_G02*du1;
          r_GDut[2]    = r_G02*du2;

          //next du/ds
          du0 = 0.f, du1 = 0.f, du2 = 0.f;

          // #pragma unroll p_Nq
          for (int m=0;m<p_Nq;m++) {
            const dfloat Ds = s_D[j][m];
            du0 += Ds*s_q[0][m][i];
            du1 += Ds*s_q[1][m][i];
            du2 += Ds*s_q[2][m][i];
          }

          s_v[0][j][i] += r_G01*du0;
          s_v[1][j][i] += r_G01*du1;
          s_v[2][j][i] += r_G01*du2;
          s_w[0][j][i] += r_G11*du0;
          s_w[1][j][i] += r_G11*du1;
          s_w[2][j][i] += r_G11*du2;
          r_GDut[0]    += r_G12*du0;
          r_GDut[1]    += r_G12*du1;
          r_GDut[2]    += r_G12*du2;

          //next du/dt
          du0 = 0.f, du1 = 0.f, du2 = 0.f;

          // #pragma unroll p_Nq
          for (int m=0;m<p_Nq;m++) {
            const dfloat Dt = s_D[k][m];
            du0 += Dt*r_u[0][m];
            du1 += Dt*r_u[1][m];
            du2 += Dt*r_u[2][m];
          }

          s_v[0][j][i] += r_G02*du0;
          s_v[1][j][i] += r_G02*du1;
          s_v[2][j][i] += r_G02*du2;
          s_w[0][j][i] += r_G12*du0;
          s_w[1][j][i] += r_G12*du1;
          s_w[2][j][i] += r_G12*du2;
          r_GDut[0]    += r_G22*du0;
          r_GDut[1]    += r_G22*du1;
          r_GDut[2]    += r_G22*du2;

          r_Auk[0] = r_GwJ*lambda*r_u[0][k];
          r_Auk[1] = r_GwJ*lambda*r_u[1][k];
          r_Auk[2] = r_GwJ*lambda*r_u[2][k];
        }
      }

      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          // #pragma unroll p_Nq
          for (int m=0;m<p_Nq;m++) {
            const dfloat Dt = s_D[k][m];
            r_Au[0][m] += Dt*r_GDut[0];
            r_Au[1][m] += Dt*r_GDut[1];
            r_Au[2][m] += Dt*r_GDut[2];
          }

          // #pragma unroll p_Nq
          for (int m=0;m<p_Nq;m++) {
            const dfloat Dr = s_D[m][i];
            r_Auk[0] += Dr*s_v[0][j][m];
            r_Auk[1] += Dr*s_v[1][j][m];
            r_Auk[2] += Dr*s_v[2][j][m];
          }

          // #pragma unroll p_Nq
          for (int m=0;m<p_Nq;m++) {
            const dfloat Ds = s_D[m][j];
            r_Auk[0] += Ds*s_w[0][m][i];
            r_Auk[1] += Ds*s_w[1][m][i];
            r_Auk[2] += Ds*s_w[2][m][i];
          }

          r_Au[0][k] += r_Auk[0];
          r_Au[1][k] += r_Auk[1];
          r_Au[2][k] += r_Auk[2];
        }
      }
    } //end Layer by layer

    // write out
    for(int j=0;j<p_Nq;++j;@inner(1)){
      for(int i=0;i<p_Nq;++i;@inner(0)){
        const dlong id = element*p_Np*p_Nfields + j*p_Nq + i;

        // #pragma unroll p_Nq
        for (int k=0;k<p_Nq;k++) {
          Aq[id+0*p_Np+k*p_Nq*p_Nq] = r_Au[0][k];
          Aq[id+1*p_Np+k*p_Nq*p_Nq] = r_Au[1][k];
          Aq[id+2*p_Np+k*p_Nq*p_Nq] = r_Au[2][k];
        }
      }
    }
  }
}

#else

/* Blocked version */
#if p_N==1
#define p_NelementsPerBlk 16
#elif p_N==2
#define p_NelementsPerBlk 56
#elif p_N==3
#define p_NelementsPerBlk 32
#elif p_N==4
#define p_NelementsPerBlk 5
#elif p_N==5
#define p_NelementsPerBlk 1
#elif p_N==6
#define p_NelementsPerBlk 5
#elif p_N==7
#define p_NelementsPerBlk 1
#elif p_N==8
#define p_NelementsPerBlk 3
#elif p_N==9
#define p_NelementsPerBlk 1
#elif p_N==10
#define p_NelementsPerBlk 1
#elif p_N==11
#define p_NelementsPerBlk 1
#elif p_N==12
#define p_NelementsPerBlk 1
#elif p_N==13
#define p_NelementsPerBlk 1
#elif p_N==14
#define p_NelementsPerBlk 1
#elif p_N==15
#define p_NelementsPerBlk 1
#else
#define p_NelementsPerBlk 1
#endif

//padding for bank conflicts
#if p_Nq==16
#define p_pad 1
#else
#define p_pad 0
#endif

@kernel void bp6AxHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  wJ,
                        @restrict const  dfloat *  vgeo,
                        @restrict const  dfloat *  ggeo,
                        @restrict const  dfloat *  gllw,
                        @restrict const  dfloat *  D,
                        @restrict const  dfloat *  S,
                        @restrict const  dfloat *  MM,
                        const dfloat lambda,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

  for(dlong eo=0; eo<Nelements; eo+=p_NelementsPerBlk; @outer(0)){

    @shared dfloat s_D[p_Nq][p_Nq+p_pad];
    @shared dfloat s_q[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq+p_pad];
    @shared dfloat s_v[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq+p_pad];
    @shared dfloat s_w[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq+p_pad];

    @exclusive dfloat r_GDut[p_Nfields], r_Auk[p_Nfields];

    // register array to hold u(i,j,0:N) private to thread
    @exclusive dfloat r_u[p_Nfields][p_Nq];
    // array for results Au(i,j,0:N)
    @exclusive dfloat r_Au[p_Nfields][p_Nq];

    @exclusive dlong r_e, element;

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          //load D into local memory
          // s_D[i][j] = d \phi_i at node j
          if (es==0) {
            s_D[j][i] = D[p_Nq*j+i];// D is column major
          }

          r_e = es+eo;

          if(r_e<Nelements){
            element = elementList[r_e];

            const dlong base = i + j*p_Nq + element*p_Np*p_Nfields;

            // load pencil of u into register
            // #pragma unroll p_Nq
            for (int k=0;k<p_Nq;k++) {
              const dlong id0 = GlobalToLocal[base + 0*p_Np + k*p_Nq*p_Nq];
              const dlong id1 = GlobalToLocal[base + 1*p_Np + k*p_Nq*p_Nq];
              const dlong id2 = GlobalToLocal[base + 2*p_Np + k*p_Nq*p_Nq];
              r_u[0][k] = (id0!=-1) ? q[id0] : 0.0;
              r_u[1][k] = (id1!=-1) ? q[id1] : 0.0;
              r_u[2][k] = (id2!=-1) ? q[id2] : 0.0;
            }

            // #pragma unroll p_Nq
            for (int k=0;k<p_Nq;k++) {
              r_Au[0][k] = 0.0;
              r_Au[1][k] = 0.0;
              r_Au[2][k] = 0.0;
            }
          }
        }
      }
    }

    // Layer by layer
#if OCCA_USE_CUDA==1
    // only force some type of unrolling in CUDA mode
    #pragma unroll p_Nq
#endif
    for(int k = 0;k < p_Nq; k++){

      for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
        for(int j=0;j<p_Nq;++j;@inner(1)){
          for(int i=0;i<p_Nq;++i;@inner(0)){
            // share u(:,:,k)
            s_q[0][es][j][i] = r_u[0][k];
            s_q[1][es][j][i] = r_u[1][k];
            s_q[2][es][j][i] = r_u[2][k];
          }
        }
      }

      for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
        for(int j=0;j<p_Nq;++j;@inner(1)){
          for(int i=0;i<p_Nq;++i;@inner(0)){

            dfloat r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;

            if(r_e<Nelements){
              // prefetch geometric factors
              const dlong gbase = element*p_Np + k*p_Nq*p_Nq + j*p_Nq + i;

              r_GwJ = wJ[gbase];
              r_G00 = ggeo[p_Nggeo*gbase+p_G00ID];
              r_G01 = ggeo[p_Nggeo*gbase+p_G01ID];
              r_G11 = ggeo[p_Nggeo*gbase+p_G11ID];
              r_G12 = ggeo[p_Nggeo*gbase+p_G12ID];
              r_G02 = ggeo[p_Nggeo*gbase+p_G02ID];
              r_G22 = ggeo[p_Nggeo*gbase+p_G22ID];
            }

            //start with du/dr
            dfloat du0 = 0.f, du1 = 0.f, du2 = 0.f;
            // #pragma unroll p_Nq
            for (int m=0;m<p_Nq;m++) {
              const dfloat Dr = s_D[i][m];
              du0 += Dr*s_q[0][es][j][m];
              du1 += Dr*s_q[1][es][j][m];
              du2 += Dr*s_q[2][es][j][m];
            }

            s_v[0][es][j][i] = r_G00*du0;
            s_v[1][es][j][i] = r_G00*du1;
            s_v[2][es][j][i] = r_G00*du2;
            s_w[0][es][j][i] = r_G01*du0;
            s_w[1][es][j][i] = r_G01*du1;
            s_w[2][es][j][i] = r_G01*du2;
            r_GDut[0]        = r_G02*du0;
            r_GDut[1]        = r_G02*du1;
            r_GDut[2]        = r_G02*du2;

            //next du/ds
            du0 = 0.f, du1 = 0.f, du2 = 0.f;

            // #pragma unroll p_Nq
            for (int m=0;m<p_Nq;m++) {
              const dfloat Ds = s_D[j][m];
              du0 += Ds*s_q[0][es][m][i];
              du1 += Ds*s_q[1][es][m][i];
              du2 += Ds*s_q[2][es][m][i];
            }

            s_v[0][es][j][i] += r_G01*du0;
            s_v[1][es][j][i] += r_G01*du1;
            s_v[2][es][j][i] += r_G01*du2;
            s_w[0][es][j][i] += r_G11*du0;
            s_w[1][es][j][i] += r_G11*du1;
            s_w[2][es][j][i] += r_G11*du2;
            r_GDut[0]        += r_G12*du0;
            r_GDut[1]        += r_G12*du1;
            r_GDut[2]        += r_G12*du2;

            //next du/dt
            du0 = 0.f, du1 = 0.f, du2 = 0.f;

            // #pragma unroll p_Nq
            for (int m=0;m<p_Nq;m++) {
              const dfloat Dt = s_D[k][m];
              du0 += Dt*r_u[0][m];
              du1 += Dt*r_u[1][m];
              du2 += Dt*r_u[2][m];
            }

            s_v[0][es][j][i] += r_G02*du0;
            s_v[1][es][j][i] += r_G02*du1;
            s_v[2][es][j][i] += r_G02*du2;
            s_w[0][es][j][i] += r_G12*du0;
            s_w[1][es][j][i] += r_G12*du1;
            s_w[2][es][j][i] += r_G12*du2;
            r_GDut[0]        += r_G22*du0;
            r_GDut[1]        += r_G22*du1;
            r_GDut[2]        += r_G22*du2;

            r_Auk[0] = r_GwJ*lambda*r_u[0][k];
            r_Auk[1] = r_GwJ*lambda*r_u[1][k];
            r_Auk[2] = r_GwJ*lambda*r_u[2][k];
          }
        }
      }

      for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
        for(int j=0;j<p_Nq;++j;@inner(1)){
          for(int i=0;i<p_Nq;++i;@inner(0)){

            // #pragma unroll p_Nq
            for (int m=0;m<p_Nq;m++) {
              const dfloat Dt = s_D[k][m];
              r_Au[0][m] += Dt*r_GDut[0];
              r_Au[1][m] += Dt*r_GDut[1];
              r_Au[2][m] += Dt*r_GDut[2];
            }

            // #pragma unroll p_Nq
            for (int m=0;m<p_Nq;m++) {
              const dfloat Dr = s_D[m][i];
              r_Auk[0] += Dr*s_v[0][es][j][m];
              r_Auk[1] += Dr*s_v[1][es][j][m];
              r_Auk[2] += Dr*s_v[2][es][j][m];
            }

            // #pragma unroll p_Nq
            for (int m=0;m<p_Nq;m++) {
              const dfloat Ds = s_D[m][j];
              r_Auk[0] += Ds*s_w[0][es][m][i];
              r_Auk[1] += Ds*s_w[1][es][m][i];
              r_Auk[2] += Ds*s_w[2][es][m][i];
            }

            r_Au[0][k] += r_Auk[0];
            r_Au[1][k] += r_Auk[1];
            r_Au[2][k] += r_Auk[2];
          }
        }
      }
    } //end Layer by layer

    // write out
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          if(r_e<Nelements){
            const dlong id = element*p_Np*p_Nfields + j*p_Nq + i;

            // #pragma unroll p_Nq
            for (int k=0;k<p_Nq;k++) {
              Aq[id+0*p_Np+k*p_Nq*p_Nq] = r_Au[0][k];
              Aq[id+1*p_Np+k*p_Nq*p_Nq] = r_Au[1][k];
              Aq[id+2*p_Np+k*p_Nq*p_Nq] = r_Au[2][k];
            }
          }
        }
      }
    }
  }
}
#endif

#elif USE_3D_SHMEM
//This kernel stores the entire hex element in shmem.
// Good for low orders, but will exceed 1024 threads per block after N=9

#if p_N==1
#define p_NelementsPerBlk 8
#elif p_N==2
#define p_NelementsPerBlk 4
#elif p_N==3
#define p_NelementsPerBlk 2
#elif p_N==4
#define p_NelementsPerBlk 1
#elif p_N==5
#define p_NelementsPerBlk 1
#elif p_N==6
#define p_NelementsPerBlk 1
#elif p_N==7
#define p_NelementsPerBlk 1
#else
#define p_NelementsPerBlk 1
#endif

@kernel void bp6AxHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  wJ,
                        @restrict const  dfloat *  vgeo,
                        @restrict const  dfloat *  ggeo,
                        @restrict const  dfloat *  gllw,
                        @restrict const  dfloat *  D,
                        @restrict const  dfloat *  S,
                        @restrict const  dfloat *  MM,
                        const dfloat lambda,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

//padding for bank conflicts
#if p_Nq==8 || p_Nq==4
#define p_pad 1
#else
#define p_pad 0
#endif

  for(int eo=0;eo<Nelements;eo+=p_NelementsPerBlk;@outer(0)){

    @shared dfloat s_D [p_Nq][p_Nq+p_pad];
    @shared dfloat s_DT[p_Nq][p_Nq+p_pad];
    @shared dfloat   s_q[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];
    @shared dfloat s_Gqr[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];
    @shared dfloat s_Gqs[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];
    @shared dfloat s_Gqt[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];

    @exclusive dlong r_e, element;

    @exclusive int k, es;

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          //load operators
          if(ke==0){
            const int id = j*p_Nq+i;
            const dfloat Dji = D[id];
            s_D[j][i] = Dji;
            s_DT[i][j] = Dji;
          }

          k  = ke%p_Nq;
          es = ke/p_Nq;
          r_e = es+eo;

          if(r_e<Nelements){
            element = elementList[r_e];
            const dlong base = i + j*p_Nq + k*p_Nq*p_Nq + element*p_Np*p_Nfields;
            const dlong id0 = GlobalToLocal[base + 0*p_Np];
            const dlong id1 = GlobalToLocal[base + 1*p_Np];
            const dlong id2 = GlobalToLocal[base + 2*p_Np];
            s_q[0][es][k][j][i] = (id0!=-1) ? q[id0] : 0.0;
            s_q[1][es][k][j][i] = (id1!=-1) ? q[id1] : 0.0;
            s_q[2][es][k][j][i] = (id2!=-1) ? q[id2] : 0.0;
          }
        }
      }
    }

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          if(r_e<Nelements){
            const dlong gbase = element*p_Np + i + j*p_Nq + k*p_Nq*p_Nq;

            dfloat tmp[p_Nfields];

            // 't' terms
            tmp[0] = 0;
            tmp[1] = 0;
            tmp[2] = 0;

            const dfloat G00 = ggeo[p_Nggeo*gbase+p_G00ID];
            const dfloat G01 = ggeo[p_Nggeo*gbase+p_G01ID];
            const dfloat G11 = ggeo[p_Nggeo*gbase+p_G11ID];
            const dfloat G02 = ggeo[p_Nggeo*gbase+p_G02ID];
            const dfloat G12 = ggeo[p_Nggeo*gbase+p_G12ID];
            const dfloat G22 = ggeo[p_Nggeo*gbase+p_G22ID];

            // #pragma unroll p_Unr
            for(int m = 0; m < p_Nq; ++m) {
              const dfloat Dkm = s_DT[m][k];
              tmp[0] += Dkm*s_q[0][es][m][j][i];
              tmp[1] += Dkm*s_q[1][es][m][j][i];
              tmp[2] += Dkm*s_q[2][es][m][j][i];
            }


            s_Gqr[0][es][k][j][i] = G02*tmp[0];
            s_Gqr[1][es][k][j][i] = G02*tmp[1];
            s_Gqr[2][es][k][j][i] = G02*tmp[2];
            s_Gqs[0][es][k][j][i] = G12*tmp[0];
            s_Gqs[1][es][k][j][i] = G12*tmp[1];
            s_Gqs[2][es][k][j][i] = G12*tmp[2];
            s_Gqt[0][es][k][j][i] = G22*tmp[0];
            s_Gqt[1][es][k][j][i] = G22*tmp[1];
            s_Gqt[2][es][k][j][i] = G22*tmp[2];


            // 'r' terms
            tmp[0] = 0;
            tmp[1] = 0;
            tmp[2] = 0;

            // #pragma unroll p_Unr
            for(int m = 0; m < p_Nq; ++m) {
              const dfloat Dim = s_D[i][m];
              tmp[0] += Dim*s_q[0][es][k][j][m];
              tmp[1] += Dim*s_q[1][es][k][j][m];
              tmp[2] += Dim*s_q[2][es][k][j][m];
            }

            s_Gqr[0][es][k][j][i] += G00*tmp[0];
            s_Gqr[1][es][k][j][i] += G00*tmp[1];
            s_Gqr[2][es][k][j][i] += G00*tmp[2];
            s_Gqs[0][es][k][j][i] += G01*tmp[0];
            s_Gqs[1][es][k][j][i] += G01*tmp[1];
            s_Gqs[2][es][k][j][i] += G01*tmp[2];
            s_Gqt[0][es][k][j][i] += G02*tmp[0];
            s_Gqt[1][es][k][j][i] += G02*tmp[1];
            s_Gqt[2][es][k][j][i] += G02*tmp[2];


            // 's' terms
            tmp[0] = 0;
            tmp[1] = 0;
            tmp[2] = 0;

            // #pragma unroll p_Unr
            for(int m = 0; m < p_Nq; ++m) {
              const dfloat Djm = s_D[j][m];
              tmp[0] += Djm*s_q[0][es][k][m][i];
              tmp[1] += Djm*s_q[1][es][k][m][i];
              tmp[2] += Djm*s_q[2][es][k][m][i];
            }

            s_Gqr[0][es][k][j][i] += G01*tmp[0];
            s_Gqr[1][es][k][j][i] += G01*tmp[1];
            s_Gqr[2][es][k][j][i] += G01*tmp[2];
            s_Gqs[0][es][k][j][i] += G11*tmp[0];
            s_Gqs[1][es][k][j][i] += G11*tmp[1];
            s_Gqs[2][es][k][j][i] += G11*tmp[2];
            s_Gqt[0][es][k][j][i] += G12*tmp[0];
            s_Gqt[1][es][k][j][i] += G12*tmp[1];
            s_Gqt[2][es][k][j][i] += G12*tmp[2];
          }
        }
      }
    }

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          if(r_e<Nelements){
            const dlong gbase = element*p_Np + i + j*p_Nq + k*p_Nq*p_Nq;
            const dfloat GWJ = wJ[gbase];

            dfloat tmpAp[p_Nfields];
            tmpAp[0] = s_q[0][es][k][j][i]*lambda*GWJ;
            tmpAp[1] = s_q[1][es][k][j][i]*lambda*GWJ;
            tmpAp[2] = s_q[2][es][k][j][i]*lambda*GWJ;

            // use same matrix for both slices
            // #pragma unroll p_Unr
            for(int m=0;m<p_Nq;++m){
              const dfloat Dmi = s_D[m][i];
              const dfloat Dmj = s_D[m][j];

              tmpAp[0] += Dmi*s_Gqr[0][es][k][j][m];
              tmpAp[1] += Dmi*s_Gqr[1][es][k][j][m];
              tmpAp[2] += Dmi*s_Gqr[2][es][k][j][m];
              tmpAp[0] += Dmj*s_Gqs[0][es][k][m][i];
              tmpAp[1] += Dmj*s_Gqs[1][es][k][m][i];
              tmpAp[2] += Dmj*s_Gqs[2][es][k][m][i];
            }

            // #pragma unroll p_Unr
            for(int m=0;m<p_Nq;++m){
              const dfloat Dmk = s_D[m][k];
              tmpAp[0] += Dmk*s_Gqt[0][es][m][j][i];
              tmpAp[1] += Dmk*s_Gqt[1][es][m][j][i];
              tmpAp[2] += Dmk*s_Gqt[2][es][m][j][i];
            }

            const dlong base = i + j*p_Nq + k*p_Nq*p_Nq + element*p_Np*p_Nfields;
            Aq[base+0*p_Np] = tmpAp[0];
            Aq[base+1*p_Np] = tmpAp[1];
            Aq[base+2*p_Np] = tmpAp[2];
          }
        }
      }
    }
  }
}

#endif
