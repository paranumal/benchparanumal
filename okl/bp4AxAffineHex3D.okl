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

#if p_N>6 || (OCCA_USE_CUDA=1 && p_N>5)

#if OCCA_USE_CUDA=1

#if p_N==1
#define p_NelementsPerBlk 28
#elif p_N==2
#define p_NelementsPerBlk 16
#elif p_N==3
#define p_NelementsPerBlk 10
#elif p_N==4
#define p_NelementsPerBlk 7
#elif p_N==5
#define p_NelementsPerBlk 5
#elif p_N==6
#define p_NelementsPerBlk 2
#elif p_N==7
#define p_NelementsPerBlk 2
#elif p_N==8
#define p_NelementsPerBlk 1
#elif p_N==9
#define p_NelementsPerBlk 1
#else
// unoptimized
#define p_NelementsPerBlk 1
#endif

#else

#if p_N==1
#define p_NelementsPerBlk 28
#elif p_N==2
#define p_NelementsPerBlk 16
#elif p_N==3
#define p_NelementsPerBlk 10
#elif p_N==4
#define p_NelementsPerBlk 7
#elif p_N==5
#define p_NelementsPerBlk 5
#elif p_N==6
#define p_NelementsPerBlk 4
#elif p_N==7
#define p_NelementsPerBlk 3
#elif p_N==8
#define p_NelementsPerBlk 2
#elif p_N==9
#define p_NelementsPerBlk 2
#else
// unoptimized
#define p_NelementsPerBlk 1
#endif

#endif

@kernel void bp4AxAffineHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  wJ,
                        @restrict const  dfloat *  ggeo,
                        @restrict const  dfloat *  D,
                        @restrict const  dfloat *  I,
                        @restrict const  dfloat *  invV,
                        @restrict const  dfloat *  S,
                        @restrict const  dfloat *  MM,
                        const dfloat lambda,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

  for(int eo=0;eo<Nelements;eo+=p_NelementsPerBlk;@outer(0)){

//padding for bank conflicts
#if p_Nq==8 || p_Nq==4
#define p_pad 1
#else
#define p_pad 0
#endif

    @shared dfloat s_q[p_Nq][p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq+p_pad];
    @shared dfloat s_D[p_Nq][p_Nq+p_pad];
    @shared dfloat s_qr[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq+p_pad];
    @shared dfloat s_qs[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq+p_pad];
    @shared dfloat s_invV[p_Nq][p_Nq];

    @exclusive dfloat r_q[p_Nfields][p_Nq];
    @exclusive dfloat r_qt[p_Nfields];

    @exclusive dlong r_e, element;

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int b=0;b<p_Nq;++b;@inner(1)){
        for(int a=0;a<p_Nq;++a;@inner(0)){

          if(es==0){
            s_invV[b][a] = invV[a+p_Nq*b];
            s_D[b][a] = D[b*p_Nq+a];
          }

          r_e = eo + es;

          if(r_e<Nelements) {
            element = elementList[r_e];

            // load q
            // #pragma unroll p_Nq
            for(int c=0;c<p_Nq;++c) {
              const dlong base = a + b*p_Nq + c*p_Nq*p_Nq + element*p_Np*p_Nfields;

              for(int f=0;f<p_Nfields;++f) {
                const dlong id = GlobalToLocal[base + f*p_Np];
                s_q[c][f][es][b][a] = (id!=-1) ? q[id] : 0.0;
              }
            }
          }
        }
      }
    }


    //============== interpolate in 3 dir ========================
    // b --> a --> c
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int c=0;c<p_Nq;++c;@inner(1)){
        for(int a=0;a<p_Nq;++a;@inner(0)){

          // #pragma unroll p_Nq
          for(int b=0;b<p_Nq;++b) {
            for(int f=0;f<p_Nfields;++f) {
              r_q[f][b] = s_q[c][f][es][b][a];
            }
          }

          // #pragma unroll p_Nq
          for(int j=0;j<p_Nq;++j){

            dfloat res1[p_Nfields];

            for(int f=0;f<p_Nfields;++f) {
              res1[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for(int b=0;b<p_Nq;++b){

              const dfloat sVjb= s_invV[j][b];
              for(int f=0;f<p_Nfields;++f) {
                res1[f] += sVjb*r_q[f][b];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_q[c][f][es][j][a] = res1[f];
            }
          }
        }
      }//for c
    }

    //transform in c
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int c=0;c<p_Nq;++c;@inner(1)){
        for(int j=0;j<p_Nq;++j;@inner(0)){

          // #pragma unroll p_Nq
          for(int a=0;a<p_Nq;++a) {
            for(int f=0;f<p_Nfields;++f) {
              r_q[f][a] = s_q[c][f][es][j][a];
            }
          }

          // #pragma unroll p_Nq
          for(int i=0;i<p_Nq;++i){

            dfloat res1[p_Nfields];

            for(int f=0;f<p_Nfields;++f) {
              res1[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for(int a=0;a<p_Nq;++a){

              const dfloat sVia = s_invV[i][a];
              for(int f=0;f<p_Nfields;++f) {
                res1[f] += sVia*r_q[f][a];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_q[c][f][es][j][i] = res1[f];
            }
          }
        }
      }
    }

    // transform in c
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          // #pragma unroll p_Nq
          for(int c=0;c<p_Nq;++c) {
            for(int f=0;f<p_Nfields;++f) {
              r_q[f][c] = s_q[c][f][es][j][i];
            }
          }

          // #pragma unroll p_Nq
          for(int k=0;k<p_Nq;++k){

            dfloat res1[p_Nfields];

            for(int f=0;f<p_Nfields;++f) {
              res1[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for(int c=0;c<p_Nq;++c){

              const dfloat sVkc = s_invV[k][c];
              for(int f=0;f<p_Nfields;++f) {
                res1[f] += sVkc*r_q[f][c];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_q[k][f][es][j][i] = res1[f];
            }
          }

          // #pragma unroll p_Nq
          for(int k=0; k<p_Nq; ++k) {
            for(int f=0;f<p_Nfields;++f) {
              r_q[f][k]=0.0;
            }
          }
        }
      }
    }

    //===============now differentiate once interpolated
    for(int k=0; k<p_Nq; ++k) {

      for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
        for(int j=0; j<p_Nq; ++j; @inner(1)) {
          for(int i=0; i<p_Nq; ++i; @inner(0)) {

            dfloat r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;

            //geofactors for k j i thread
            if (r_e<Nelements) {
              r_G00 = ggeo[p_Nggeo*element+p_G00ID];
              r_G01 = ggeo[p_Nggeo*element+p_G01ID];
              r_G02 = ggeo[p_Nggeo*element+p_G02ID];
            }

            // start with dq/dr
            dfloat dq[p_Nfields];
            for(int f=0;f<p_Nfields;++f) {
              dq[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for (int n = 0; n<p_Nq; ++n) {
              const dfloat Dr = s_D[i][n];
              for(int f=0;f<p_Nfields;++f) {
                dq[f] += Dr*s_q[k][f][es][j][n];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_qr[f][es][j][i] = r_G00*dq[f];
              s_qs[f][es][j][i] = r_G01*dq[f];
              r_qt[f] = r_G02*dq[f];
            }

            if (r_e<Nelements) {
              r_G11 = ggeo[p_Nggeo*element+p_G11ID];
              r_G12 = ggeo[p_Nggeo*element+p_G12ID];
            }

            // next dq/ds
            for(int f=0;f<p_Nfields;++f) {
              dq[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for (int n = 0; n<p_Nq; ++n) {
              const dfloat Ds = s_D[j][n];
              for(int f=0;f<p_Nfields;++f) {
                dq[f] += Ds*s_q[k][f][es][n][i];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_qr[f][es][j][i] += r_G01*dq[f];
              s_qs[f][es][j][i] += r_G11*dq[f];
              r_qt[f] += r_G12*dq[f];
            }

            if (r_e<Nelements) {
              r_G22 = ggeo[p_Nggeo*element+p_G22ID];
              r_GwJ = wJ[element];
            }

            // next dq/dt
            for(int f=0;f<p_Nfields;++f) {
              dq[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for (int n = 0; n<p_Nq; ++n) {
              const dfloat Dt = s_D[k][n];
              for(int f=0;f<p_Nfields;++f) {
                dq[f] += Dt*s_q[n][f][es][j][i];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_qr[f][es][j][i] += r_G02*dq[f];
              s_qs[f][es][j][i] += r_G12*dq[f];
              r_qt[f] += r_G22*dq[f];

              // last lambda*q
              r_q[f][k] += lambda*r_GwJ*s_q[k][f][es][j][i];
            }
          }
        }
      }

      // weak diff
      for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
        for(int j=0;j<p_Nq;++j;@inner(1)){
          for(int i=0;i<p_Nq;++i;@inner(0)){

            dfloat r_qk[p_Nfields];
            for(int f=0;f<p_Nfields;++f) {
              r_qk[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for(int n=0;n<p_Nq;++n){
              const dfloat Dr = s_D[n][i];
              for(int f=0;f<p_Nfields;++f) {
                r_qk[f] += Dr*s_qr[f][es][j][n];
              }
            }

            // #pragma unroll p_Nq
            for(int n=0;n<p_Nq;++n){
              const dfloat Ds = s_D[n][j];
              for(int f=0;f<p_Nfields;++f) {
                r_qk[f] += Ds*s_qs[f][es][n][i];
              }
            }

            // #pragma unroll p_Nq
            for(int n=0;n<p_Nq;++n){
              const dfloat Dt = s_D[k][n];
              for(int f=0;f<p_Nfields;++f) {
                r_q[f][n] += Dt*r_qt[f];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              r_q[f][k] += r_qk[f];
            }
          }
        }
      }
    }//k

    //Loop 7
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          // #pragma unroll p_Nq
          for(int k=0; k<p_Nq; ++k) {
            for(int f=0;f<p_Nfields;++f) {
              s_q[k][f][es][j][i] = r_q[f][k];
            }
          }
        }
      }
    }

    //=========== now project =================================================
    // b -> c -> a

    // transform back in b
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_Nq;++k;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          // #pragma unroll p_Nq
          for(int j=0;j<p_Nq;++j){
            for(int f=0;f<p_Nfields;++f) {
              r_q[f][j] = s_q[k][f][es][j][i];
            }
          }

          // #pragma unroll p_Nq
          for(int b=0;b<p_Nq;++b){

            dfloat res1[p_Nfields];

            for(int f=0;f<p_Nfields;++f) {
              res1[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for(int j=0;j<p_Nq;++j){
              const dfloat sVjb = s_invV[j][b];
              for(int f=0;f<p_Nfields;++f) {
                res1[f] += sVjb*r_q[f][j];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_q[k][f][es][b][i] = res1[f];
            }
          }
        }
      }
    }

    // transform back in a
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_Nq;++k;@inner(1)){
        for(int b=0;b<p_Nq;++b;@inner(0)){
          // #pragma unroll p_Nq
          for(int i=0;i<p_Nq;++i) {
            for(int f=0;f<p_Nfields;++f) {
              r_q[f][i] = s_q[k][f][es][b][i];
            }
          }

          // #pragma unroll p_Nq
          for(int a=0;a<p_Nq;++a){

            dfloat res1[p_Nfields];

            for(int f=0;f<p_Nfields;++f) {
              res1[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for(int i=0;i<p_Nq;++i){
              const dfloat sVia = s_invV[i][a];
              for(int f=0;f<p_Nfields;++f) {
                res1[f] += sVia*r_q[f][i];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_q[k][f][es][b][a] = res1[f];
            }
          }
        }
      }
    }

    // transform back in c
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int b=0;b<p_Nq;++b;@inner(1)){
        for(int a=0;a<p_Nq;++a;@inner(0)){
          if(r_e<Nelements){

            // #pragma unroll p_Nq
            for(int k=0;k<p_Nq;++k){
              for(int f=0;f<p_Nfields;++f) {
                r_q[f][k] = s_q[k][f][es][b][a];
              }
            }

            const dlong id = element*p_Np*p_Nfields + b*p_Nq + a;

            // #pragma unroll p_Nq
            for(int c=0;c<p_Nq;++c){

              dfloat res1[p_Nfields];

              for(int f=0;f<p_Nfields;++f) {
                res1[f] = 0.0;
              }

              // #pragma unroll p_Nq
              for(int k=0;k<p_Nq;++k){
                const dfloat sVkc = s_invV[k][c];
                for(int f=0;f<p_Nfields;++f) {
                  res1[f] += sVkc*r_q[f][k];
                }
              }

              for(int f=0;f<p_Nfields;++f) {
                Aq[id+f*p_Np+c*p_Nq*p_Nq] = res1[f];
              }
            }//c
          }//if
        }//a
      }//b
    }//es
  }//eo
}//kernel


#else

#if p_N==1
#define p_NelementsPerBlk 9
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
// unoptimized
#define p_NelementsPerBlk 1
#endif

@kernel void bp4AxAffineHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  wJ,
                        @restrict const  dfloat *  ggeo,
                        @restrict const  dfloat *  D,
                        @restrict const  dfloat *  I,
                        @restrict const  dfloat *  invV,
                        @restrict const  dfloat *  S,
                        @restrict const  dfloat *  MM,
                        const dfloat lambda,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

  for(int eo=0;eo<Nelements;eo+=p_NelementsPerBlk;@outer(0)){

//padding for bank conflicts
#if p_Nq==8 || p_Nq==4
#define p_pad 1
#else
#define p_pad 0
#endif

    @shared dfloat s_D [p_Nq][p_Nq+p_pad];
    @shared dfloat s_invV [p_Nq][p_Nq+p_pad];
    @shared dfloat   s_q[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];
    @shared dfloat s_Gqr[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];
    @shared dfloat s_Gqs[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];
    @shared dfloat s_Gqt[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];

    @exclusive dlong element;
    @exclusive int k, es;

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          //load operators
          if(ke==0){
            s_D[j][i] = D[j*p_Nq+i];
            s_invV[j][i] = invV[j*p_Nq+i];
          }

          k  = ke%p_Nq;
          es = ke/p_Nq;

          if(es+eo<Nelements) {
            element = elementList[es+eo];

            const dlong base = i + j*p_Nq + k*p_Nq*p_Nq + element*p_Np*p_Nfields;
            for(int f=0;f<p_Nfields;++f) {
              const dlong id = GlobalToLocal[base + f*p_Np];
              s_q[f][es][k][j][i] = (id!=-1) ? q[id] : 0.0;
            }
          }
        }
      }
    }

    // interpolate in 't'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields];
          for(int f=0;f<p_Nfields;++f) {
            tmp[f] = 0.0;
          }

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            const dfloat Vkm = s_invV[k][m];
            for(int f=0;f<p_Nfields;++f) {
              tmp[f] += Vkm*s_q[f][es][m][j][i];
            }
          }

          for(int f=0;f<p_Nfields;++f) {
            s_Gqr[f][es][k][j][i] = tmp[f];
          }
        }
      }
    }

    // interpolate in 'r'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields];
          for(int f=0;f<p_Nfields;++f) {
            tmp[f] = 0.0;
          }

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            const dfloat Vim = s_invV[i][m];
            for(int f=0;f<p_Nfields;++f) {
              tmp[f] += Vim*s_Gqr[f][es][k][j][m];
            }
          }

          for(int f=0;f<p_Nfields;++f) {
            s_Gqs[f][es][k][j][i] = tmp[f];
          }
        }
      }
    }

    // interpolate in 's'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields];
          for(int f=0;f<p_Nfields;++f) {
            tmp[f] = 0.0;
          }

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            const dfloat Vim = s_invV[j][m];
            for(int f=0;f<p_Nfields;++f) {
              tmp[f] += Vim*s_Gqs[f][es][k][m][i];
            }
          }

          for(int f=0;f<p_Nfields;++f) {
            s_q[f][es][k][j][i] = tmp[f];
          }
        }
      }
    }

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          if(eo+es<Nelements){
            dfloat tmp[p_Nfields];
            for(int f=0;f<p_Nfields;++f) {
              tmp[f] = 0.0;
            }

            // 't' terms
            // #pragma unroll p_cubUnr
            for(int m = 0; m < p_Nq; ++m) {
              const dfloat Dkm = s_D[k][m];
              for(int f=0;f<p_Nfields;++f) {
                tmp[f] += Dkm*s_q[f][es][m][j][i];
              }
            }

            const dfloat G02 = ggeo[p_Nggeo*element+p_G02ID];
            const dfloat G12 = ggeo[p_Nggeo*element+p_G12ID];
            const dfloat G22 = ggeo[p_Nggeo*element+p_G22ID];

            for(int f=0;f<p_Nfields;++f) {
              s_Gqr[f][es][k][j][i] = G02*tmp[f];
              s_Gqs[f][es][k][j][i] = G12*tmp[f];
              s_Gqt[f][es][k][j][i] = G22*tmp[f];
            }

            for(int f=0;f<p_Nfields;++f) {
              tmp[f] = 0.0;
            }

            // 'r' terms
            // #pragma unroll p_cubUnr
            for(int m = 0; m < p_Nq; ++m) {
              const dfloat Dim = s_D[i][m];
              for(int f=0;f<p_Nfields;++f) {
                tmp[f] += Dim*s_q[f][es][k][j][m];
              }
            }

            // 'r' terms
            const dfloat G00 = ggeo[p_Nggeo*element+p_G00ID];
            const dfloat G01 = ggeo[p_Nggeo*element+p_G01ID];

            for(int f=0;f<p_Nfields;++f) {
              s_Gqr[f][es][k][j][i] += G00*tmp[f];
              s_Gqs[f][es][k][j][i] += G01*tmp[f];
              s_Gqt[f][es][k][j][i] += G02*tmp[f];
            }

            for(int f=0;f<p_Nfields;++f) {
              tmp[f] = 0.0;
            }

            // 's' terms
            // #pragma unroll p_cubUnr
            for(int m = 0; m < p_Nq; ++m) {
              const dfloat Djm = s_D[j][m];
              for(int f=0;f<p_Nfields;++f) {
                tmp[f] += Djm*s_q[f][es][k][m][i];
              }
            }

            const dfloat G11 = ggeo[p_Nggeo*element+p_G11ID];

            for(int f=0;f<p_Nfields;++f) {
              s_Gqr[f][es][k][j][i] += G01*tmp[f];
              s_Gqs[f][es][k][j][i] += G11*tmp[f];
              s_Gqt[f][es][k][j][i] += G12*tmp[f];
            }
          }
        }
      }
    }

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          if(eo+es<Nelements){
            const dfloat GWJ  = wJ[element];

            dfloat tmpAp[p_Nfields];
            for(int f=0;f<p_Nfields;++f) {
              tmpAp[f] = s_q[f][es][k][j][i]*lambda*GWJ;
            }

            // use same matrix for both slices
            // #pragma unroll p_cubUnr
            for(int m=0;m<p_Nq;++m){
              const dfloat Dmi = s_D[m][i];
              const dfloat Dmj = s_D[m][j];

              for(int f=0;f<p_Nfields;++f) {
                tmpAp[f] += Dmi*s_Gqr[f][es][k][j][m];
                tmpAp[f] += Dmj*s_Gqs[f][es][k][m][i];
              }
            }

            // #pragma unroll p_cubUnr
            for(int m=0;m<p_Nq;++m){
              const dfloat Dmk = s_D[m][k];
              for(int f=0;f<p_Nfields;++f) {
                tmpAp[f] += Dmk*s_Gqt[f][es][m][j][i];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_q[f][es][k][j][i] = tmpAp[f];
            }
          }
        }
      }
    }

    // test in 's'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields];
          for(int f=0;f<p_Nfields;++f) {
            tmp[f] = 0.0;
          }

          // #pragma unroll p_cubUnr
          for(int m=0;m<p_Nq;++m){
            const dfloat Vmj = s_invV[m][j];
            for(int f=0;f<p_Nfields;++f) {
              tmp[f] += Vmj*s_q[f][es][k][m][i];
            }
          }

          for(int f=0;f<p_Nfields;++f) {
            s_Gqr[f][es][k][j][i] = tmp[f];
          }
        }
      }
    }

    // test in 'r'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields];
          for(int f=0;f<p_Nfields;++f) {
            tmp[f] = 0.0;
          }

          // #pragma unroll p_cubUnr
          for(int m=0;m<p_Nq;++m){
            const dfloat Vmj = s_invV[m][i];
            for(int f=0;f<p_Nfields;++f) {
              tmp[f] += Vmj*s_Gqr[f][es][k][j][m];
            }
          }

          for(int f=0;f<p_Nfields;++f) {
            s_Gqs[f][es][k][j][i] = tmp[f];
          }
        }
      }
    }

    // test in 't'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          if(eo+es<Nelements){

            dfloat tmp[p_Nfields];
            for(int f=0;f<p_Nfields;++f) {
              tmp[f] = 0.0;
            }

            // #pragma unroll p_cubUnr
            for(int m=0;m<p_Nq;++m){
              const dfloat Vmk = s_invV[m][k];
              for(int f=0;f<p_Nfields;++f) {
                tmp[f] += Vmk*s_Gqs[f][es][m][j][i];
              }
            }

            const dlong base = i + j*p_Nq + k*p_Nq*p_Nq + element*p_Np*p_Nfields;
            for(int f=0;f<p_Nfields;++f) {
              Aq[base+f*p_Np] = tmp[f];
            }
          }
        }
      }
    }
  }
}

#endif
