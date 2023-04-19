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

@kernel void bp4AxHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  cubwJ,
                        @restrict const  dfloat *  cubggeo,
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
#if p_cubNq==8 || p_cubNq==4
#define p_pad 1
#else
#define p_pad 0
#endif

    @shared dfloat s_q[p_cubNq][p_Nfields][p_NelementsPerBlk][p_cubNq][p_cubNq+p_pad];
    @shared dfloat s_D[p_cubNq][p_cubNq+p_pad];
    @shared dfloat s_qr[p_Nfields][p_NelementsPerBlk][p_cubNq][p_cubNq+p_pad];
    @shared dfloat s_qs[p_Nfields][p_NelementsPerBlk][p_cubNq][p_cubNq+p_pad];
    @shared dfloat s_I[p_cubNq][p_Nq];

    @exclusive dfloat r_q[p_Nfields][p_cubNq];
    @exclusive dfloat r_qt[p_Nfields];

    @exclusive dlong r_e, element;

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int b=0;b<p_cubNq;++b;@inner(1)){
        for(int a=0;a<p_cubNq;++a;@inner(0)){

          if(es==0 && a<p_Nq){
            s_I[b][a] = I[a+p_Nq*b];
          }

          if (es==0)
            s_D[b][a] = D[b*p_cubNq+a];

          r_e = eo + es;

          if(r_e<Nelements) {
            element = elementList[r_e];

            // load q
            if(a<p_Nq && b<p_Nq){
              const dlong base = a + b*p_Nq + element*p_Np*p_Nfields;

              // #pragma unroll p_Nq
              for(int c=0;c<p_Nq;++c) {
                const dlong id0 = GlobalToLocal[base + 0*p_Np + c*p_Nq*p_Nq];
                const dlong id1 = GlobalToLocal[base + 1*p_Np + c*p_Nq*p_Nq];
                const dlong id2 = GlobalToLocal[base + 2*p_Np + c*p_Nq*p_Nq];
                s_q[c][0][es][b][a] = (id0!=-1) ? q[id0] : 0.0;
                s_q[c][1][es][b][a] = (id1!=-1) ? q[id1] : 0.0;
                s_q[c][2][es][b][a] = (id2!=-1) ? q[id2] : 0.0;
              }
            }
          }
        }
      }
    }


    //============== interpolate in 3 dir ========================
    // b --> a --> c
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int c=0;c<p_cubNq;++c;@inner(1)){
        for(int a=0;a<p_cubNq;++a;@inner(0)){

          if(a<p_Nq && c<p_Nq){

            // #pragma unroll p_Nq
            for(int b=0;b<p_Nq;++b) {
              r_q[0][b] = s_q[c][0][es][b][a];
              r_q[1][b] = s_q[c][1][es][b][a];
              r_q[2][b] = s_q[c][2][es][b][a];
            }

            // #pragma unroll p_Nq
            for(int j=0;j<(p_cubNq+1)/2;++j){

              dfloat res1_0 = 0, res1_1 = 0, res1_2 = 0;
              dfloat res2_0 = 0, res2_1 = 0, res2_2 = 0;

              // #pragma unroll p_Nq
              for(int b=0;b<p_Nq;++b){

                const dfloat sIjb= s_I[j][b];
                res1_0 += sIjb*r_q[0][b];
                res1_1 += sIjb*r_q[1][b];
                res1_2 += sIjb*r_q[2][b];
                res2_0 += sIjb*r_q[0][p_Nq-1-b];
                res2_1 += sIjb*r_q[1][p_Nq-1-b];
                res2_2 += sIjb*r_q[2][p_Nq-1-b];
              }

              s_q[c][0][es][j][a] = res1_0;
              s_q[c][1][es][j][a] = res1_1;
              s_q[c][2][es][j][a] = res1_2;
              s_q[c][0][es][p_cubNq-1-j][a] = res2_0;
              s_q[c][1][es][p_cubNq-1-j][a] = res2_1;
              s_q[c][2][es][p_cubNq-1-j][a] = res2_2;
            }
          }
        }
      }//for c
    }

    //transform in c
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int c=0;c<p_cubNq;++c;@inner(1)){
        for(int j=0;j<p_cubNq;++j;@inner(0)){

          if(c<p_Nq){

            // #pragma unroll p_Nq
            for(int a=0;a<p_Nq;++a) {
              r_q[0][a] = s_q[c][0][es][j][a];
              r_q[1][a] = s_q[c][1][es][j][a];
              r_q[2][a] = s_q[c][2][es][j][a];
            }

            // #pragma unroll p_Nq
            for(int i=0;i<(p_cubNq+1)/2;++i){

              dfloat res1_0 = 0, res1_1 = 0, res1_2 = 0;
              dfloat res2_0 = 0, res2_1 = 0, res2_2 = 0;

              // #pragma unroll p_Nq
              for(int a=0;a<p_Nq;++a){

                const dfloat sIia = s_I[i][a];
                res1_0 += sIia*r_q[0][a];
                res1_1 += sIia*r_q[1][a];
                res1_2 += sIia*r_q[2][a];
                res2_0 += sIia*r_q[0][p_Nq-1-a];
                res2_1 += sIia*r_q[1][p_Nq-1-a];
                res2_2 += sIia*r_q[2][p_Nq-1-a];
              }

              s_q[c][0][es][j][i] = res1_0;
              s_q[c][1][es][j][i] = res1_1;
              s_q[c][2][es][j][i] = res1_2;
              s_q[c][0][es][j][p_cubNq-1-i] = res2_0;
              s_q[c][1][es][j][p_cubNq-1-i] = res2_1;
              s_q[c][2][es][j][p_cubNq-1-i] = res2_2;
            }
          }
        }
      }
    }

    // transform in c
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          // #pragma unroll p_Nq
          for(int c=0;c<p_Nq;++c) {
            r_q[0][c] = s_q[c][0][es][j][i];
            r_q[1][c] = s_q[c][1][es][j][i];
            r_q[2][c] = s_q[c][2][es][j][i];
          }

          // #pragma unroll p_Nq
          for(int k=0;k<(p_cubNq+1)/2;++k){

            dfloat res1_0 = 0, res1_1 = 0, res1_2 = 0;
            dfloat res2_0 = 0, res2_1 = 0, res2_2 = 0;

            // #pragma unroll p_Nq
            for(int c=0;c<p_Nq;++c){

              const dfloat sIkc = s_I[k][c];
              res1_0 += sIkc*r_q[0][c];
              res1_1 += sIkc*r_q[1][c];
              res1_2 += sIkc*r_q[2][c];
              res2_0 += sIkc*r_q[0][p_Nq-1-c];
              res2_1 += sIkc*r_q[1][p_Nq-1-c];
              res2_2 += sIkc*r_q[2][p_Nq-1-c];
            }

            s_q[k][0][es][j][i] = res1_0;
            s_q[k][1][es][j][i] = res1_1;
            s_q[k][2][es][j][i] = res1_2;
            s_q[p_cubNq-k-1][0][es][j][i] = res2_0;
            s_q[p_cubNq-k-1][1][es][j][i] = res2_1;
            s_q[p_cubNq-k-1][2][es][j][i] = res2_2;
          }

          // #pragma unroll p_cubNq
          for(int k=0; k<p_cubNq; ++k) {
            r_q[0][k]=0.0f;
            r_q[1][k]=0.0f;
            r_q[2][k]=0.0f;
          }
        }
      }
    }

    //===============now differentiate once interpolated
    for(int k=0; k<p_cubNq; ++k) {

      for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
        for(int j=0; j<p_cubNq; ++j; @inner(1)) {
          for(int i=0; i<p_cubNq; ++i; @inner(0)) {

            dfloat r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
            const dlong base = element*p_cubNp + k*p_cubNq*p_cubNq + j*p_cubNq + i;

            //geofactors for k j i thread
            if (r_e<Nelements) {
              r_G00 = cubggeo[p_Nggeo*base+p_G00ID];
              r_G01 = cubggeo[p_Nggeo*base+p_G01ID];
              r_G02 = cubggeo[p_Nggeo*base+p_G02ID];
            }

            // start with dq/dr
            dfloat dq0 = 0.0f, dq1 = 0.0f, dq2 = 0.0f;

            // #pragma unroll p_cubNq
            for (int n = 0; n<p_cubNq; ++n) {
              const dfloat Dr = s_D[i][n];
              dq0 += Dr*s_q[k][0][es][j][n];
              dq1 += Dr*s_q[k][1][es][j][n];
              dq2 += Dr*s_q[k][2][es][j][n];
            }

            s_qr[0][es][j][i] = r_G00*dq0;
            s_qr[1][es][j][i] = r_G00*dq1;
            s_qr[2][es][j][i] = r_G00*dq2;
            s_qs[0][es][j][i] = r_G01*dq0;
            s_qs[1][es][j][i] = r_G01*dq1;
            s_qs[2][es][j][i] = r_G01*dq2;
            r_qt[0] = r_G02*dq0;
            r_qt[1] = r_G02*dq1;
            r_qt[2] = r_G02*dq2;

            if (r_e<Nelements) {
              r_G11 = cubggeo[p_Nggeo*base+p_G11ID];
              r_G12 = cubggeo[p_Nggeo*base+p_G12ID];
            }

            // next dq/ds
            dq0 = 0.0f, dq1 = 0.0f, dq2 = 0.0f;

            // #pragma unroll p_cubNq
            for (int n = 0; n<p_cubNq; ++n) {
              const dfloat Ds = s_D[j][n];
              dq0 += Ds*s_q[k][0][es][n][i];
              dq1 += Ds*s_q[k][1][es][n][i];
              dq2 += Ds*s_q[k][2][es][n][i];
            }

            s_qr[0][es][j][i] += r_G01*dq0;
            s_qr[1][es][j][i] += r_G01*dq1;
            s_qr[2][es][j][i] += r_G01*dq2;
            s_qs[0][es][j][i] += r_G11*dq0;
            s_qs[1][es][j][i] += r_G11*dq1;
            s_qs[2][es][j][i] += r_G11*dq2;
            r_qt[0] += r_G12*dq0;
            r_qt[1] += r_G12*dq1;
            r_qt[2] += r_G12*dq2;

            if (r_e<Nelements) {
              r_G22 = cubggeo[p_Nggeo*base+p_G22ID];
              r_GwJ = cubwJ[base];
            }

            // next dq/dt
            dq0 = 0.0f, dq1 = 0.0f, dq2 = 0.0f;

            // #pragma unroll p_cubNq
            for (int n = 0; n<p_cubNq; ++n) {
              const dfloat Dt = s_D[k][n];
              dq0 += Dt*s_q[n][0][es][j][i];
              dq1 += Dt*s_q[n][1][es][j][i];
              dq2 += Dt*s_q[n][2][es][j][i];
            }

            s_qr[0][es][j][i] += r_G02*dq0;
            s_qr[1][es][j][i] += r_G02*dq1;
            s_qr[2][es][j][i] += r_G02*dq2;
            s_qs[0][es][j][i] += r_G12*dq0;
            s_qs[1][es][j][i] += r_G12*dq1;
            s_qs[2][es][j][i] += r_G12*dq2;
            r_qt[0] += r_G22*dq0;
            r_qt[1] += r_G22*dq1;
            r_qt[2] += r_G22*dq2;

            // last lambda*q
            r_q[0][k] += lambda*r_GwJ*s_q[k][0][es][j][i];
            r_q[1][k] += lambda*r_GwJ*s_q[k][1][es][j][i];
            r_q[2][k] += lambda*r_GwJ*s_q[k][2][es][j][i];
          }
        }
      }

      // weak diff
      for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
        for(int j=0;j<p_cubNq;++j;@inner(1)){
          for(int i=0;i<p_cubNq;++i;@inner(0)){
            dfloat r_qk0=0.0, r_qk1=0.0, r_qk2=0.0;
            // #pragma unroll p_cubNq
            for(int n=0;n<p_cubNq;++n){
              const dfloat Dr = s_D[n][i];
              r_qk0 += Dr*s_qr[0][es][j][n];
              r_qk1 += Dr*s_qr[1][es][j][n];
              r_qk2 += Dr*s_qr[2][es][j][n];
            }

            // #pragma unroll p_cubNq
            for(int n=0;n<p_cubNq;++n){
              const dfloat Ds = s_D[n][j];
              r_qk0 += Ds*s_qs[0][es][n][i];
              r_qk1 += Ds*s_qs[1][es][n][i];
              r_qk2 += Ds*s_qs[2][es][n][i];
            }

            // #pragma unroll p_cubNq
            for(int n=0;n<p_cubNq;++n){
              const dfloat Dt = s_D[k][n];
              r_q[0][n] += Dt*r_qt[0];
              r_q[1][n] += Dt*r_qt[1];
              r_q[2][n] += Dt*r_qt[2];
            }

            r_q[0][k] += r_qk0;
            r_q[1][k] += r_qk1;
            r_q[2][k] += r_qk2;
          }
        }
      }
    }//k

    //Loop 7
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          // #pragma unroll p_cubNq
          for(int k=0; k<p_cubNq; ++k) {
            s_q[k][0][es][j][i] = r_q[0][k];
            s_q[k][1][es][j][i] = r_q[1][k];
            s_q[k][2][es][j][i] = r_q[2][k];
          }
        }
      }
    }

    //=========== now project =================================================
    // b -> c -> a

    // transform back in b
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_cubNq;++k;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          // #pragma unroll p_cubNq
          for(int j=0;j<p_cubNq;++j){
            r_q[0][j] = s_q[k][0][es][j][i];
            r_q[1][j] = s_q[k][1][es][j][i];
            r_q[2][j] = s_q[k][2][es][j][i];
          }

          // #pragma unroll p_Nq
          for(int b=0;b<(p_Nq+1)/2;++b){
            dfloat res1_0 = 0, res1_1 = 0, res1_2 = 0;
            dfloat res2_0 = 0, res2_1 = 0, res2_2 = 0;

            // #pragma unroll p_cubNq
            for(int j=0;j<p_cubNq;++j){
              const dfloat sIjb = s_I[j][b];
              res1_0 += sIjb*r_q[0][j];
              res1_1 += sIjb*r_q[1][j];
              res1_2 += sIjb*r_q[2][j];
              res2_0 += sIjb*r_q[0][p_cubNq-1-j];
              res2_1 += sIjb*r_q[1][p_cubNq-1-j];
              res2_2 += sIjb*r_q[2][p_cubNq-1-j];
            }

            s_q[k][0][es][b][i] = res1_0;
            s_q[k][1][es][b][i] = res1_1;
            s_q[k][2][es][b][i] = res1_2;
            s_q[k][0][es][p_Nq-1-b][i] = res2_0;
            s_q[k][1][es][p_Nq-1-b][i] = res2_1;
            s_q[k][2][es][p_Nq-1-b][i] = res2_2;
          }
        }
      }
    }

    // transform back in a
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_cubNq;++k;@inner(1)){
        for(int b=0;b<p_cubNq;++b;@inner(0)){
          if(b<p_Nq){
            // #pragma unroll p_cubNq
            for(int i=0;i<p_cubNq;++i) {
              r_q[0][i] = s_q[k][0][es][b][i];
              r_q[1][i] = s_q[k][1][es][b][i];
              r_q[2][i] = s_q[k][2][es][b][i];
            }

            // #pragma unroll p_Nq
            for(int a=0;a<(p_Nq+1)/2;++a){

              dfloat res1_0 = 0, res1_1 = 0, res1_2 = 0;
              dfloat res2_0 = 0, res2_1 = 0, res2_2 = 0;

              // #pragma unroll p_cubNq
              for(int i=0;i<p_cubNq;++i){
                const dfloat sIia = s_I[i][a];
                res1_0 += sIia*r_q[0][i];
                res1_1 += sIia*r_q[1][i];
                res1_2 += sIia*r_q[2][i];
                res2_0 += sIia*r_q[0][p_cubNq-1-i];
                res2_1 += sIia*r_q[1][p_cubNq-1-i];
                res2_2 += sIia*r_q[2][p_cubNq-1-i];
              }

              s_q[k][0][es][b][a] = res1_0;
              s_q[k][1][es][b][a] = res1_1;
              s_q[k][2][es][b][a] = res1_2;
              s_q[k][0][es][b][p_Nq-1-a] = res2_0;
              s_q[k][1][es][b][p_Nq-1-a] = res2_1;
              s_q[k][2][es][b][p_Nq-1-a] = res2_2;
            }
          }
        }
      }
    }

    // transform back in c
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int b=0;b<p_cubNq;++b;@inner(1)){
        for(int a=0;a<p_cubNq;++a;@inner(0)){
          if(r_e<Nelements && a<p_Nq && b<p_Nq){

            // #pragma unroll p_cubNq
            for(int k=0;k<p_cubNq;++k){
              r_q[0][k] = s_q[k][0][es][b][a];
              r_q[1][k] = s_q[k][1][es][b][a];
              r_q[2][k] = s_q[k][2][es][b][a];
            }

            const dlong id = element*p_Np*p_Nfields + b*p_Nq + a;

            // #pragma unroll p_Nq
            for(int c=0;c<(p_Nq+1)/2;++c){

              dfloat res1_0 = 0, res1_1 = 0, res1_2 = 0;
              dfloat res2_0 = 0, res2_1 = 0, res2_2 = 0;

              // #pragma unroll p_cubNq
              for(int k=0;k<p_cubNq;++k){
                const dfloat sIkc = s_I[k][c];
                res1_0 += sIkc*r_q[0][k];
                res1_1 += sIkc*r_q[1][k];
                res1_2 += sIkc*r_q[2][k];
                res2_0 += sIkc*r_q[0][p_cubNq-1-k];
                res2_1 += sIkc*r_q[1][p_cubNq-1-k];
                res2_2 += sIkc*r_q[2][p_cubNq-1-k];
              }

              Aq[id+0*p_Np+c*p_Nq*p_Nq] = res1_0;
              Aq[id+0*p_Np+(p_Nq-1-c)*p_Nq*p_Nq] = res2_0;
              Aq[id+1*p_Np+c*p_Nq*p_Nq] = res1_1;
              Aq[id+1*p_Np+(p_Nq-1-c)*p_Nq*p_Nq] = res2_1;
              Aq[id+2*p_Np+c*p_Nq*p_Nq] = res1_2;
              Aq[id+2*p_Np+(p_Nq-1-c)*p_Nq*p_Nq] = res2_2;
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

@kernel void bp4AxHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  cubwJ,
                        @restrict const  dfloat *  cubggeo,
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
#if p_cubNq==8 || p_cubNq==4
#define p_pad 1
#else
#define p_pad 0
#endif

    @shared dfloat s_D [p_cubNq][p_cubNq+p_pad];
    @shared dfloat s_I [p_cubNq][p_cubNq+p_pad];
    @shared dfloat   s_q[p_Nfields][p_NelementsPerBlk][p_cubNq][p_cubNq][p_cubNq+p_pad];
    @shared dfloat s_Gqr[p_Nfields][p_NelementsPerBlk][p_cubNq][p_cubNq][p_cubNq+p_pad];
    @shared dfloat s_Gqs[p_Nfields][p_NelementsPerBlk][p_cubNq][p_cubNq][p_cubNq+p_pad];
    @shared dfloat s_Gqt[p_Nfields][p_NelementsPerBlk][p_cubNq][p_cubNq][p_cubNq+p_pad];

    @exclusive dlong element;
    @exclusive int k, es;

    for(int ke=0;ke<p_cubNq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          //load operators
          if(ke==0){
            const int id = j*p_cubNq+i;
            const dfloat Dji = D[id];
            s_D[j][i] = Dji;

            if(i<p_Nq){
              const int id = j*p_Nq+i;
              const dfloat Iji = I[id];
              s_I[j][i] = Iji;
            }
          }

          k  = ke%p_cubNq;
          es = ke/p_cubNq;

          if(es+eo<Nelements) {
            element = elementList[es+eo];
            if(i<p_Nq && j<p_Nq && k<p_Nq){
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
    }

    // interpolate in 't'
    for(int ke=0;ke<p_cubNq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){
          if(i<p_Nq && j<p_Nq){

            dfloat tmp0=0.0;
            dfloat tmp1=0.0;
            dfloat tmp2=0.0;

            // #pragma unroll p_Nq
            for(int m=0;m<p_Nq;++m){
              const dfloat Ikm = s_I[k][m];
              tmp0 += Ikm*s_q[0][es][m][j][i];
              tmp1 += Ikm*s_q[1][es][m][j][i];
              tmp2 += Ikm*s_q[2][es][m][j][i];
            }

            s_Gqr[0][es][k][j][i] = tmp0;
            s_Gqr[1][es][k][j][i] = tmp1;
            s_Gqr[2][es][k][j][i] = tmp2;
          }
        }
      }
    }

    // interpolate in 'r'
    for(int ke=0;ke<p_cubNq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){
          if(j<p_Nq){

            dfloat tmp0 =0.0;
            dfloat tmp1 =0.0;
            dfloat tmp2 =0.0;

            // #pragma unroll p_Nq
            for(int m=0;m<p_Nq;++m){
              const dfloat Iim = s_I[i][m];
              tmp0 += Iim*s_Gqr[0][es][k][j][m];
              tmp1 += Iim*s_Gqr[1][es][k][j][m];
              tmp2 += Iim*s_Gqr[2][es][k][j][m];
            }

            s_Gqs[0][es][k][j][i] = tmp0;
            s_Gqs[1][es][k][j][i] = tmp1;
            s_Gqs[2][es][k][j][i] = tmp2;
          }
        }
      }
    }

    // interpolate in 's'
    for(int ke=0;ke<p_cubNq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          dfloat tmp0 =0.0;
          dfloat tmp1 =0.0;
          dfloat tmp2 =0.0;

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            const dfloat Iim = s_I[j][m];
            tmp0 += Iim*s_Gqs[0][es][k][m][i];
            tmp1 += Iim*s_Gqs[1][es][k][m][i];
            tmp2 += Iim*s_Gqs[2][es][k][m][i];
          }

          s_q[0][es][k][j][i] = tmp0;
          s_q[1][es][k][j][i] = tmp1;
          s_q[2][es][k][j][i] = tmp2;
        }
      }
    }

    for(int ke=0;ke<p_cubNq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          if(eo+es<Nelements){
            dfloat tmp0=0.0;
            dfloat tmp1=0.0;
            dfloat tmp2=0.0;

            // 't' terms
            // #pragma unroll p_cubUnr
            for(int m = 0; m < p_cubNq; ++m) {
              const dfloat Dkm = s_D[k][m];
              tmp0 += Dkm*s_q[0][es][m][j][i];
              tmp1 += Dkm*s_q[1][es][m][j][i];
              tmp2 += Dkm*s_q[2][es][m][j][i];
            }

            const dlong gbase = element*p_cubNp + i + j*p_cubNq + k*p_cubNq*p_cubNq;
            const dfloat G02 = cubggeo[p_Nggeo*gbase+p_G02ID];
            const dfloat G12 = cubggeo[p_Nggeo*gbase+p_G12ID];
            const dfloat G22 = cubggeo[p_Nggeo*gbase+p_G22ID];

            s_Gqr[0][es][k][j][i] = G02*tmp0;
            s_Gqr[1][es][k][j][i] = G02*tmp1;
            s_Gqr[2][es][k][j][i] = G02*tmp2;
            s_Gqs[0][es][k][j][i] = G12*tmp0;
            s_Gqs[1][es][k][j][i] = G12*tmp1;
            s_Gqs[2][es][k][j][i] = G12*tmp2;
            s_Gqt[0][es][k][j][i] = G22*tmp0;
            s_Gqt[1][es][k][j][i] = G22*tmp1;
            s_Gqt[2][es][k][j][i] = G22*tmp2;

            tmp0 = 0;
            tmp1 = 0;
            tmp2 = 0;

            // 'r' terms
            // #pragma unroll p_cubUnr
            for(int m = 0; m < p_cubNq; ++m) {
              const dfloat Dim = s_D[i][m];
              tmp0 += Dim*s_q[0][es][k][j][m];
              tmp1 += Dim*s_q[1][es][k][j][m];
              tmp2 += Dim*s_q[2][es][k][j][m];
            }

            // 'r' terms
            const dfloat G00 = cubggeo[p_Nggeo*gbase+p_G00ID];
            const dfloat G01 = cubggeo[p_Nggeo*gbase+p_G01ID];

            s_Gqr[0][es][k][j][i] += G00*tmp0;
            s_Gqr[1][es][k][j][i] += G00*tmp1;
            s_Gqr[2][es][k][j][i] += G00*tmp2;
            s_Gqs[0][es][k][j][i] += G01*tmp0;
            s_Gqs[1][es][k][j][i] += G01*tmp1;
            s_Gqs[2][es][k][j][i] += G01*tmp2;
            s_Gqt[0][es][k][j][i] += G02*tmp0;
            s_Gqt[1][es][k][j][i] += G02*tmp1;
            s_Gqt[2][es][k][j][i] += G02*tmp2;

            tmp0 = 0;
            tmp1 = 0;
            tmp2 = 0;

            // 's' terms
            // #pragma unroll p_cubUnr
            for(int m = 0; m < p_cubNq; ++m) {
              const dfloat Djm = s_D[j][m];
              tmp0 += Djm*s_q[0][es][k][m][i];
              tmp1 += Djm*s_q[1][es][k][m][i];
              tmp2 += Djm*s_q[2][es][k][m][i];
            }

            const dfloat G11 = cubggeo[p_Nggeo*gbase+p_G11ID];

            s_Gqr[0][es][k][j][i] += G01*tmp0;
            s_Gqr[1][es][k][j][i] += G01*tmp1;
            s_Gqr[2][es][k][j][i] += G01*tmp2;
            s_Gqs[0][es][k][j][i] += G11*tmp0;
            s_Gqs[1][es][k][j][i] += G11*tmp1;
            s_Gqs[2][es][k][j][i] += G11*tmp2;
            s_Gqt[0][es][k][j][i] += G12*tmp0;
            s_Gqt[1][es][k][j][i] += G12*tmp1;
            s_Gqt[2][es][k][j][i] += G12*tmp2;
          }
        }
      }
    }

    for(int ke=0;ke<p_cubNq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          if(eo+es<Nelements){
            const dlong gbase = element*p_cubNp + i + j*p_cubNq + k*p_cubNq*p_cubNq;
            const dfloat GWJ  = cubwJ[gbase];
            dfloat tmpAp0 = s_q[0][es][k][j][i]*lambda*GWJ;
            dfloat tmpAp1 = s_q[1][es][k][j][i]*lambda*GWJ;
            dfloat tmpAp2 = s_q[2][es][k][j][i]*lambda*GWJ;

            // use same matrix for both slices
            // #pragma unroll p_cubUnr
            for(int m=0;m<p_cubNq;++m){
              const dfloat Dmi = s_D[m][i];
              const dfloat Dmj = s_D[m][j];

              tmpAp0 += Dmi*s_Gqr[0][es][k][j][m];
              tmpAp1 += Dmi*s_Gqr[1][es][k][j][m];
              tmpAp2 += Dmi*s_Gqr[2][es][k][j][m];
              tmpAp0 += Dmj*s_Gqs[0][es][k][m][i];
              tmpAp1 += Dmj*s_Gqs[1][es][k][m][i];
              tmpAp2 += Dmj*s_Gqs[2][es][k][m][i];
            }

            // #pragma unroll p_cubUnr
            for(int m=0;m<p_cubNq;++m){
              const dfloat Dmk = s_D[m][k];
              tmpAp0 += Dmk*s_Gqt[0][es][m][j][i];
              tmpAp1 += Dmk*s_Gqt[1][es][m][j][i];
              tmpAp2 += Dmk*s_Gqt[2][es][m][j][i];
            }

            s_q[0][es][k][j][i] = tmpAp0;
            s_q[1][es][k][j][i] = tmpAp1;
            s_q[2][es][k][j][i] = tmpAp2;
          }
        }
      }
    }

    // test in 's'
    for(int ke=0;ke<p_cubNq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){
          if(j<p_Nq){

            dfloat tmp0=0.0;
            dfloat tmp1=0.0;
            dfloat tmp2=0.0;

            // #pragma unroll p_cubUnr
            for(int m=0;m<p_cubNq;++m){
              const dfloat Imj = s_I[m][j];
              tmp0 += Imj*s_q[0][es][k][m][i];
              tmp1 += Imj*s_q[1][es][k][m][i];
              tmp2 += Imj*s_q[2][es][k][m][i];
            }

            s_Gqr[0][es][k][j][i] = tmp0;
            s_Gqr[1][es][k][j][i] = tmp1;
            s_Gqr[2][es][k][j][i] = tmp2;
          }
        }
      }
    }

    // test in 'r'
    for(int ke=0;ke<p_cubNq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){
          if(i<p_Nq && j<p_Nq){

            dfloat tmp0=0.0;
            dfloat tmp1=0.0;
            dfloat tmp2=0.0;

            // #pragma unroll p_cubUnr
            for(int m=0;m<p_cubNq;++m){
              const dfloat Imj = s_I[m][i];
              tmp0 += Imj*s_Gqr[0][es][k][j][m];
              tmp1 += Imj*s_Gqr[1][es][k][j][m];
              tmp2 += Imj*s_Gqr[2][es][k][j][m];
            }

            s_Gqs[0][es][k][j][i] = tmp0;
            s_Gqs[1][es][k][j][i] = tmp1;
            s_Gqs[2][es][k][j][i] = tmp2;
          }
        }
      }
    }

    // test in 't'
    for(int ke=0;ke<p_cubNq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          if(eo+es<Nelements){
            if(i<p_Nq && j<p_Nq && k<p_Nq){

              dfloat tmp0=0.0;
              dfloat tmp1=0.0;
              dfloat tmp2=0.0;

              // #pragma unroll p_cubUnr
              for(int m=0;m<p_cubNq;++m){
                const dfloat Imk = s_I[m][k];
                tmp0 += Imk*s_Gqs[0][es][m][j][i];
                tmp1 += Imk*s_Gqs[1][es][m][j][i];
                tmp2 += Imk*s_Gqs[2][es][m][j][i];
              }

              const dlong base = i + j*p_Nq + k*p_Nq*p_Nq + element*p_Np*p_Nfields;
              Aq[base+0*p_Np] = tmp0;
              Aq[base+1*p_Np] = tmp1;
              Aq[base+2*p_Np] = tmp2;
            }
          }
        }
      }
    }
  }
}

#endif
