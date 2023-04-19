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

/* Use slice-by-slice kernel for all orders*/
#if p_N>0

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
#define p_NelementsPerBlk 4
#elif p_N==6
#define p_NelementsPerBlk 3
#elif p_N==7
#define p_NelementsPerBlk 2
#elif p_N==8
#define p_NelementsPerBlk 2
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

//padding for bank conflicts
#if p_cubNq==8 || p_cubNq==4
#define p_pad 1
#else
#define p_pad 0
#endif


@kernel void bp2AxHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  cubwJ,
                        @restrict const  dfloat *  I,
                        @restrict const  dfloat *  MM,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

  for(dlong eo=0; eo<Nelements; eo+=p_NelementsPerBlk; @outer(0)){

    @shared dfloat s_I[p_cubNq][p_Nq+p_pad];
    @shared dfloat s_q[p_cubNq][p_Nfields][p_NelementsPerBlk][p_cubNq][p_cubNq+p_pad];

    @exclusive dlong r_e, element;

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int b=0;b<p_cubNq;++b;@inner(1)){
        for(int a=0;a<p_cubNq;++a;@inner(0)){

          if(es==0 && a<p_Nq)
            s_I[b][a] = I[a+p_Nq*b];

          r_e = eo + es;

          if(r_e<Nelements) {
            element = elementList[r_e];

            if (a<p_Nq && b<p_Nq){
              // load pencil of u into register
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

    // transform in b
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int c=0;c<p_cubNq;++c;@inner(1)){
        for(int a=0;a<p_cubNq;++a;@inner(0)){

          dfloat r_q[p_Nfields][p_cubNq];

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

                const dfloat rI = s_I[j][b];

                res1_0 += rI*r_q[0][b];
                res1_1 += rI*r_q[1][b];
                res1_2 += rI*r_q[2][b];
                res2_0 += rI*r_q[0][p_Nq-1-b];
                res2_1 += rI*r_q[1][p_Nq-1-b];
                res2_2 += rI*r_q[2][p_Nq-1-b];
              }

              // ok since only this thread
              s_q[c][0][es][j][a] = res1_0;
              s_q[c][1][es][j][a] = res1_1;
              s_q[c][2][es][j][a] = res1_2;
              s_q[c][0][es][p_cubNq-1-j][a] = res2_0;
              s_q[c][1][es][p_cubNq-1-j][a] = res2_1;
              s_q[c][2][es][p_cubNq-1-j][a] = res2_2;
            }
          }
        }
      }
    }

    // transform in a
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int c=0;c<p_cubNq;++c;@inner(1)){
        for(int j=0;j<p_cubNq;++j;@inner(0)){

          dfloat r_q[p_Nfields][p_cubNq];

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

                const dfloat rI = s_I[i][a];

                res1_0 += rI*r_q[0][a];
                res1_1 += rI*r_q[1][a];
                res1_2 += rI*r_q[2][a];
                res2_0 += rI*r_q[0][p_Nq-1-a];
                res2_1 += rI*r_q[1][p_Nq-1-a];
                res2_2 += rI*r_q[2][p_Nq-1-a];
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

          dfloat r_q[p_Nfields][p_cubNq];

          // #pragma unroll p_Nq
          for(int c=0;c<p_Nq;++c) {
            r_q[0][c] = s_q[c][0][es][j][i];
            r_q[1][c] = s_q[c][1][es][j][i];
            r_q[2][c] = s_q[c][2][es][j][i];
          }

          // #pragma unroll p_Nq
          for(int k=0;k<(p_cubNq+1)/2;++k){

            dfloat r_GwJ;
            dfloat r_GwJ2;
            if (r_e<Nelements) {
              const dlong id  = element*p_cubNp+k*p_cubNq*p_cubNq+j*p_cubNq+i;
              const dlong id2 = element*p_cubNp+(p_cubNq-1-k)*p_cubNq*p_cubNq+j*p_cubNq+i;
              r_GwJ  = cubwJ[id ];
              r_GwJ2 = cubwJ[id2];
            }

            dfloat res1_0 = 0, res1_1 = 0, res1_2 = 0;
            dfloat res2_0 = 0, res2_1 = 0, res2_2 = 0;

            // #pragma unroll p_Nq
            for(int c=0;c<p_Nq;++c){

              const dfloat rI = s_I[k][c];

              res1_0 += rI*r_q[0][c];
              res1_1 += rI*r_q[1][c];
              res1_2 += rI*r_q[2][c];
              res2_0 += rI*r_q[0][p_Nq-1-c];
              res2_1 += rI*r_q[1][p_Nq-1-c];
              res2_2 += rI*r_q[2][p_Nq-1-c];
            }

            // ok since only this thread
            s_q[k][0][es][j][i] = r_GwJ*res1_0;
            s_q[k][1][es][j][i] = r_GwJ*res1_1;
            s_q[k][2][es][j][i] = r_GwJ*res1_2;
            s_q[p_cubNq-k-1][0][es][j][i] = r_GwJ2*res2_0;
            s_q[p_cubNq-k-1][1][es][j][i] = r_GwJ2*res2_1;
            s_q[p_cubNq-k-1][2][es][j][i] = r_GwJ2*res2_2;
          }
        }
      }
    }

    // transform back in b
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_cubNq;++k;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          dfloat r_q[p_Nfields][p_cubNq];

          // #pragma unroll p_cubNq
          for(int j=0;j<p_cubNq;++j) {
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

              const dfloat rI = s_I[j][b];

              res1_0 += rI*r_q[0][j];
              res1_1 += rI*r_q[1][j];
              res1_2 += rI*r_q[2][j];
              res2_0 += rI*r_q[0][p_cubNq-1-j];
              res2_1 += rI*r_q[1][p_cubNq-1-j];
              res2_2 += rI*r_q[2][p_cubNq-1-j];
            }

            // ok since only this thread
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

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_cubNq;++k;@inner(1)){
        for(int b=0;b<p_cubNq;++b;@inner(0)){

          dfloat r_q[p_Nfields][p_cubNq];

          if(b<p_Nq){

            // #pragma unroll p_cubNq
            for(int i=0;i<p_cubNq;++i) {
              r_q[0][i] = s_q[k][0][es][b][i];
              r_q[1][i] = s_q[k][1][es][b][i];
              r_q[2][i] = s_q[k][2][es][b][i];
            }

            // #pragma unroll p_cubNq
            for(int a=0;a<(p_Nq+1)/2;++a){

              dfloat res1_0 = 0, res1_1 = 0, res1_2 = 0;
              dfloat res2_0 = 0, res2_1 = 0, res2_2 = 0;

              // #pragma unroll p_cubNq
              for(int i=0;i<p_cubNq;++i){

                const dfloat rI = s_I[i][a];

                res1_0 += rI*r_q[0][i];
                res1_1 += rI*r_q[1][i];
                res1_2 += rI*r_q[2][i];
                res2_0 += rI*r_q[0][p_cubNq-1-i];
                res2_1 += rI*r_q[1][p_cubNq-1-i];
                res2_2 += rI*r_q[2][p_cubNq-1-i];
              }

              // ok since only this thread
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

          dfloat r_q[p_Nfields][p_cubNq];

          if(r_e<Nelements && a<p_Nq && b<p_Nq){

            // #pragma unroll p_cubNq
            for(int k=0;k<p_cubNq;++k) {
              r_q[0][k] = s_q[k][0][es][b][a];
              r_q[1][k] = s_q[k][1][es][b][a];
              r_q[2][k] = s_q[k][2][es][b][a];
            }

            const dlong id = element*p_Np*p_Nfields + b*p_Nq + a;
            // #pragma unroll p_cubNq
            for(int c=0;c<(p_Nq+1)/2;++c){

              dfloat res1_0 = 0, res1_1 = 0, res1_2 = 0;
              dfloat res2_0 = 0, res2_1 = 0, res2_2 = 0;

              // #pragma unroll p_cubNq
              for(int k=0;k<p_cubNq;++k){

                const dfloat rI = s_I[k][c];

                res1_0 += rI*r_q[0][k];
                res1_1 += rI*r_q[1][k];
                res1_2 += rI*r_q[2][k];
                res2_0 += rI*r_q[0][p_cubNq-1-k];
                res2_1 += rI*r_q[1][p_cubNq-1-k];
                res2_2 += rI*r_q[2][p_cubNq-1-k];
              }

              Aq[id+0*p_Np+c*p_Nq*p_Nq] = res1_0;
              Aq[id+0*p_Np+(p_Nq-1-c)*p_Nq*p_Nq] = res2_0;
              Aq[id+1*p_Np+c*p_Nq*p_Nq] = res1_1;
              Aq[id+1*p_Np+(p_Nq-1-c)*p_Nq*p_Nq] = res2_1;
              Aq[id+2*p_Np+c*p_Nq*p_Nq] = res1_2;
              Aq[id+2*p_Np+(p_Nq-1-c)*p_Nq*p_Nq] = res2_2;
            }
          }
        }
      }
    }
  }
}



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
#else
// unoptimized
#define p_NelementsPerBlk 1
#endif

@kernel void bp2AxHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  cubwJ,
                        @restrict const  dfloat *  I,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

//padding for bank conflicts
#if p_cubNq==8 || p_cubNq==4
#define p_pad 1
#else
#define p_pad 0
#endif

  for(dlong eo=0; eo<Nelements; eo+=p_NelementsPerBlk; @outer(0)){

    @shared dfloat s_Iq[p_cubNq][p_Nfields][p_NelementsPerBlk][p_cubNq][p_cubNq+p_pad];

    @shared dfloat s_I[p_cubNq*p_Nq];
    @shared dfloat s_IT[p_Nq*p_cubNq];

    // heavy on registers (FP64, 2*3*8 for N=7)
    @exclusive dfloat r_Aq[p_Nfields][p_cubNq];

    @exclusive dlong element, r_e;

    // array of threads
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          if(es==0){
            if(i<p_Nq){
              dfloat Iji = I[p_Nq*j+i];
              s_I[j*p_Nq+i] = Iji;
              s_IT[i*p_cubNq+j] = Iji;
            }
          }

          r_e = eo + es;
          if(r_e<Nelements){
            element = elementList[r_e];

            // load pencil of u into register
            const dlong base = i + j*p_Nq + element*p_Np*p_Nfields;

            if(i<p_Nq && j<p_Nq){
              for(int k = 0; k < p_Nq; k++) {
                const dlong id0 = GlobalToLocal[base + 0*p_Np + k*p_Nq*p_Nq];
                const dlong id1 = GlobalToLocal[base + 1*p_Np + k*p_Nq*p_Nq];
                const dlong id2 = GlobalToLocal[base + 2*p_Np + k*p_Nq*p_Nq];
                r_Aq[0][k] = (id0!=-1) ? q[id0] : 0.0;
                r_Aq[1][k] = (id1!=-1) ? q[id1] : 0.0;
                r_Aq[2][k] = (id2!=-1) ? q[id2] : 0.0;
              }
            }
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int b=0;b<p_cubNq;++b;@inner(1)){
        for(int a=0;a<p_cubNq;++a;@inner(0)){
          if(a<p_Nq && b<p_Nq){
            @restrict const dfloat *s_tmp = s_I;

            // #pragma unroll p_unrCubNq
            for(int k=0;k<p_cubNq;++k){
              dfloat res0 = 0;
              dfloat res1 = 0;
              dfloat res2 = 0;
              // #pragma unroll p_Nq
              for(int c=0;c<p_Nq;++c){
                res0 += s_tmp[0]*r_Aq[0][c];
                res1 += s_tmp[0]*r_Aq[1][c];
                res2 += s_tmp[0]*r_Aq[2][c];
                ++s_tmp;
              }
              s_Iq[k][0][es][b][a] = res0;
              s_Iq[k][1][es][b][a] = res1;
              s_Iq[k][2][es][b][a] = res2;
            }
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_cubNq;++k;@inner(1)){
        for(int a=0;a<p_cubNq;++a;@inner(0)){

          if(a<p_Nq){

            for(int b=0;b<p_Nq;++b){
              r_Aq[0][b] = s_Iq[k][0][es][b][a];
              r_Aq[1][b] = s_Iq[k][1][es][b][a];
              r_Aq[2][b] = s_Iq[k][2][es][b][a];
            }

            @restrict const dfloat *s_tmp = s_I;

            // #pragma unroll p_unrCubNq
            for(int j=0;j<p_cubNq;++j){
              dfloat res0 = 0;
              dfloat res1 = 0;
              dfloat res2 = 0;
              // #pragma unroll p_Nq
              for(int b=0;b<p_Nq;++b){
                res0 += s_tmp[0]*r_Aq[0][b];
                res1 += s_tmp[0]*r_Aq[1][b];
                res2 += s_tmp[0]*r_Aq[2][b];
                ++s_tmp;
              }
              s_Iq[k][0][es][j][a] = res0;
              s_Iq[k][1][es][j][a] = res1;
              s_Iq[k][2][es][j][a] = res2;
            }
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_cubNq;++k;@inner(1)){
        for(int j=0;j<p_cubNq;++j;@inner(0)){
          for(int a=0;a<p_Nq;++a){
            r_Aq[0][a] = s_Iq[k][0][es][j][a];
            r_Aq[1][a] = s_Iq[k][1][es][j][a];
            r_Aq[2][a] = s_Iq[k][2][es][j][a];
          }

          @restrict const dfloat *s_tmp = s_I;

          // #pragma unroll p_unrCubNq
          for(int i=0;i<p_cubNq;++i){
            dfloat res0 = 0;
            dfloat res1 = 0;
            dfloat res2 = 0;
            // #pragma unroll p_Nq
            for(int a=0;a<p_Nq;++a){
              res0 += s_tmp[0]*r_Aq[0][a];
              res1 += s_tmp[0]*r_Aq[1][a];
              res2 += s_tmp[0]*r_Aq[2][a];
              ++s_tmp;
            }

            s_Iq[k][0][es][j][i] = res0;
            s_Iq[k][1][es][j][i] = res1;
            s_Iq[k][2][es][j][i] = res2;
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          if(r_e<Nelements){
            for(int k=0;k<p_cubNq;++k){
              const dlong gbase = element*p_cubNp + k*p_cubNq*p_cubNq + j*p_cubNq + i;
              const dfloat GWJ = cubwJ[gbase];
              r_Aq[0][k] = GWJ*s_Iq[k][0][es][j][i];
              r_Aq[1][k] = GWJ*s_Iq[k][1][es][j][i];
              r_Aq[2][k] = GWJ*s_Iq[k][2][es][j][i];
            }

            @restrict const dfloat *s_tmp = s_IT;

            // #pragma unroll p_unrNq
            for(int c=0;c<p_Nq;++c){
              dfloat res0 = 0;
              dfloat res1 = 0;
              dfloat res2 = 0;
              // #pragma unroll p_cubNq
              for(int k=0;k<p_cubNq;++k){
                res0 += s_tmp[0]*r_Aq[0][k];
                res1 += s_tmp[0]*r_Aq[1][k];
                res2 += s_tmp[0]*r_Aq[2][k];
                ++s_tmp;
              }
              s_Iq[c][0][es][j][i] = res0;
              s_Iq[c][1][es][j][i] = res1;
              s_Iq[c][2][es][j][i] = res2;
            }
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int c=0;c<p_cubNq;++c;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          if(c<p_Nq){
            for(int j=0;j<p_cubNq;++j){
              r_Aq[0][j] = s_Iq[c][0][es][j][i];
              r_Aq[1][j] = s_Iq[c][1][es][j][i];
              r_Aq[2][j] = s_Iq[c][2][es][j][i];
            }

            @restrict const dfloat *s_tmp = s_IT;

            // #pragma unroll p_unrNq
            for(int b=0;b<p_Nq;++b){
              dfloat res0 = 0;
              dfloat res1 = 0;
              dfloat res2 = 0;
              // #pragma unroll p_cubNq
              for(int j=0;j<p_cubNq;++j){
                res0 += s_tmp[0]*r_Aq[0][j];
                res1 += s_tmp[0]*r_Aq[1][j];
                res2 += s_tmp[0]*r_Aq[2][j];
                ++s_tmp;
              }
              s_Iq[c][0][es][b][i] = res0;
              s_Iq[c][1][es][b][i] = res1;
              s_Iq[c][2][es][b][i] = res2;
            }
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int c=0;c<p_cubNq;++c;@inner(1)){
        for(int b=0;b<p_cubNq;++b;@inner(0)){

          if(b<p_Nq && c<p_Nq){
            for(int i=0;i<p_cubNq;++i){
              r_Aq[0][i] = s_Iq[c][0][es][b][i];
              r_Aq[1][i] = s_Iq[c][1][es][b][i];
              r_Aq[2][i] = s_Iq[c][2][es][b][i];
            }

            @restrict const dfloat *s_tmp = s_IT;

            // #pragma unroll p_unrNq
            for(int a=0;a<p_Nq;++a){
              dfloat res0 = 0;
              dfloat res1 = 0;
              dfloat res2 = 0;
              // #pragma unroll p_cubNq
              for(int i=0;i<p_cubNq;++i){
                res0 += s_tmp[0]*r_Aq[0][i];
                res1 += s_tmp[0]*r_Aq[1][i];
                res2 += s_tmp[0]*r_Aq[2][i];
                ++s_tmp;
              }

              s_Iq[c][0][es][b][a] = res0;
              s_Iq[c][1][es][b][a] = res1;
              s_Iq[c][2][es][b][a] = res2;
            }
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){
          if(i<p_Nq && j<p_Nq){
            if(r_e<Nelements){
              for(int k = 0; k < p_Nq; k++){
                const dlong id = element*p_Np*p_Nfields +k*p_Nq*p_Nq+ j*p_Nq + i;
                const dfloat Aqid0 = s_Iq[k][0][es][j][i];
                const dfloat Aqid1 = s_Iq[k][1][es][j][i];
                const dfloat Aqid2 = s_Iq[k][2][es][j][i];
                Aq[id+0*p_Np] = Aqid0;
                Aq[id+1*p_Np] = Aqid1;
                Aq[id+2*p_Np] = Aqid2;
              }
            }
          }
        }
      }
    }
  }
}

#endif
