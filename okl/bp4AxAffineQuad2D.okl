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

#if p_N==1
#define p_NelementsPerBlk 38
#define p_NelementsPerThread 2
#elif p_N==2
#define p_NelementsPerBlk 28
#define p_NelementsPerThread 2
#elif p_N==3
#define p_NelementsPerBlk 20
#define p_NelementsPerThread 2
#elif p_N==4
#define p_NelementsPerBlk 14
#define p_NelementsPerThread 2
#elif p_N==5
#define p_NelementsPerBlk 10
#define p_NelementsPerThread 2
#elif p_N==6
#define p_NelementsPerBlk 8
#define p_NelementsPerThread 2
#else
// unoptimized
#define p_NelementsPerBlk 1
#define p_NelementsPerThread 1
#endif


@kernel void bp4AxAffineQuad2D(const dlong Nelements,
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

  for(int eo=0;eo<Nelements;eo+=p_NelementsPerBlk*p_NelementsPerThread;@outer(0)){

    @shared dfloat s_D [p_Nq][p_Nq];
    @shared dfloat s_invV[p_Nq][p_Nq];
    @shared dfloat   s_q[p_NelementsPerBlk][p_NelementsPerThread][p_Nfields][p_Nq][p_Nq];
    @shared dfloat s_Gqr[p_NelementsPerBlk][p_NelementsPerThread][p_Nfields][p_Nq][p_Nq];
    @shared dfloat s_Gqs[p_NelementsPerBlk][p_NelementsPerThread][p_Nfields][p_Nq][p_Nq];

    @exclusive dlong element[p_NelementsPerThread];

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          //load operators
          if(es==0){
            s_D[j][i] = D[j*p_Nq+i];
            s_invV[j][i] = invV[j*p_Nq+i];
          }

          for (int ek=0;ek<p_NelementsPerThread;++ek) {
            if(eo+es*p_NelementsPerThread+ek<Nelements) {
              element[ek] = elementList[eo+es*p_NelementsPerThread+ek];

              const dlong base = i + j*p_Nq + element[ek]*p_Np*p_Nfields;
              for(int f=0;f<p_Nfields;++f) {
                const dlong id = GlobalToLocal[base + f*p_Np];
                s_q[es][ek][f][j][i] = (id!=-1) ? q[id] : 0.0;
              }
            }
          }
        }
      }
    }

    // interpolate in 'r'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields][p_NelementsPerThread];
          for(int s=0;s<p_NelementsPerThread;++s) {
            for(int f=0;f<p_Nfields;++f) {
              tmp[f][s] = 0;
            }
          }

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            dfloat Vim = s_invV[i][m];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              for(int f=0;f<p_Nfields;++f) {
                tmp[f][s] += Vim*s_q[es][s][f][j][m];
              }
            }
          }

          for(int s=0;s<p_NelementsPerThread;++s){
            for(int f=0;f<p_Nfields;++f) {
              s_Gqr[es][s][f][j][i] = tmp[f][s];
            }
          }
        }
      }
    }

    // interpolate in 's'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields][p_NelementsPerThread];
          for(int s=0;s<p_NelementsPerThread;++s) {
            for(int f=0;f<p_Nfields;++f) {
              tmp[f][s] = 0;
            }
          }

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            dfloat Vim = s_invV[j][m];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              for(int f=0;f<p_Nfields;++f) {
                tmp[f][s] += Vim*s_Gqr[es][s][f][m][i];
              }
            }
          }

          for(int s=0;s<p_NelementsPerThread;++s){
            for(int f=0;f<p_Nfields;++f) {
              s_q[es][s][f][j][i] = tmp[f][s];
            }
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields][p_NelementsPerThread];
          for(int s=0;s<p_NelementsPerThread;++s) {
            for(int f=0;f<p_Nfields;++f) {
              tmp[f][s] = 0;
            }
          }

          // 'r' terms
          // #pragma unroll p_cubUnr
          for(int m = 0; m < p_Nq; ++m) {
            const dfloat Dim = s_D[i][m];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              for(int f=0;f<p_Nfields;++f) {
                tmp[f][s] += Dim*s_q[es][s][f][j][m];
              }
            }
          }

          for(int ek=0;ek<p_NelementsPerThread;++ek){
            if(eo+es*p_NelementsPerThread+ek<Nelements){
              const dfloat G00 = ggeo[p_Nggeo*element[ek]+p_G00ID];
              const dfloat G01 = ggeo[p_Nggeo*element[ek]+p_G01ID];

              for(int f=0;f<p_Nfields;++f) {
                s_Gqr[es][ek][f][j][i] = G00*tmp[f][ek];
                s_Gqs[es][ek][f][j][i] = G01*tmp[f][ek];
              }
            }
            for(int f=0;f<p_Nfields;++f) {
              tmp[f][ek] = 0;
            }
          }

          // 's' terms
          // #pragma unroll p_cubUnr
          for(int m = 0; m < p_Nq; ++m) {
            const dfloat Djm = s_D[j][m];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              for(int f=0;f<p_Nfields;++f) {
                tmp[f][s] += Djm*s_q[es][s][f][m][i];
              }
            }
          }

          // #pragma unroll p_NelementsPerThread
          for(int ek=0;ek<p_NelementsPerThread;++ek){
            if(eo+es*p_NelementsPerThread+ek<Nelements){
              const dfloat G01 = ggeo[p_Nggeo*element[ek]+p_G01ID];
              const dfloat G11 = ggeo[p_Nggeo*element[ek]+p_G11ID];

              for(int f=0;f<p_Nfields;++f) {
                s_Gqr[es][ek][f][j][i] += G01*tmp[f][ek];
                s_Gqs[es][ek][f][j][i] += G11*tmp[f][ek];
              }
            }
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmpAp[p_Nfields][p_NelementsPerThread];
          // #pragma unroll p_NelementsPerThread
          for(int ek=0;ek<p_NelementsPerThread;++ek){
            for(int f=0;f<p_Nfields;++f) {
              tmpAp[f][ek] = 0;
            }

            if(eo+es*p_NelementsPerThread+ek<Nelements){
              const dfloat GWJ = wJ[element[ek]];

              for(int f=0;f<p_Nfields;++f) {
                tmpAp[f][ek] = s_q[es][ek][f][j][i]*lambda*GWJ;
              }
            }
          }

          // use same matrix for both slices
          // #pragma unroll p_cubUnr
          for(int m=0;m<p_Nq;++m){
            const dfloat Dmi = s_D[m][i];
            const dfloat Dmj = s_D[m][j];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              for(int f=0;f<p_Nfields;++f) {
                tmpAp[f][s] += Dmi*s_Gqr[es][s][f][j][m];
                tmpAp[f][s] += Dmj*s_Gqs[es][s][f][m][i];
              }
            }
          }

          for(int s=0;s<p_NelementsPerThread;++s){
            for(int f=0;f<p_Nfields;++f) {
              s_q[es][s][f][j][i] = tmpAp[f][s];
            }
          }
        }
      }
    }

    // test in 's'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields][p_NelementsPerThread];
          for(int s=0;s<p_NelementsPerThread;++s) {
            for(int f=0;f<p_Nfields;++f) {
              tmp[f][s] = 0;
            }
          }

          // #pragma unroll p_cubUnr
          for(int m=0;m<p_Nq;++m){
            const dfloat Vmj = s_invV[m][j];
            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              for(int f=0;f<p_Nfields;++f) {
                tmp[f][s] += Vmj*s_q[es][s][f][m][i];
              }
            }
          }

          for(int s=0;s<p_NelementsPerThread;++s){
            for(int f=0;f<p_Nfields;++f) {
              s_Gqr[es][s][f][j][i] = tmp[f][s];
            }
          }
        }
      }
    }

    // test in 'r'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields][p_NelementsPerThread];
          for(int s=0;s<p_NelementsPerThread;++s) {
            for(int f=0;f<p_Nfields;++f) {
              tmp[f][s] = 0;
            }
          }

          // #pragma unroll p_cubUnr
          for(int m=0;m<p_Nq;++m){
            const dfloat Vmj = s_invV[m][i];
            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              for(int f=0;f<p_Nfields;++f) {
                tmp[f][s] += Vmj*s_Gqr[es][s][f][j][m];
              }
            }
          }

          // #pragma unroll p_NelementsPerThread
          for(int ek=0;ek<p_NelementsPerThread;++ek){
            if(eo+es*p_NelementsPerThread+ek<Nelements){
              const dlong base = i + j*p_Nq + element[ek]*p_Np*p_Nfields;
              for(int f=0;f<p_Nfields;++f) {
                Aq[base+f*p_Np] = tmp[f][ek];
              }
            }
          }
        }
      }
    }
  }
}
