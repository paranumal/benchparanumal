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
#define p_NelementsPerThread 1
#elif p_N==2
#define p_NelementsPerBlk 28
#define p_NelementsPerThread 1
#elif p_N==3
#define p_NelementsPerBlk 20
#define p_NelementsPerThread 1
#elif p_N==4
#define p_NelementsPerBlk 2
#define p_NelementsPerThread 1
#elif p_N==5
#define p_NelementsPerBlk 10
#define p_NelementsPerThread 1
#elif p_N==6
#define p_NelementsPerBlk 8
#define p_NelementsPerThread 1
#else
// unoptimized
#define p_NelementsPerBlk 1
#define p_NelementsPerThread 1
#endif


@kernel void bp2AxQuad2D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  cubwJ,
                        @restrict const  dfloat *  I,
                        @restrict const  dfloat *  MM,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

  for(int eo=0;eo<Nelements;eo+=p_NelementsPerBlk*p_NelementsPerThread;@outer(0)){

    @shared dfloat s_I [p_cubNq][p_Nq];
    @shared dfloat  s_q[p_NelementsPerBlk][p_NelementsPerThread][p_Nfields][p_cubNq][p_cubNq];
    @shared dfloat s_Iq[p_NelementsPerBlk][p_NelementsPerThread][p_Nfields][p_cubNq][p_cubNq];

    @exclusive dlong element[p_NelementsPerThread];

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          //load operators
          if(es==0){
            if(i<p_Nq){
              const int id = j*p_Nq+i;
              const dfloat Iji = I[id];
              s_I[j][i] = Iji;
            }
          }

          for (int ek=0;ek<p_NelementsPerThread;++ek) {
            if(eo+es*p_NelementsPerThread+ek<Nelements) {
              element[ek] = elementList[eo+es*p_NelementsPerThread+ek];
              if (i<p_Nq && j<p_Nq) {
                const dlong base = i + j*p_Nq + element[ek]*p_Np*p_Nfields;
                const dlong id0 = GlobalToLocal[base + 0*p_Np];
                const dlong id1 = GlobalToLocal[base + 1*p_Np];
                s_q[es][ek][0][j][i] = (id0!=-1) ? q[id0] : 0.0;
                s_q[es][ek][1][j][i] = (id1!=-1) ? q[id1] : 0.0;
              }
            }
          }
        }
      }
    }

    // interpolate in 'r'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){
          if(j<p_Nq){

            dfloat tmp[p_Nfields][p_NelementsPerThread];
            for(int s=0;s<p_NelementsPerThread;++s) {
              tmp[0][s] = 0;
              tmp[1][s] = 0;
            }

            // #pragma unroll p_Nq
            for(int m=0;m<p_Nq;++m){
              dfloat Iim = s_I[i][m];

              // #pragma unroll p_NelementsPerThread
              for(int s=0;s<p_NelementsPerThread;++s){
                tmp[0][s] += Iim*s_q[es][s][0][j][m];
                tmp[1][s] += Iim*s_q[es][s][1][j][m];
              }
            }

            for(int s=0;s<p_NelementsPerThread;++s){
              s_Iq[es][s][0][j][i] = tmp[0][s];
              s_Iq[es][s][1][j][i] = tmp[1][s];
            }
          }
        }
      }
    }

    // interpolate in 's'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          dfloat tmp[p_Nfields][p_NelementsPerThread];
          for(int s=0;s<p_NelementsPerThread;++s) {
            tmp[0][s] = 0;
            tmp[1][s] = 0;
          }

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            dfloat Iim = s_I[j][m];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              tmp[0][s] += Iim*s_Iq[es][s][0][m][i];
              tmp[1][s] += Iim*s_Iq[es][s][1][m][i];
            }
          }

          for(int s=0;s<p_NelementsPerThread;++s){
            s_q[es][s][0][j][i] = tmp[0][s];
            s_q[es][s][1][j][i] = tmp[1][s];
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){

          // #pragma unroll p_NelementsPerThread
          for(int ek=0;ek<p_NelementsPerThread;++ek){
            if(eo+es*p_NelementsPerThread+ek<Nelements){
              const dlong gbase = element[ek]*p_cubNp + i + j*p_cubNq;
              const dfloat GWJ = cubwJ[gbase];

              s_q[es][ek][0][j][i] *= GWJ;
              s_q[es][ek][1][j][i] *= GWJ;
            }
          }
        }
      }
    }

    // test in 's'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){
          if(j<p_Nq){

            dfloat tmp[p_Nfields][p_NelementsPerThread];
            for(int s=0;s<p_NelementsPerThread;++s) {
              tmp[0][s] = 0;
              tmp[1][s] = 0;
            }

            // #pragma unroll p_cubUnr
            for(int m=0;m<p_cubNq;++m){
              const dfloat Imj = s_I[m][j];
              // #pragma unroll p_NelementsPerThread
              for(int s=0;s<p_NelementsPerThread;++s){
                tmp[0][s] += Imj*s_q[es][s][0][m][i];
                tmp[1][s] += Imj*s_q[es][s][1][m][i];
              }
            }

            for(int s=0;s<p_NelementsPerThread;++s){
              s_Iq[es][s][0][j][i] = tmp[0][s];
              s_Iq[es][s][1][j][i] = tmp[1][s];
            }
          }
        }
      }
    }

    // test in 'r'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){
          if(i<p_Nq && j<p_Nq){

            dfloat tmp[p_Nfields][p_NelementsPerThread];
            for(int s=0;s<p_NelementsPerThread;++s) {
              tmp[0][s] = 0;
              tmp[1][s] = 0;
            }

            // #pragma unroll p_cubUnr
            for(int m=0;m<p_cubNq;++m){
              const dfloat Imj = s_I[m][i];
              // #pragma unroll p_NelementsPerThread
              for(int s=0;s<p_NelementsPerThread;++s){
                tmp[0][s] += Imj*s_Iq[es][s][0][j][m];
                tmp[1][s] += Imj*s_Iq[es][s][1][j][m];
              }
            }

            // #pragma unroll p_NelementsPerThread
            for(int ek=0;ek<p_NelementsPerThread;++ek){
              if(eo+es*p_NelementsPerThread+ek<Nelements){
                const dlong base = i + j*p_Nq + element[ek]*p_Np*p_Nfields;
                Aq[base+0*p_Np] = tmp[0][ek];
                Aq[base+1*p_Np] = tmp[1][ek];
              }
            }
          }
        }
      }
    }
  }
}
