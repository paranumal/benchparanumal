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
#define p_NelementsPerBlk 14
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


@kernel void bp6AxQuad2D(const dlong Nelements,
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

  for(int eo=0;eo<Nelements;eo+=p_NelementsPerBlk*p_NelementsPerThread;@outer(0)){

    @shared dfloat   s_D[p_Nq][p_Nq];
    @shared dfloat   s_q[p_NelementsPerBlk][p_NelementsPerThread][p_Nfields][p_Nq][p_Nq];
    @shared dfloat s_Gqr[p_NelementsPerBlk][p_NelementsPerThread][p_Nfields][p_Nq][p_Nq];
    @shared dfloat s_Gqs[p_NelementsPerBlk][p_NelementsPerThread][p_Nfields][p_Nq][p_Nq];

    @exclusive dlong element[p_NelementsPerThread];

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          //load operators
          if(es==0){
            const int id = j*p_Nq+i;
            const dfloat Dji = D[id];
            s_D[j][i] = Dji;
          }

          for (int ek=0;ek<p_NelementsPerThread;++ek) {
            if(eo+es*p_NelementsPerThread+ek<Nelements) {
              element[ek] = elementList[eo+es*p_NelementsPerThread+ek];
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

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp[p_Nfields][p_NelementsPerThread];
          for(int s=0;s<p_NelementsPerThread;++s) {
            tmp[0][s] = 0;
            tmp[1][s] = 0;
          }

          // 'r' terms
          // #pragma unroll p_cubUnr
          for(int m = 0; m < p_Nq; ++m) {
            const dfloat Dim = s_D[i][m];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              tmp[0][s] += Dim*s_q[es][s][0][j][m];
              tmp[1][s] += Dim*s_q[es][s][1][j][m];
            }
          }

          for(int ek=0;ek<p_NelementsPerThread;++ek){
            if(eo+es*p_NelementsPerThread+ek<Nelements){
              const dlong gbase = element[ek]*p_Np + i + j*p_Nq;
              const dfloat G00 = ggeo[p_Nggeo*gbase+p_G00ID];
              const dfloat G01 = ggeo[p_Nggeo*gbase+p_G01ID];

              s_Gqr[es][ek][0][j][i] = G00*tmp[0][ek];
              s_Gqr[es][ek][1][j][i] = G00*tmp[1][ek];
              s_Gqs[es][ek][0][j][i] = G01*tmp[0][ek];
              s_Gqs[es][ek][1][j][i] = G01*tmp[1][ek];
            }
            tmp[0][ek] = 0;
            tmp[1][ek] = 0;
          }

          // 's' terms
          // #pragma unroll p_cubUnr
          for(int m = 0; m < p_Nq; ++m) {
            const dfloat Djm = s_D[j][m];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              tmp[0][s] += Djm*s_q[es][s][0][m][i];
              tmp[1][s] += Djm*s_q[es][s][1][m][i];
            }
          }

          // #pragma unroll p_NelementsPerThread
          for(int ek=0;ek<p_NelementsPerThread;++ek){
            if(eo+es*p_NelementsPerThread+ek<Nelements){
              const dlong gbase = element[ek]*p_Np + i + j*p_Nq;
              const dfloat G01 = ggeo[p_Nggeo*gbase+p_G01ID];
              const dfloat G11 = ggeo[p_Nggeo*gbase+p_G11ID];

              s_Gqr[es][ek][0][j][i] += G01*tmp[0][ek];
              s_Gqr[es][ek][1][j][i] += G01*tmp[1][ek];
              s_Gqs[es][ek][0][j][i] += G11*tmp[0][ek];
              s_Gqs[es][ek][1][j][i] += G11*tmp[1][ek];
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
            tmpAp[0][ek] = 0;
            tmpAp[1][ek] = 0;

            if(eo+es*p_NelementsPerThread+ek<Nelements){
              const dlong gbase = element[ek]*p_Np + i + j*p_Nq;
              const dfloat GWJ = wJ[gbase];

              tmpAp[0][ek] = s_q[es][ek][0][j][i]*lambda*GWJ;
              tmpAp[1][ek] = s_q[es][ek][1][j][i]*lambda*GWJ;
            }
          }

          // use same matrix for both slices
          // #pragma unroll p_cubUnr
          for(int m=0;m<p_Nq;++m){
            const dfloat Dmi = s_D[m][i];
            const dfloat Dmj = s_D[m][j];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              tmpAp[0][s] += Dmi*s_Gqr[es][s][0][j][m];
              tmpAp[1][s] += Dmi*s_Gqr[es][s][1][j][m];
              tmpAp[0][s] += Dmj*s_Gqs[es][s][0][m][i];
              tmpAp[1][s] += Dmj*s_Gqs[es][s][1][m][i];
            }
          }

          // #pragma unroll p_NelementsPerThread
          for(int ek=0;ek<p_NelementsPerThread;++ek){
            if(eo+es*p_NelementsPerThread+ek<Nelements){
              const dlong base = i + j*p_Nq + element[ek]*p_Np*p_Nfields;
              Aq[base+0*p_Np] = tmpAp[0][ek];
              Aq[base+1*p_Np] = tmpAp[1][ek];
            }
          }
        }
      }
    }
  }
}
