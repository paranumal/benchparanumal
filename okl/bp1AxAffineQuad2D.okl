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


@kernel void bp1AxAffineQuad2D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  cubwJ,
                        @restrict const  dfloat *  I,
                        @restrict const  dfloat *  MM,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

  for(int eo=0;eo<Nelements;eo+=p_NelementsPerBlk*p_NelementsPerThread;@outer(0)){

    @shared dfloat s_MM[p_Nq][p_Nq];
    @shared dfloat  s_q[p_NelementsPerBlk][p_NelementsPerThread][p_Nq][p_Nq];

    @exclusive dlong element[p_NelementsPerThread];

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          //load operators
          if(es==0){
            const int id = j*p_Nq+i;
            const dfloat MMji = MM[id];
            s_MM[j][i] = MMji;
          }

          for (int ek=0;ek<p_NelementsPerThread;++ek) {
            if(eo+es*p_NelementsPerThread+ek<Nelements) {
              element[ek] = elementList[eo+es*p_NelementsPerThread+ek];
              const dlong id = GlobalToLocal[i + j*p_Nq + element[ek]*p_Np];
              s_q[es][ek][j][i] = (id!=-1) ? q[id] : 0.0;
            }
          }
        }
      }
    }

    // multiply by M in 'r'
    @exclusive dfloat Mq[p_NelementsPerThread];

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          for(int s=0;s<p_NelementsPerThread;++s)
            Mq[s] = 0;

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            dfloat Mim = s_MM[m][i];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              Mq[s] += Mim*s_q[es][s][j][m];
            }
          }
        }
      }
    }

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          for(int s=0;s<p_NelementsPerThread;++s){
            s_q[es][s][j][i] = Mq[s];
          }
        }
      }
    }

    // multiply by M in 's'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          // Load J
          dfloat J[p_NelementsPerThread];
          for (int ek=0;ek<p_NelementsPerThread;++ek) {
            if(eo+es*p_NelementsPerThread+ek<Nelements) {
              J[ek] = cubwJ[element[ek]];
            }
          }

          for(int s=0;s<p_NelementsPerThread;++s)
            Mq[s] = 0;

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            dfloat Mim = s_MM[m][j];

            // #pragma unroll p_NelementsPerThread
            for(int s=0;s<p_NelementsPerThread;++s){
              Mq[s] += Mim*s_q[es][s][m][i];
            }
          }

          // #pragma unroll p_NelementsPerThread
          for(int ek=0;ek<p_NelementsPerThread;++ek){
            if(eo+es*p_NelementsPerThread+ek<Nelements){
              const dlong base = i + j*p_Nq + element[ek]*p_Np;
              Aq[base] = J[ek]*Mq[ek];
            }
          }
        }
      }
    }
  }
}
