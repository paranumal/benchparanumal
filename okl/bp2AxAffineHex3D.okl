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
// no packing
#define p_NelementsPerBlk 1
#endif

//padding for bank conflicts
#if p_Nq==8 || p_Nq==4
#define p_pad 1
#else
#define p_pad 0
#endif


@kernel void bp2AxAffineHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  cubwJ,
                        @restrict const  dfloat *  I,
                        @restrict const  dfloat *  MM,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){


  for(dlong eo=0; eo<Nelements; eo+=p_NelementsPerBlk; @outer(0)){

    @shared dfloat s_MM[p_Nq][p_Nq+p_pad];
    @shared dfloat s_q[p_Nq][p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq+p_pad];

    @exclusive dlong r_e, element;

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          if(es==0)
            s_MM[j][i] = MM[i+p_Nq*j];

          r_e = eo + es;

          if(r_e<Nelements) {
            element = elementList[r_e];

            // load pencil of u into register
            const dlong base = i + j*p_Nq + element*p_Np*p_Nfields;

            // #pragma unroll p_Nq
            for(int k=0;k<p_Nq;++k) {
              for(int f=0;f<p_Nfields;++f) {
                const dlong id = GlobalToLocal[base + f*p_Np + k*p_Nq*p_Nq];
                s_q[k][f][es][j][i] = (id!=-1) ? q[id] : 0.0;
              }
            }
          }
        }
      }
    }

    // multiply by M in 'r'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_Nq;++k;@inner(1)){
        for(int j=0;j<p_Nq;++j;@inner(0)){

          dfloat r_q[p_Nfields][p_Nq];

          // #pragma unroll p_Nq
          for(int i=0;i<p_Nq;++i) {
            for(int f=0;f<p_Nfields;++f) {
              r_q[f][i] = s_q[k][f][es][j][i];
            }
          }

          // #pragma unroll p_Nq
          for(int i=0;i<p_Nq;++i){

            dfloat Mq[p_Nfields];

            for(int f=0;f<p_Nfields;++f) {
              Mq[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for(int m=0;m<p_Nq;++m){
              const dfloat Mi = s_MM[m][i];
              for(int f=0;f<p_Nfields;++f) {
                Mq[f] += Mi*r_q[f][m];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_q[k][f][es][j][i] = Mq[f]; // ok since only this thread
            }
          }
        }
      }
    }

    // multiply by M in 's'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_Nq;++k;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat r_q[p_Nfields][p_Nq];

          // #pragma unroll p_Nq
          for(int j=0;j<p_Nq;++j) {
            for(int f=0;f<p_Nfields;++f) {
              r_q[f][j] = s_q[k][f][es][j][i];
            }
          }

          // #pragma unroll p_Nq
          for(int j=0;j<p_Nq;++j){

            dfloat Mq[p_Nfields];

            for(int f=0;f<p_Nfields;++f) {
              Mq[f] = 0.0;
            }

            // #pragma unroll p_Nq
            for(int m=0;m<p_Nq;++m){
              const dfloat Mj = s_MM[m][j];
              for(int f=0;f<p_Nfields;++f) {
                Mq[f] += Mj*r_q[f][m];
              }
            }

            for(int f=0;f<p_Nfields;++f) {
              s_q[k][f][es][j][i] = Mq[f]; // ok since only this thread
            }
          }
        }
      }
    }

    // multiply by M in 't'
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          if (r_e<Nelements) {
            dfloat J = cubwJ[element];

            dfloat r_q[p_Nfields][p_Nq];

            // #pragma unroll p_Nq
            for(int k=0;k<p_Nq;++k) {
              for(int f=0;f<p_Nfields;++f) {
                r_q[f][k] = s_q[k][f][es][j][i];
              }
            }

            // #pragma unroll p_Nq
            for(int k=0;k<p_Nq;++k){

              dfloat Mq[p_Nfields];

              for(int f=0;f<p_Nfields;++f) {
                Mq[f] = 0.0;
              }

              // #pragma unroll p_Nq
              for(int m=0;m<p_Nq;++m){
                const dfloat Mk = s_MM[m][k];
                for(int f=0;f<p_Nfields;++f) {
                  Mq[f] += Mk*r_q[f][m];
                }
              }

              const dlong id = element*p_Np*p_Nfields + j*p_Nq + i;
              for(int f=0;f<p_Nfields;++f) {
                Aq[id+f*p_Np+k*p_Nq*p_Nq] = J*Mq[f];
              }
            }
          }
        }
      }
    }
  }
}



#else

#if p_N==1
#define p_NelementsPerBlk 7
#elif p_N==2
#define p_NelementsPerBlk 3
#elif p_N==3
#define p_NelementsPerBlk 1
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

@kernel void bp2AxAffineHex3D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  cubwJ,
                        @restrict const  dfloat *  I,
                        @restrict const  dfloat *  MM,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

  for(int eo=0;eo<Nelements;eo+=p_NelementsPerBlk;@outer(0)){

//padding for bank conflicts
#if p_Nq==8 || p_Nq==4
#define p_pad 1
#else
#define p_pad 0
#endif

    @shared dfloat s_MM [p_Nq][p_Nq+p_pad];
    @shared dfloat s_q[p_Nfields][p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];

    @exclusive dlong element;
    @exclusive int k, es;
    @exclusive dfloat r_q[p_Nfields], r_J;

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          //load operators
          if(ke==0){
            const int id = j*p_Nq+i;
            const dfloat Mji = MM[id];
            s_MM[j][i] = Mji;
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

    // multiply by M in 't'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          for(int f=0;f<p_Nfields;++f) {
            r_q[f]=0.0;
          }
          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            for(int f=0;f<p_Nfields;++f) {
              r_q[f] += s_MM[m][k]*s_q[f][es][m][j][i];
            }
          }
        }
      }
    }

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          for(int f=0;f<p_Nfields;++f) {
            s_q[f][es][k][j][i] = r_q[f];
          }
        }
      }
    }

    // multiply by M in 's'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          for(int f=0;f<p_Nfields;++f) {
            r_q[f]=0.0;
          }
          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            for(int f=0;f<p_Nfields;++f) {
              r_q[f] += s_MM[m][j]*s_q[f][es][k][m][i];
            }
          }
        }
      }
    }

    // multiply by M in 'r'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          if(eo+es<Nelements){
            r_J = cubwJ[element];
          }

          for(int f=0;f<p_Nfields;++f) {
            r_q[f]=0.0;
          }

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            for(int f=0;f<p_Nfields;++f) {
              r_q[f] += s_MM[m][i]*s_q[f][es][k][j][m];
            }
          }

          if(eo+es<Nelements){
            const dlong base = i + j*p_Nq + k*p_Nq*p_Nq + element*p_Np*p_Nfields;
            for(int f=0;f<p_Nfields;++f) {
              Aq[base+f*p_Np] = r_J*r_q[f];
            }
          }
        }
      }
    }
  }
}

#endif
