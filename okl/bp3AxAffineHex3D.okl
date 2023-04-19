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
#if OCCA_USE_CUDA==1 && p_N>1

/*Unblocked 2D threadblock kernel*/
@kernel void bp3AxAffineHex3D(const dlong Nelements,
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

  for(int e=0;e<Nelements;e++;@outer(0)){

//padding for bank conflicts
#if p_Nq==8 || p_Nq==4
#define p_pad 1
#else
#define p_pad 0
#endif

    @shared dfloat s_q[p_Nq][p_Nq][p_Nq+p_pad];
    @shared dfloat s_D[p_Nq][p_Nq+p_pad];
    @shared dfloat s_qr[p_Nq][p_Nq+p_pad];
    @shared dfloat s_qs[p_Nq][p_Nq+p_pad];
    @shared dfloat s_invV[p_Nq][p_Nq];

    @exclusive dfloat r_q[p_Nq];
    @exclusive dfloat r_qt;

    @exclusive dlong element;

    for(int b=0;b<p_Nq;++b;@inner(1)){
      for(int a=0;a<p_Nq;++a;@inner(0)){

        s_invV[b][a] = invV[a+p_Nq*b];
        s_D[b][a] = D[b*p_Nq+a];

        element = elementList[e];

        // load q
        const dlong base = a + b*p_Nq + element*p_Np;

        //#pragma unroll p_Nq
        for(int c=0;c<p_Nq;++c) {
          const dlong id = GlobalToLocal[base + c*p_Nq*p_Nq];
          s_q[c][b][a] = (id!=-1) ? q[id] : 0.0;
        }
      }
    }


    //============== interpolate in 3 dir ========================
    // b --> a --> c
    for(int c=0;c<p_Nq;++c;@inner(1)){
      for(int a=0;a<p_Nq;++a;@inner(0)){

        // #pragma unroll p_Nq
        for(int b=0;b<p_Nq;++b)
          r_q[b] = s_q[c][b][a];

        // #pragma unroll p_Nq
        for(int j=0;j<p_Nq;++j){

          dfloat tmp = 0;

          // #pragma unroll p_Nq
          for(int b=0;b<p_Nq;++b){
            const dfloat sVjb= s_invV[j][b];
            tmp  += sVjb*r_q[b];
          }

          s_q[c][j][a] = tmp;
        }
      }
    }//for c

    //transform in c
    for(int c=0;c<p_Nq;++c;@inner(1)){
      for(int j=0;j<p_Nq;++j;@inner(0)){

        // #pragma unroll p_Nq
        for(int a=0;a<p_Nq;++a)
          r_q[a] = s_q[c][j][a];

        // #pragma unroll p_Nq
        for(int i=0;i<p_Nq;++i){

          dfloat tmp = 0;

          //#pragma unroll p_Nq
          for(int a=0;a<p_Nq;++a){

            const dfloat sVia = s_invV[i][a];
            tmp  += sVia*r_q[a];
          }

          s_q[c][j][i] = tmp;
        }
      }
    }

    // transform in c
    for(int j=0;j<p_Nq;++j;@inner(1)){
      for(int i=0;i<p_Nq;++i;@inner(0)){

        // #pragma unroll p_Nq
        for(int c=0;c<p_Nq;++c)
          r_q[c] = s_q[c][j][i];

        // #pragma unroll p_Nq
        for(int k=0;k<p_Nq;++k){

          dfloat tmp = 0;

          //#pragma unroll p_Nq
          for(int c=0;c<p_Nq;++c){

            const dfloat sVkc = s_invV[k][c];
            tmp  += sVkc*r_q[c];
          }

          s_q[k][j][i] = tmp; // ok since only this thread
        }

        // #pragma unroll p_Nq
        for(int k=0; k<p_Nq; ++k)
          r_q[k]=0.0f;
      }
    }

    //===============now differentiate once interpolated
    #pragma unroll p_Nq
    for(int k=0; k<p_Nq; ++k) {

      for(int j=0; j<p_Nq; ++j; @inner(1)) {
        for(int i=0; i<p_Nq; ++i; @inner(0)) {
          //geofactors for k j i thread
          const dfloat r_GwJ = wJ[element];

          const dfloat r_G00 = ggeo[p_Nggeo*element+p_G00ID];
          const dfloat r_G01 = ggeo[p_Nggeo*element+p_G01ID];
          const dfloat r_G02 = ggeo[p_Nggeo*element+p_G02ID];

          const dfloat r_G11 = ggeo[p_Nggeo*element+p_G11ID];
          const dfloat r_G12 = ggeo[p_Nggeo*element+p_G12ID];
          const dfloat r_G22 = ggeo[p_Nggeo*element+p_G22ID];

          // now, put together dq/dr, qq/ds, dq/dt and dq/dI
          dfloat dr = 0.0f;
          dfloat ds = 0.0f;
          dfloat dt = 0.0f;

          //#pragma unroll p_Nq
          for (int n = 0; n<p_Nq; ++n) {
            dr += s_D[i][n]*s_q[k][j][n];
            ds += s_D[j][n]*s_q[k][n][i];
            dt += s_D[k][n]*s_q[n][j][i];
          }

          s_qr[j][i] = r_G00*dr + r_G01*ds + r_G02*dt;
          s_qs[j][i] = r_G01*dr + r_G11*ds + r_G12*dt;
          r_qt = r_G02*dr + r_G12*ds + r_G22*dt;

          r_q[k] += lambda*r_GwJ*s_q[k][j][i];
        }
      }

      // weak diff
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          dfloat r_qk=0.0;
          //#pragma unroll p_Nq
          for(int n=0;n<p_Nq;++n){
            r_qk += s_D[n][i]*s_qr[j][n];
            r_qk += s_D[n][j]*s_qs[n][i];
            r_q[n] += s_D[k][n]*r_qt;
          }
          r_q[k] += r_qk;
        }
      }
    }//k

    //Loop 7
    for(int j=0;j<p_Nq;++j;@inner(1)){
      for(int i=0;i<p_Nq;++i;@inner(0)){

        // #pragma unroll p_Nq
        for(int k=0; k<p_Nq; ++k) {
          s_q[k][j][i] = r_q[k];
        }
      }
    }

    //=========== now project =================================================
    // b -> c -> a

    // transform back in b
    for(int k=0;k<p_Nq;++k;@inner(1)){
      for(int i=0;i<p_Nq;++i;@inner(0)){

        // #pragma unroll p_Nq
        for(int j=0;j<p_Nq;++j){
          r_q[j] = s_q[k][j][i];
        }

        // #pragma unroll p_Nq
        for(int b=0;b<p_Nq;++b){

          dfloat tmp = 0.0f;

          //#pragma unroll p_Nq
          for(int j=0;j<p_Nq;++j){

            const dfloat sVjb = s_invV[j][b];
            tmp  += sVjb*r_q[j];
          }

          s_q[k][b][i] = tmp;
        }
      }
    }

    // transform back in a
    for(int k=0;k<p_Nq;++k;@inner(1)){
      for(int b=0;b<p_Nq;++b;@inner(0)){
        // #pragma unroll p_Nq
        for(int i=0;i<p_Nq;++i)
          r_q[i] = s_q[k][b][i];

        // #pragma unroll p_Nq
        for(int a=0;a<p_Nq;++a){

          dfloat tmp  = 0.0f;

          //#pragma unroll p_Nq
          for(int i=0;i<p_Nq;++i){
            const dfloat sVia = s_invV[i][a];
            tmp  += sVia*r_q[i];
          }

          s_q[k][b][a] = tmp;
        }
      }
    }

    // transform back in c
    for(int b=0;b<p_Nq;++b;@inner(1)){
      for(int a=0;a<p_Nq;++a;@inner(0)){

        // #pragma unroll p_Nq
        for(int k=0;k<p_Nq;++k){
          r_q[k] = s_q[k][b][a];
        }

        // #pragma unroll p_Nq
        for(int c=0;c<p_Nq;++c){

          dfloat tmp  = 0.0f;

          //#pragma unroll p_Nq
          for(int k=0;k<p_Nq;++k){

            const dfloat sVkc = s_invV[k][c];
            tmp  += sVkc*r_q[k];
          }

          Aq[element*p_Np + c*p_Nq*p_Nq + b*p_Nq + a] = tmp;
        }//c
      }//a
    }//b
  }//e
}//kernel

#else //either OCCA_USE_CUDA!=1 or p_N==1

#if p_N>6

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

@kernel void bp3AxAffineHex3D(const dlong Nelements,
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

    @shared dfloat s_q[p_Nq][p_NelementsPerBlk][p_Nq][p_Nq+p_pad];
    @shared dfloat s_D[p_Nq][p_Nq+p_pad];
    @shared dfloat s_qr[p_NelementsPerBlk][p_Nq][p_Nq+p_pad];
    @shared dfloat s_qs[p_NelementsPerBlk][p_Nq][p_Nq+p_pad];
    @shared dfloat s_invV[p_Nq][p_Nq];

    @exclusive dfloat r_q[p_Nq];
    @exclusive dfloat r_qt;

    @exclusive dlong r_e, element;

    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int b=0;b<p_Nq;++b;@inner(1)){
        for(int a=0;a<p_Nq;++a;@inner(0)){

          if(es==0){
            s_invV[b][a] = invV[a+p_Nq*b];
          }

          if (es==0)
            s_D[b][a] = D[b*p_Nq+a];

          r_e = eo + es;

          if(r_e<Nelements) {
            element = elementList[r_e];

            // load q
            const dlong base = a + b*p_Nq + element*p_Np;

            //#pragma unroll p_Nq
            for(int c=0;c<p_Nq;++c) {
              const dlong id = GlobalToLocal[base + c*p_Nq*p_Nq];
              s_q[c][es][b][a] = (id!=-1) ? q[id] : 0.0;
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
          for(int b=0;b<p_Nq;++b)
            r_q[b] = s_q[c][es][b][a];

          // #pragma unroll p_Nq
          for(int j=0;j<p_Nq;++j){

            dfloat tmp = 0;

            // #pragma unroll p_Nq
            for(int b=0;b<p_Nq;++b){

              const dfloat sVjb= s_invV[j][b];
              tmp  += sVjb*r_q[b];
            }

            s_q[c][es][j][a] = tmp;
          }
        }
      }//for c
    }

    //transform in c
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int c=0;c<p_Nq;++c;@inner(1)){
        for(int j=0;j<p_Nq;++j;@inner(0)){

          // #pragma unroll p_Nq
          for(int a=0;a<p_Nq;++a)
            r_q[a] = s_q[c][es][j][a];

          // #pragma unroll p_Nq
          for(int i=0;i<p_Nq;++i){

            dfloat tmp = 0;

            //#pragma unroll p_Nq
            for(int a=0;a<p_Nq;++a){

              const dfloat sVia = s_invV[i][a];
              tmp  += sVia*r_q[a];
            }

            s_q[c][es][j][i] = tmp;
          }
        }
      }
    }

    // transform in c
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          // #pragma unroll p_Nq
          for(int c=0;c<p_Nq;++c)
            r_q[c] = s_q[c][es][j][i];

          // #pragma unroll p_Nq
          for(int k=0;k<p_Nq;++k){

            dfloat tmp = 0;

            //#pragma unroll p_Nq
            for(int c=0;c<p_Nq;++c){

              const dfloat sVkc = s_invV[k][c];
              tmp  += sVkc*r_q[c];
            }

            s_q[k][es][j][i] = tmp; // ok since only this thread
          }

          // #pragma unroll p_Nq
          for(int k=0; k<p_Nq; ++k)
            r_q[k]=0.0f;
        }
      }
    }

    //===============now differentiate once interpolated
#if OCCA_USE_CUDA==1
    #pragma unroll p_Nq
#endif
    for(int k=0; k<p_Nq; ++k) {

      for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
        for(int j=0; j<p_Nq; ++j; @inner(1)) {
          for(int i=0; i<p_Nq; ++i; @inner(0)) {

            dfloat r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;
            if (r_e<Nelements) {
              //geofactors for k j i thread
              r_GwJ = wJ[element];

              r_G00 = ggeo[p_Nggeo*element+p_G00ID];
              r_G01 = ggeo[p_Nggeo*element+p_G01ID];
              r_G02 = ggeo[p_Nggeo*element+p_G02ID];

              r_G11 = ggeo[p_Nggeo*element+p_G11ID];
              r_G12 = ggeo[p_Nggeo*element+p_G12ID];
              r_G22 = ggeo[p_Nggeo*element+p_G22ID];
            }

            // now, put together dq/dr, qq/ds, dq/dt and dq/dI
            dfloat dr = 0.0f;
            dfloat ds = 0.0f;
            dfloat dt = 0.0f;

            //#pragma unroll p_Nq
            for (int n = 0; n<p_Nq; ++n) {
              dr += s_D[i][n]*s_q[k][es][j][n];
              ds += s_D[j][n]*s_q[k][es][n][i];
              dt += s_D[k][n]*s_q[n][es][j][i];
            }

            s_qr[es][j][i] = r_G00*dr + r_G01*ds + r_G02*dt;
            s_qs[es][j][i] = r_G01*dr + r_G11*ds + r_G12*dt;
            r_qt = r_G02*dr + r_G12*ds + r_G22*dt;

            r_q[k] += lambda*r_GwJ*s_q[k][es][j][i];
          }
        }
      }

      // weak diff
      for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
        for(int j=0;j<p_Nq;++j;@inner(1)){
          for(int i=0;i<p_Nq;++i;@inner(0)){
            dfloat r_qk=0.0;
            //#pragma unroll p_Nq
            for(int n=0;n<p_Nq;++n){
              r_qk += s_D[n][i]*s_qr[es][j][n];
              r_qk += s_D[n][j]*s_qs[es][n][i];
              r_q[n] += s_D[k][n]*r_qt;
            }
            r_q[k] += r_qk;
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
            s_q[k][es][j][i] = r_q[k];
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
            r_q[j] = s_q[k][es][j][i];
          }

          // #pragma unroll p_Nq
          for(int b=0;b<p_Nq;++b){

            dfloat tmp = 0.0f;

            //#pragma unroll p_Nq
            for(int j=0;j<p_Nq;++j){

              const dfloat sVjb = s_invV[j][b];
              tmp  += sVjb*r_q[j];
            }

            s_q[k][es][b][i] = tmp;
          }
        }
      }
    }

    // transform back in a
    for(int es=0;es<p_NelementsPerBlk;++es;@inner(2)){
      for(int k=0;k<p_Nq;++k;@inner(1)){
        for(int b=0;b<p_Nq;++b;@inner(0)){
          // #pragma unroll p_Nq
          for(int i=0;i<p_Nq;++i)
            r_q[i] = s_q[k][es][b][i];

          // #pragma unroll p_Nq
          for(int a=0;a<p_Nq;++a){

            dfloat tmp  = 0.0f;

            //#pragma unroll p_Nq
            for(int i=0;i<p_Nq;++i){

              const dfloat sVia = s_invV[i][a];
              tmp  += sVia*r_q[i];
            }

            s_q[k][es][b][a] = tmp;
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
              r_q[k] = s_q[k][es][b][a];
            }

            // #pragma unroll p_Nq
            for(int c=0;c<p_Nq;++c){

              dfloat tmp  = 0.0f;

              //#pragma unroll p_Nq
              for(int k=0;k<p_Nq;++k){

                const dfloat sVkc = s_invV[k][c];
                tmp  += sVkc*r_q[k];
              }

              Aq[element*p_Np + c*p_Nq*p_Nq + b*p_Nq + a] = tmp;
            }//c
          }//if
        }//a
      }//b
    }//es
  }//eo
}//kernel


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

@kernel void bp3AxAffineHex3D(const dlong Nelements,
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
    @shared dfloat   s_q[p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];
    @shared dfloat s_Gqr[p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];
    @shared dfloat s_Gqs[p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];
    @shared dfloat s_Gqt[p_NelementsPerBlk][p_Nq][p_Nq][p_Nq+p_pad];

    @exclusive dlong element;
    @exclusive int k, es;

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          //load operators
          if(ke==0){
            const int id = j*p_Nq+i;
            const dfloat Dji = D[id];
            s_D[j][i] = Dji;

            s_invV[j][i] = invV[j*p_Nq+i];
          }

          k  = ke%p_Nq;
          es = ke/p_Nq;

          if(es+eo<Nelements) {
            element = elementList[es+eo];
            const dlong id = GlobalToLocal[i + j*p_Nq + k*p_Nq*p_Nq + element*p_Np];
            s_q[es][k][j][i] = (id!=-1) ? q[id] : 0.0;
          }
        }
      }
    }

    // interpolate in 't'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp=0.0;

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            const dfloat tmpp = s_q[es][m][j][i];
            tmp += s_invV[k][m]*tmpp;
          }

          s_Gqr[es][k][j][i] = tmp;
        }
      }
    }

    // interpolate in 'r'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          dfloat tmp =0.0;

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            const dfloat Vim = s_invV[i][m];
            tmp += Vim*s_Gqr[es][k][j][m];
          }

          s_Gqs[es][k][j][i] = tmp;
        }
      }
    }

    // interpolate in 's'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          dfloat tmp =0.0;

          // #pragma unroll p_Nq
          for(int m=0;m<p_Nq;++m){
            const dfloat Vim = s_invV[j][m];
            tmp += Vim*s_Gqs[es][k][m][i];
          }

          s_q[es][k][j][i] = tmp;
        }
      }
    }

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          if(eo+es<Nelements){
            dfloat tmp=0.0;

            // 't' terms
            // #pragma unroll p_cubUnr
            for(int m = 0; m < p_Nq; ++m) {
              const dfloat pmji = s_q[es][m][j][i];
              const dfloat Dkm = s_D[k][m];
              tmp += Dkm*pmji;
            }

            const dfloat G02 = ggeo[p_Nggeo*element+p_G02ID];
            const dfloat G12 = ggeo[p_Nggeo*element+p_G12ID];
            const dfloat G22 = ggeo[p_Nggeo*element+p_G22ID];

            s_Gqr[es][k][j][i] = G02*tmp;
            s_Gqs[es][k][j][i] = G12*tmp;
            s_Gqt[es][k][j][i] = G22*tmp;

            tmp = 0;

            // 'r' terms
            // #pragma unroll p_cubUnr
            for(int m = 0; m < p_Nq; ++m) {
              const dfloat Dim = s_D[i][m];
              tmp += Dim*s_q[es][k][j][m];
            }

            // 'r' terms
            const dfloat G00 = ggeo[p_Nggeo*element+p_G00ID];
            const dfloat G01 = ggeo[p_Nggeo*element+p_G01ID];

            s_Gqr[es][k][j][i] += G00*tmp;
            s_Gqs[es][k][j][i] += G01*tmp;
            s_Gqt[es][k][j][i] += G02*tmp;

            tmp = 0;

            // 's' terms
            // #pragma unroll p_cubUnr
            for(int m = 0; m < p_Nq; ++m) {
              const dfloat Djm = s_D[j][m];
              tmp += Djm*s_q[es][k][m][i];
            }

            const dfloat G11 = ggeo[p_Nggeo*element+p_G11ID];

            s_Gqr[es][k][j][i] += G01*tmp;
            s_Gqs[es][k][j][i] += G11*tmp;
            s_Gqt[es][k][j][i] += G12*tmp;
          }
        }
      }
    }

    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          if(eo+es<Nelements){
            const dfloat GWJ  = wJ[element];
            dfloat tmpAp = s_q[es][k][j][i]*lambda*GWJ;

            // use same matrix for both slices
            // #pragma unroll p_cubUnr
            for(int m=0;m<p_Nq;++m){
              const dfloat Dmi = s_D[m][i];
              const dfloat Dmj = s_D[m][j];

              tmpAp += Dmi*s_Gqr[es][k][j][m];
              tmpAp += Dmj*s_Gqs[es][k][m][i];
            }

            // #pragma unroll p_cubUnr
            for(int m=0;m<p_Nq;++m){
              const dfloat Gpt = s_Gqt[es][m][j][i];
              const dfloat Dmk = s_D[m][k];
              tmpAp += Dmk*Gpt;
            }

            s_q[es][k][j][i] = tmpAp;
          }
        }
      }
    }

    // test in 's'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          dfloat tmp=0.0;

          // #pragma unroll p_cubUnr
          for(int m=0;m<p_Nq;++m){
            const dfloat Imj = s_invV[m][j];
            tmp += Imj*s_q[es][k][m][i];
          }

          s_Gqr[es][k][j][i] = tmp;
        }
      }
    }

    // test in 'r'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){
          dfloat tmp=0.0;

          // #pragma unroll p_cubUnr
          for(int m=0;m<p_Nq;++m){
            const dfloat Imj = s_invV[m][i];
            tmp += Imj*s_Gqr[es][k][j][m];
          }

          s_Gqs[es][k][j][i] = tmp;
        }
      }
    }

    // test in 't'
    for(int ke=0;ke<p_Nq*p_NelementsPerBlk;++ke;@inner(2)){
      for(int j=0;j<p_Nq;++j;@inner(1)){
        for(int i=0;i<p_Nq;++i;@inner(0)){

          if(eo+es<Nelements){
            dfloat tmp=0.0;

            // #pragma unroll p_cubUnr
            for(int m=0;m<p_Nq;++m){
              const dfloat tmp2 = s_Gqs[es][m][j][i];
              tmp += s_invV[m][k]*tmp2;
            }

            const dlong base = i + j*p_Nq + k*p_Nq*p_Nq + element*p_Np;
            Aq[base] = tmp;
          }
        }
      }
    }
  }
}
