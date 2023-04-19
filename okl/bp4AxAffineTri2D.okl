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

@kernel void bp4AxAffineTri2D(const dlong Nelements,
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

    @shared dfloat s_q[p_Nfields][p_Np];

    @exclusive dlong element;

    for(int n=0;n<p_Np;++n;@inner(0)){

      element = elementList[e];

      const dlong base = n + element*p_Np*p_Nfields;
      for(int f=0;f<p_Nfields;++f) {
        const dlong id = GlobalToLocal[base + f*p_Np];
        s_q[f][n] = (id!=-1) ? q[id] : 0.0;
      }
    }

    for(int n=0;n<p_Np;++n;@inner(0)){
      const dfloat J   = wJ[element];
      const dfloat Grr = ggeo[p_Nggeo*element + p_G00ID];
      const dfloat Grs = ggeo[p_Nggeo*element + p_G01ID];
      const dfloat Gss = ggeo[p_Nggeo*element + p_G11ID];

      dfloat qrr[p_Nfields];
      dfloat qrs[p_Nfields];
      dfloat qss[p_Nfields];
      dfloat qM [p_Nfields];
      for(int f=0;f<p_Nfields;++f) {
        qrr[f] = 0.0;
        qrs[f] = 0.0;
        qss[f] = 0.0;
        qM [f] = 0.0;
      }

      // #pragma unroll p_Np
      for (int k=0;k<p_Np;k++) {
        for(int f=0;f<p_Nfields;++f) {
          dfloat qn = s_q[f][k];
          qrr[f] += S[n+k*p_Np+0*p_Np*p_Np]*qn;
          qrs[f] += S[n+k*p_Np+1*p_Np*p_Np]*qn;
          qss[f] += S[n+k*p_Np+2*p_Np*p_Np]*qn;
          qM [f] += MM[n+k*p_Np]*qn;
        }
      }

      const dlong id = n + element*p_Np*p_Nfields;
      for(int f=0;f<p_Nfields;++f) {
        Aq[id+f*p_Np] = Grr*qrr[f]+Grs*qrs[f]+Gss*qss[f] + J*lambda*qM[f];
      }
    }
  }
}
