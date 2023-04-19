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

@kernel void bp4AxTri2D(const dlong Nelements,
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

  for(int e=0;e<Nelements;e++;@outer(0)){

    @shared dfloat   s_q[p_Nfields][p_cubNp];
    @shared dfloat s_Gqr[p_Nfields][p_cubNp];
    @shared dfloat s_Gqs[p_Nfields][p_cubNp];

    @exclusive dlong element;

    for(int n=0;n<p_cubNp;++n;@inner(0)){

      element = elementList[e];

      if (n<p_Np) {
        const dlong base = n + element*p_Np*p_Nfields;
        for (int f=0;f<p_Nfields;++f) {
          const dlong id = GlobalToLocal[base + f*p_Np];
          s_q[f][n] = (id!=-1) ? q[id] : 0.0;
        }
      }
    }

    @exclusive dfloat Iq[p_Nfields], qr[p_Nfields], qs[p_Nfields];

    for(int n=0;n<p_cubNp;++n;@inner(0)){

      for (int f=0;f<p_Nfields;++f) {
        Iq[f] = 0.0;
        qr[f] = 0.0;
        qs[f] = 0.0;
      }

      for(int m=0;m<p_Np;++m){
        const dfloat Im = I[n + m*p_cubNp];
        const dfloat Dr = D[n + m*p_cubNp + 0*p_Np*p_cubNp];
        const dfloat Ds = D[n + m*p_cubNp + 1*p_Np*p_cubNp];
        for (int f=0;f<p_Nfields;++f) {
          const dfloat qm = s_q[f][m];
          Iq[f] += Im * qm;
          qr[f] += Dr * qm;
          qs[f] += Ds * qm;
        }
      }
    }

    for(int n=0;n<p_cubNp;++n;@inner(0)){
      const dlong gbase = element*p_cubNp + n;
      const dfloat GWJ = cubwJ[gbase];
      const dfloat G00 = cubggeo[p_Nggeo*gbase+p_G00ID];
      const dfloat G01 = cubggeo[p_Nggeo*gbase+p_G01ID];
      const dfloat G11 = cubggeo[p_Nggeo*gbase+p_G11ID];

      for (int f=0;f<p_Nfields;++f) {
        s_q  [f][n] = lambda * GWJ * Iq[f];
        s_Gqr[f][n] = G00 * qr[f] + G01 * qs[f];
        s_Gqs[f][n] = G01 * qr[f] + G11 * qs[f];
      }
    }

    for(int n=0;n<p_cubNp;++n;@inner(0)){

      if (n<p_Np) {

        dfloat Aqn[p_Nfields];

        for (int f=0;f<p_Nfields;++f) {
          Aqn[f] = 0.0;
        }

        for(int m=0;m<p_cubNp;++m){
          const dfloat ITm = I[m + n*p_cubNp];
          const dfloat DrT = D[m + n*p_cubNp + 0*p_Np*p_cubNp];
          const dfloat DsT = D[m + n*p_cubNp + 1*p_Np*p_cubNp];
          for (int f=0;f<p_Nfields;++f) {
            Aqn[f] += ITm * s_q  [f][m];
            Aqn[f] += DrT * s_Gqr[f][m];
            Aqn[f] += DsT * s_Gqs[f][m];
          }
        }

        const dlong base = n + element*p_Np*p_Nfields;
        for (int f=0;f<p_Nfields;++f) {
          Aq[base+f*p_Np] = Aqn[f];
        }
      }
    }
  }
}
