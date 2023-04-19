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

@kernel void bp6AxTri2D(const dlong Nelements,
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

  for(int e=0;e<Nelements;e++;@outer(0)){

    @shared dfloat  s_q[p_Nfields][p_Np];
    @shared dfloat s_qx[p_Nfields][p_Np];
    @shared dfloat s_qy[p_Nfields][p_Np];

    @exclusive dlong element;

    for(int n=0;n<p_Np;++n;@inner(0)){

      element = elementList[e];

      const dlong base = n + element*p_Np*p_Nfields;
      for (int f=0;f<p_Nfields;++f) {
        const dlong id = GlobalToLocal[base+f*p_Np];
        s_q[f][n] = (id!=-1) ? q[id] : 0.0;
      }
    }

    @exclusive dfloat qr[p_Nfields], qs[p_Nfields];

    for(int n=0;n<p_Np;++n;@inner(0)){

      for (int f=0;f<p_Nfields;++f) {
        qr[f] = 0.0;
        qs[f] = 0.0;
      }

      for(int m=0;m<p_Np;++m){
        const dfloat Dr = D[n + m*p_Np + 0*p_Np*p_Np];
        const dfloat Ds = D[n + m*p_Np + 1*p_Np*p_Np];
        for (int f=0;f<p_Nfields;++f) {
          const dfloat qm = s_q[f][m];
          qr[f] += Dr * qm;
          qs[f] += Ds * qm;
        }
      }
    }

    @exclusive dfloat rx, ry, sx, sy;

    for(int n=0;n<p_Np;++n;@inner(0)){
      const dlong base = element*p_Np + n;
      const dfloat J = wJ[base];
      rx = vgeo[p_Nvgeo*base+p_RXID];
      ry = vgeo[p_Nvgeo*base+p_RYID];
      sx = vgeo[p_Nvgeo*base+p_SXID];
      sy = vgeo[p_Nvgeo*base+p_SYID];

      for (int f=0;f<p_Nfields;++f) {
        //this is dubious, since we lose symmetry for truly curved elements
        s_q [f][n] *= lambda * J;

        s_qx[f][n] = rx * qr[f] + sx * qs[f];
        s_qy[f][n] = ry * qr[f] + sy * qs[f];
      }
    }

    @exclusive dfloat Mq[p_Nfields], Mqx[p_Nfields], Mqy[p_Nfields];

    for(int n=0;n<p_Np;++n;@inner(0)){
      for (int f=0;f<p_Nfields;++f) {
        Mq [f] = 0.0;
        Mqx[f] = 0.0;
        Mqy[f] = 0.0;
      }

      for(int m=0;m<p_Np;++m){
        const dfloat MMn = MM[n + m*p_Np];
        for (int f=0;f<p_Nfields;++f) {
          Mq [f] += MMn * s_q [f][m];
          Mqx[f] += MMn * s_qx[f][m];
          Mqy[f] += MMn * s_qy[f][m];
        }
      }
    }

    for(int n=0;n<p_Np;++n;@inner(0)){
      for (int f=0;f<p_Nfields;++f) {
        s_qx[f][n] = rx * Mqx[f] + ry * Mqy[f];
        s_qy[f][n] = sx * Mqx[f] + sy * Mqy[f];
      }
    }

    for(int n=0;n<p_Np;++n;@inner(0)){

      dfloat Aqn[p_Nfields];

      for (int f=0;f<p_Nfields;++f) {
        Aqn[f] = Mq[f];
      }

      for(int m=0;m<p_Np;++m){
        const dfloat DrT = D[m + n*p_Np + 0*p_Np*p_Np];
        const dfloat DsT = D[m + n*p_Np + 1*p_Np*p_Np];
        for (int f=0;f<p_Nfields;++f) {
          Aqn[f] += DrT * s_qx[f][m];
          Aqn[f] += DsT * s_qy[f][m];
        }
      }

      const dlong base = n + element*p_Np*p_Nfields;
      for (int f=0;f<p_Nfields;++f) {
        Aq[base+f*p_Np] = Aqn[f];
      }
    }
  }
}
