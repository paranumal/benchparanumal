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

@kernel void bp5AxTet3D(const dlong Nelements,
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

    @shared dfloat  s_q[p_Np];
    @shared dfloat s_qx[p_Np];
    @shared dfloat s_qy[p_Np];
    @shared dfloat s_qz[p_Np];

    @exclusive dlong element;

    for(int n=0;n<p_Np;++n;@inner(0)){

      element = elementList[e];

      const dlong id = GlobalToLocal[n + element*p_Np];
      s_q[n] = (id!=-1) ? q[id] : 0.0;
    }

    @exclusive dfloat qr, qs, qt;

    for(int n=0;n<p_Np;++n;@inner(0)){

      qr = 0.0;
      qs = 0.0;
      qt = 0.0;

      for(int m=0;m<p_Np;++m){
        const dfloat qm = s_q[m];
        const dfloat Dr = D[n + m*p_Np + 0*p_Np*p_Np];
        const dfloat Ds = D[n + m*p_Np + 1*p_Np*p_Np];
        const dfloat Dt = D[n + m*p_Np + 2*p_Np*p_Np];
        qr += Dr * qm;
        qs += Ds * qm;
        qt += Dt * qm;
      }
    }

    @exclusive dfloat rx, ry, rz, sx, sy, sz, tx, ty, tz;

    for(int n=0;n<p_Np;++n;@inner(0)){
      const dlong base = element*p_Np + n;
      const dfloat J = wJ[base];
      rx = vgeo[p_Nvgeo*base+p_RXID];
      ry = vgeo[p_Nvgeo*base+p_RYID];
      rz = vgeo[p_Nvgeo*base+p_RZID];
      sx = vgeo[p_Nvgeo*base+p_SXID];
      sy = vgeo[p_Nvgeo*base+p_SYID];
      sz = vgeo[p_Nvgeo*base+p_SZID];
      tx = vgeo[p_Nvgeo*base+p_TXID];
      ty = vgeo[p_Nvgeo*base+p_TYID];
      tz = vgeo[p_Nvgeo*base+p_TZID];

      //this is dubious, since we lose symmetry for truly curved elements
      s_q[n] *= lambda * J;

      s_qx[n] = rx * qr + sx * qs + tx * qt;
      s_qy[n] = ry * qr + sy * qs + ty * qt;
      s_qz[n] = rz * qr + sz * qs + tz * qt;
    }

    @exclusive dfloat Mq, Mqx, Mqy, Mqz;

    for(int n=0;n<p_Np;++n;@inner(0)){
      Mq  = 0.0;
      Mqx = 0.0;
      Mqy = 0.0;
      Mqz = 0.0;

      for(int m=0;m<p_Np;++m){
        const dfloat MMn = MM[n + m*p_Np];
        Mq  += MMn * s_q[m];
        Mqx += MMn * s_qx[m];
        Mqy += MMn * s_qy[m];
        Mqz += MMn * s_qz[m];
      }
    }

    for(int n=0;n<p_Np;++n;@inner(0)){
      s_qx[n] = rx * Mqx + ry * Mqy + rz * Mqz;
      s_qy[n] = sx * Mqx + sy * Mqy + sz * Mqz;
      s_qz[n] = tx * Mqx + ty * Mqy + tz * Mqz;
    }

    for(int n=0;n<p_Np;++n;@inner(0)){

      dfloat Aqn = Mq;

      for(int m=0;m<p_Np;++m){
        const dfloat DrT = D[m + n*p_Np + 0*p_Np*p_Np];
        const dfloat DsT = D[m + n*p_Np + 1*p_Np*p_Np];
        const dfloat DtT = D[m + n*p_Np + 2*p_Np*p_Np];
        Aqn += DrT * s_qx[m];
        Aqn += DsT * s_qy[m];
        Aqn += DtT * s_qz[m];
      }

      const dlong base = n + element*p_Np;
      Aq[base] = Aqn;
    }
  }
}
