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

#include "mesh.hpp"

namespace libp {

void mesh_t::GeometricFactorsTet3D(){

  wJ.malloc(Nelements*Np);

  /* number of first order geometric factors */
  Nvgeo = 9;
  vgeo.malloc(Nelements*Nvgeo*Np);

  RXID  = 0;
  RYID  = 1;
  RZID  = 2;
  SXID  = 3;
  SYID  = 4;
  SZID  = 5;
  TXID  = 6;
  TYID  = 7;
  TZID  = 8;

  /* number of second order geometric factors */
  Nggeo = 6;
  ggeo.malloc(Nelements*Nggeo*Np);

  G00ID=0;
  G01ID=1;
  G02ID=2;
  G11ID=3;
  G12ID=4;
  G22ID=5;

  #pragma omp parallel for
  for(dlong e=0;e<Nelements;++e){ /* for each element */
    for(int n=0;n<Np;++n){
      //differentiate physical coordinates
      dfloat xr = 0.0, xs = 0.0, xt = 0.0;
      dfloat yr = 0.0, ys = 0.0, yt = 0.0;
      dfloat zr = 0.0, zs = 0.0, zt = 0.0;

      for(int m=0;m<Np;++m){
        xr += Dr[n*Np+m]*x[m+e*Np];
        xs += Ds[n*Np+m]*x[m+e*Np];
        xt += Dt[n*Np+m]*x[m+e*Np];
        yr += Dr[n*Np+m]*y[m+e*Np];
        ys += Ds[n*Np+m]*y[m+e*Np];
        yt += Dt[n*Np+m]*y[m+e*Np];
        zr += Dr[n*Np+m]*z[m+e*Np];
        zs += Ds[n*Np+m]*z[m+e*Np];
        zt += Dt[n*Np+m]*z[m+e*Np];
      }

      /* compute geometric factors for affine coordinate transform*/
      dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);

      dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
      dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
      dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;

      LIBP_ABORT("Negative J found at element " << e,
                 J<0);

      wJ[Np*e + n] = J;

      /* store geometric factors */
      dlong vbase = Nvgeo*(Np*e + n);
      vgeo[vbase + RXID] = rx;
      vgeo[vbase + RYID] = ry;
      vgeo[vbase + RZID] = rz;
      vgeo[vbase + SXID] = sx;
      vgeo[vbase + SYID] = sy;
      vgeo[vbase + SZID] = sz;
      vgeo[vbase + TXID] = tx;
      vgeo[vbase + TYID] = ty;
      vgeo[vbase + TZID] = tz;

      /* store second order geometric factors */
      dlong gbase = Nggeo*(Np*e + n);
      ggeo[gbase + G00ID] = J*(rx*rx + ry*ry + rz*rz);
      ggeo[gbase + G01ID] = J*(rx*sx + ry*sy + rz*sz);
      ggeo[gbase + G02ID] = J*(rx*tx + ry*ty + rz*tz);
      ggeo[gbase + G11ID] = J*(sx*sx + sy*sy + sz*sz);
      ggeo[gbase + G12ID] = J*(sx*tx + sy*ty + sz*tz);
      ggeo[gbase + G22ID] = J*(tx*tx + ty*ty + tz*tz);
    }
  }

  o_wJ = platform.malloc<dfloat>(wJ);
  o_vgeo = platform.malloc<dfloat>(vgeo);
  o_ggeo = platform.malloc<dfloat>(ggeo);

  props["defines/" "p_Nvgeo"]= Nvgeo;
  props["defines/" "p_RXID"]= RXID;
  props["defines/" "p_RYID"]= RYID;
  props["defines/" "p_RZID"]= RZID;
  props["defines/" "p_SXID"]= SXID;
  props["defines/" "p_SYID"]= SYID;
  props["defines/" "p_SZID"]= SZID;
  props["defines/" "p_TXID"]= TXID;
  props["defines/" "p_TYID"]= TYID;
  props["defines/" "p_TZID"]= TZID;

  props["defines/" "p_Nggeo"]= Nggeo;
  props["defines/" "p_G00ID"]= G00ID;
  props["defines/" "p_G01ID"]= G01ID;
  props["defines/" "p_G02ID"]= G02ID;
  props["defines/" "p_G11ID"]= G11ID;
  props["defines/" "p_G12ID"]= G12ID;
  props["defines/" "p_G22ID"]= G22ID;
}

} //namespace libp
