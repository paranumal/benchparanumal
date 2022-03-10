/*

The MIT License (MIT)

Copyright (c) 2020 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

void mesh_t::GeometricFactorsHex3D(){

  wJ.malloc(Nelements*Np);

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

    for(int k=0;k<Nq;++k){
      for(int j=0;j<Nq;++j){
        for(int i=0;i<Nq;++i){

          int n = i + j*Nq + k*Nq*Nq;

          dfloat xr = 0, xs = 0, xt = 0;
          dfloat yr = 0, ys = 0, yt = 0;
          dfloat zr = 0, zs = 0, zt = 0;
          for(int m=0;m<Nq;++m){
            int idr = e*Np + k*Nq*Nq + j*Nq + m;
            int ids = e*Np + k*Nq*Nq + m*Nq + i;
            int idt = e*Np + m*Nq*Nq + j*Nq + i;
            xr += D[i*Nq+m]*x[idr];
            xs += D[j*Nq+m]*x[ids];
            xt += D[k*Nq+m]*x[idt];
            yr += D[i*Nq+m]*y[idr];
            ys += D[j*Nq+m]*y[ids];
            yt += D[k*Nq+m]*y[idt];
            zr += D[i*Nq+m]*z[idr];
            zs += D[j*Nq+m]*z[ids];
            zt += D[k*Nq+m]*z[idt];
          }

          /* compute geometric factors for affine coordinate transform*/
          dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);

          LIBP_ABORT("Negative J found at element " << e,
                     J<1e-12);

          dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
          dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
          dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;

          dfloat JW = J*gllw[i]*gllw[j]*gllw[k];

          wJ[Np*e + n] = JW;

          /* store second order geometric factors */
          ggeo[Nggeo*(Np*e + n) + G00ID] = JW*(rx*rx + ry*ry + rz*rz);
          ggeo[Nggeo*(Np*e + n) + G01ID] = JW*(rx*sx + ry*sy + rz*sz);
          ggeo[Nggeo*(Np*e + n) + G02ID] = JW*(rx*tx + ry*ty + rz*tz);
          ggeo[Nggeo*(Np*e + n) + G11ID] = JW*(sx*sx + sy*sy + sz*sz);
          ggeo[Nggeo*(Np*e + n) + G12ID] = JW*(sx*tx + sy*ty + sz*tz);
          ggeo[Nggeo*(Np*e + n) + G22ID] = JW*(tx*tx + ty*ty + tz*tz);
        }
      }
    }
  }

  o_wJ = platform.malloc<dfloat>(wJ);
  o_ggeo = platform.malloc<dfloat>(ggeo);

  props["defines/" "p_Nggeo"]= Nggeo;

  props["defines/" "p_G00ID"]= G00ID;
  props["defines/" "p_G01ID"]= G01ID;
  props["defines/" "p_G02ID"]= G02ID;
  props["defines/" "p_G11ID"]= G11ID;
  props["defines/" "p_G12ID"]= G12ID;
  props["defines/" "p_G22ID"]= G22ID;
}

} //namespace libp
