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

void mesh_t::GeometricFactorsHex3D(){

  /* number of second order geometric factors */
  Nggeo = 6;
  G00ID=0;
  G01ID=1;
  G02ID=2;
  G11ID=3;
  G12ID=4;
  G22ID=5;

  if (settings.compareSetting("AFFINE MESH", "TRUE")) {
    wJ.malloc(Nelements);
    ggeo.malloc(Nelements*Nggeo);

#pragma omp parallel for
    for(dlong e=0;e<Nelements;++e){ /* for each element */
      /* find vertex indices and physical coordinates */
      dlong id = e*Nverts+0;

      /* vertex coordinates */
      dfloat xe1 = EX[id+0], ye1 = EY[id+0], ze1 = EZ[id+0];
      dfloat xe2 = EX[id+1], ye2 = EY[id+1], ze2 = EZ[id+1];
      dfloat xe3 = EX[id+3], ye3 = EY[id+3], ze3 = EZ[id+3];
      dfloat xe4 = EX[id+4], ye4 = EY[id+4], ze4 = EZ[id+4];

      /* Jacobian matrix */
      dfloat xr = 0.5*(xe2-xe1), xs = 0.5*(xe3-xe1), xt = 0.5*(xe4-xe1);
      dfloat yr = 0.5*(ye2-ye1), ys = 0.5*(ye3-ye1), yt = 0.5*(ye4-ye1);
      dfloat zr = 0.5*(ze2-ze1), zs = 0.5*(ze3-ze1), zt = 0.5*(ze4-ze1);

      /* compute geometric factors for affine coordinate transform*/
      dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);

      dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
      dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
      dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;

      LIBP_ABORT("Negative J found at element " << e,
                 J<0);

      wJ[e] = J;

      /* store second order geometric factors */
      dlong gbase = Nggeo*e;
      ggeo[gbase + G00ID] = J*(rx*rx + ry*ry + rz*rz);
      ggeo[gbase + G01ID] = J*(rx*sx + ry*sy + rz*sz);
      ggeo[gbase + G02ID] = J*(rx*tx + ry*ty + rz*tz);
      ggeo[gbase + G11ID] = J*(sx*sx + sy*sy + sz*sz);
      ggeo[gbase + G12ID] = J*(sx*tx + sy*ty + sz*tz);
      ggeo[gbase + G22ID] = J*(tx*tx + ty*ty + tz*tz);
    }

  } else {
    wJ.malloc(Nelements*Np);
    ggeo.malloc(Nelements*Nggeo*Np);

    //    #pragma omp parallel for
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

#if 0
	    printf("xr=%e,yr=%e,zr=%e,xs=%e,ys=%e,zs=%e,xt=%e,yt=%e,zt=%e,J=%g\n",
		   xr, yr, zr, xs, ys, zs, xt, yt, zt, J);
#endif
	    
            LIBP_ABORT("Negative J found at element " << e, J<1e-12);

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
