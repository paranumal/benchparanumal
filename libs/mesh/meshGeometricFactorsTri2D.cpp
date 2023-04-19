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

void mesh_t::GeometricFactorsTri2D(){

  /* number of first order geometric factors */
  Nvgeo = 4;
  RXID  = 0;
  RYID  = 1;
  SXID  = 2;
  SYID  = 3;

  /* number of second order geometric factors */
  Nggeo = 3;
  G00ID=0;
  G01ID=1;
  G11ID=2;

  if (settings.compareSetting("AFFINE MESH", "TRUE")) {
    wJ.malloc(Nelements);
    vgeo.malloc(Nelements*Nvgeo);
    ggeo.malloc(Nelements*Nggeo);

    #pragma omp parallel for
    for(dlong e=0;e<Nelements;++e){ /* for each element */
      /* find vertex indices and physical coordinates */
      dlong id = e*Nverts+0;

      dfloat xe1 = EX[id+0];
      dfloat xe2 = EX[id+1];
      dfloat xe3 = EX[id+2];

      dfloat ye1 = EY[id+0];
      dfloat ye2 = EY[id+1];
      dfloat ye3 = EY[id+2];

      dfloat xr = 0.5*(xe2-xe1);
      dfloat xs = 0.5*(xe3-xe1);
      dfloat yr = 0.5*(ye2-ye1);
      dfloat ys = 0.5*(ye3-ye1);

      /* compute geometric factors for affine coordinate transform*/
      dfloat J = xr*ys - xs*yr;

      LIBP_ABORT("Negative J found at element " << e,
                 J<1e-8);

      dfloat rx =  ys/J;
      dfloat ry = -xs/J;
      dfloat sx = -yr/J;
      dfloat sy =  xr/J;

      wJ[e] = J;

      /* store geometric factors */
      dlong vbase = Nvgeo*e;
      vgeo[vbase + RXID] = rx;
      vgeo[vbase + RYID] = ry;
      vgeo[vbase + SXID] = sx;
      vgeo[vbase + SYID] = sy;

      /* store second order geometric factors */
      dlong gbase = Nggeo*e;
      ggeo[gbase + G00ID] = J*(rx*rx + ry*ry);
      ggeo[gbase + G01ID] = J*(rx*sx + ry*sy);
      ggeo[gbase + G11ID] = J*(sx*sx + sy*sy);
    }
  } else {
    wJ.malloc(Nelements*Np);
    vgeo.malloc(Nelements*Nvgeo*Np);
    ggeo.malloc(Nelements*Nggeo*Np);

    #pragma omp parallel for
    for(dlong e=0;e<Nelements;++e){ /* for each element */
      for(int n=0;n<Np;++n){
        //differentiate physical coordinates
        dfloat xr = 0.0, xs = 0.0;
        dfloat yr = 0.0, ys = 0.0;

        for(int m=0;m<Np;++m){
          xr += Dr[n*Np+m]*x[m+e*Np];
          xs += Ds[n*Np+m]*x[m+e*Np];
          yr += Dr[n*Np+m]*y[m+e*Np];
          ys += Ds[n*Np+m]*y[m+e*Np];
        }

        /* compute geometric factors for affine coordinate transform*/
        dfloat J = xr*ys - xs*yr;

        LIBP_ABORT("Negative J found at element " << e,
                   J<1e-8);

        dfloat rx =  ys/J;
        dfloat ry = -xs/J;
        dfloat sx = -yr/J;
        dfloat sy =  xr/J;

        wJ[Np*e + n] = J;

        /* store geometric factors */
        dlong vbase = Nvgeo*(Np*e + n);
        vgeo[vbase + RXID] = rx;
        vgeo[vbase + RYID] = ry;
        vgeo[vbase + SXID] = sx;
        vgeo[vbase + SYID] = sy;

        /* store second order geometric factors */
        dlong gbase = Nggeo*(Np*e + n);
        ggeo[gbase + G00ID] = J*(rx*rx + ry*ry);
        ggeo[gbase + G01ID] = J*(rx*sx + ry*sy);
        ggeo[gbase + G11ID] = J*(sx*sx + sy*sy);
      }
    }
  }

  o_wJ = platform.malloc<dfloat>(wJ);
  o_vgeo = platform.malloc<dfloat>(vgeo);
  o_ggeo = platform.malloc<dfloat>(ggeo);

  props["defines/" "p_Nvgeo"]= Nvgeo;
  props["defines/" "p_RXID"]= RXID;
  props["defines/" "p_SXID"]= SXID;
  props["defines/" "p_RYID"]= RYID;
  props["defines/" "p_SYID"]= SYID;

  props["defines/" "p_Nggeo"]= Nggeo;
  props["defines/" "p_G00ID"]= G00ID;
  props["defines/" "p_G01ID"]= G01ID;
  props["defines/" "p_G11ID"]= G11ID;
}

} //namespace libp
