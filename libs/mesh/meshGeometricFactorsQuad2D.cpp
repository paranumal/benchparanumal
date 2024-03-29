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

void mesh_t::GeometricFactorsQuad2D(){

  /* number of second order geometric factors */
  Nggeo = 3;

  G00ID=0;
  G01ID=1;
  G11ID=2;

  if (settings.compareSetting("AFFINE MESH", "TRUE")) {
    wJ.malloc(Nelements);
    ggeo.malloc(Nelements*Nggeo);

    #pragma omp parallel for
    for(dlong e=0;e<Nelements;++e){ /* for each element */
      /* find vertex indices and physical coordinates */
      dlong id = e*Nverts+0;

      dfloat xe1 = EX[id+0];
      dfloat xe2 = EX[id+1];
      dfloat xe3 = EX[id+3];

      dfloat ye1 = EY[id+0];
      dfloat ye2 = EY[id+1];
      dfloat ye3 = EY[id+3];

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

      /* store second order geometric factors */
      dlong gbase = Nggeo*e;
      ggeo[gbase + G00ID] = J*(rx*rx + ry*ry);
      ggeo[gbase + G01ID] = J*(rx*sx + ry*sy);
      ggeo[gbase + G11ID] = J*(sx*sx + sy*sy);
    }
  } else {
    wJ.malloc(Nelements*Np);
    ggeo.malloc(Nelements*Nggeo*Np);

    #pragma omp parallel for
    for(dlong e=0;e<Nelements;++e){ /* for each element */
      for(int j=0;j<Nq;++j){
        for(int i=0;i<Nq;++i){

          int n = i + j*Nq;

          //differentiate physical coordinates
          dfloat xr = 0.0;
          dfloat xs = 0.0;
          dfloat yr = 0.0;
          dfloat ys = 0.0;

          for(int m=0;m<Nq;++m){
            int idr = e*Np + j*Nq + m;
            int ids = e*Np + m*Nq + i;
            xr += D[i*Nq+m]*x[idr];
            xs += D[j*Nq+m]*x[ids];
            yr += D[i*Nq+m]*y[idr];
            ys += D[j*Nq+m]*y[ids];
          }

          /* compute geometric factors for affine coordinate transform*/
          dfloat J = xr*ys - xs*yr;

          LIBP_ABORT("Negative J found at element " << e,
                     J<1e-8);

          dfloat rx =  ys/J;
          dfloat ry = -xs/J;
          dfloat sx = -yr/J;
          dfloat sy =  xr/J;
          dfloat JW = J*gllw[i]*gllw[j];

          wJ[Np*e + n] = JW;

          /* store second order geometric factors */
          ggeo[Nggeo*(Np*e + n) + G00ID] = JW*(rx*rx + ry*ry);
          ggeo[Nggeo*(Np*e + n) + G01ID] = JW*(rx*sx + ry*sy);
          ggeo[Nggeo*(Np*e + n) + G11ID] = JW*(sx*sx + sy*sy);
        }
      }
    }
  }

  o_wJ = platform.malloc<dfloat>(wJ);
  o_ggeo = platform.malloc<dfloat>(ggeo);

  props["defines/" "p_Nggeo"]= Nggeo;

  props["defines/" "p_G00ID"]= G00ID;
  props["defines/" "p_G01ID"]= G01ID;
  props["defines/" "p_G11ID"]= G11ID;
}

} //namespace libp
