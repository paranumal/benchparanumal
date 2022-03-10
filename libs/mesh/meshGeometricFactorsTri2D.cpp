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

  wJ.malloc(Nelements);

  /* number of second order geometric factors */
  Nggeo = 3;
  ggeo.malloc(Nelements*Nggeo);

  G00ID=0;
  G01ID=1;
  G11ID=2;

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

    /* compute geometric factors for affine coordinate transform*/
    dfloat J = 0.25*((xe2-xe1)*(ye3-ye1) - (xe3-xe1)*(ye2-ye1));

    LIBP_ABORT("Negative J found at element " << e,
               J<0);

    dfloat rx =  (0.5/J)*(ye3-ye1);
    dfloat ry = -(0.5/J)*(xe3-xe1);
    dfloat sx = -(0.5/J)*(ye2-ye1);
    dfloat sy =  (0.5/J)*(xe2-xe1);

    wJ[e] = J;

    /* store second order geometric factors */
    ggeo[Nggeo*e + G00ID] = J*(rx*rx + ry*ry);
    ggeo[Nggeo*e + G01ID] = J*(rx*sx + ry*sy);
    ggeo[Nggeo*e + G11ID] = J*(sx*sx + sy*sy);
  }

  o_wJ = platform.malloc<dfloat>(wJ);
  o_ggeo = platform.malloc<dfloat>(ggeo);

  props["defines/" "p_Nggeo"]= Nggeo;

  props["defines/" "p_G00ID"]= G00ID;
  props["defines/" "p_G01ID"]= G01ID;
  props["defines/" "p_G11ID"]= G11ID;
}

} //namespace libp
