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

void mesh_t::CubatureSetupTet3D(){

  LIBP_FORCE_ABORT("Tet node setup not complete.");

  /* Cubature data */
  cubN = 2*N; //cubature order
  // CubatureNodesTet3D(cubN, &cubNp, &cubr, &cubs, &cubt, &cubw);

  cubInterp.malloc(Np*cubNp);
  // InterpolationMatrixTet3D(N, Np, r, s, t, cubNp, cubr, cubs, cubt, cubInterp);

  //cubature project cubProject = M^{-1} * cubInterp^T
  // Defined such that cubProject * cubW * cubInterp = Identity
  cubProject.malloc(cubNp*Np);
  // CubaturePmatrixTet3D(N, Np, r, s, t, cubNp, cubr, cubs, cubt, cubProject);

  //cubature derivates matrices, cubD: differentiate on cubature nodes
  // we dont use cubD on Tris/Tets  so skip computing

  // add compile time constants to kernels
  props["defines/" "p_cubN"]= cubN;
  props["defines/" "p_cubNq"]= cubNq;
  props["defines/" "p_cubNp"]= cubNp;
  props["defines/" "p_cubNfp"]= cubNfp;

  // build transposes (we hold matrices as column major on device)
  memory<dfloat> cubProjectT(cubNp*Np);
  memory<dfloat> cubInterpT(cubNp*Np);
  linAlg_t::matrixTranspose(cubNp, Np, cubInterp.ptr(), Np, cubInterpT.ptr(), cubNp);
  linAlg_t::matrixTranspose(Np, cubNp, cubProject.ptr(), cubNp, cubProjectT.ptr(), Np);

  //pre-multiply cubProject by W on device
  for(int n=0;n<cubNp;++n){
    for(int m=0;m<Np;++m){
      cubProjectT[m+n*Np] *= cubw[n];
    }
  }

  o_cubInterp  = platform.malloc<dfloat>(cubInterpT);
  o_cubProject = platform.malloc<dfloat>(cubProjectT);
}

} //namespace libp
