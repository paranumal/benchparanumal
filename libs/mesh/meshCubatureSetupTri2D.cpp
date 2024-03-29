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

void mesh_t::CubatureSetupTri2D(){

  /* Cubature data */
  cubN = 2*N+2; //cubature order
  CubatureNodesTri2D(cubN, cubNp, cubr, cubs, cubw);

  InterpolationMatrixTri2D(N, r, s, cubr, cubs, cubInterp);

  // add compile time constants to kernels
  props["defines/" "p_cubN"]= cubN;
  props["defines/" "p_cubNp"]= cubNp;

  // build transposes (we hold matrices as column major on device)
  memory<dfloat> cubInterpT(cubNp*Np);
  linAlg_t::matrixTranspose(cubNp, Np, cubInterp, Np, cubInterpT, cubNp);
  o_cubInterp  = platform.malloc<dfloat>(cubInterpT);

  cubD.malloc(2*cubNp*Np);
  memory<dfloat> cubDr = cubD + 0*cubNp*Np;
  memory<dfloat> cubDs = cubD + 1*cubNp*Np;

  for (int m=0;m<cubNp;++m) {
    for (int n=0;n<Np;++n) {
      cubDr[n+m*Np] = 0.0;
      cubDs[n+m*Np] = 0.0;
      for (int k=0;k<Np;++k) {
        cubDr[n+m*Np] += cubInterp[k+m*Np]*Dr[n+k*Np];
        cubDs[n+m*Np] += cubInterp[k+m*Np]*Ds[n+k*Np];
      }
    }
  }

  memory<dfloat> cubDT(2*cubNp*Np);
  memory<dfloat> cubDrT = cubDT + 0*Np*cubNp;
  memory<dfloat> cubDsT = cubDT + 1*Np*cubNp;
  linAlg_t::matrixTranspose(cubNp, Np, cubDr, Np, cubDrT, cubNp);
  linAlg_t::matrixTranspose(cubNp, Np, cubDs, Np, cubDsT, cubNp);
  o_cubD = platform.malloc<dfloat>(cubDT);

  /*Geofactors*/
  if (settings.compareSetting("AFFINE MESH", "TRUE")) {

    cubwJ = wJ;
    cubggeo = ggeo;
    o_cubwJ = o_wJ;
    o_cubggeo = o_ggeo;

  } else {

    cubwJ.malloc(Nelements*cubNp);
    // cubvgeo.malloc(Nelements*Nvgeo*cubNp);
    // cubsgeo.malloc(Nelements*Nsgeo*cubNfp*Nfaces);
    cubggeo.malloc(Nelements*Nggeo*cubNp);

    //temp arrays
    memory<dfloat> xre(Np);
    memory<dfloat> xse(Np);
    memory<dfloat> yre(Np);
    memory<dfloat> yse(Np);

    //geometric data for quadrature
    for(dlong e=0;e<Nelements;++e){ /* for each element */
      for(int n=0;n<Np;++n){
        //differentiate physical coordinates
        xre[n] = 0.0; xse[n] = 0.0;
        yre[n] = 0.0; yse[n] = 0.0;

        for(int m=0;m<Np;++m){
          xre[n] += Dr[n*Np+m]*x[m+e*Np];
          xse[n] += Ds[n*Np+m]*x[m+e*Np];
          yre[n] += Dr[n*Np+m]*y[m+e*Np];
          yse[n] += Ds[n*Np+m]*y[m+e*Np];
        }
      }

      //interpolate derivaties to cubature
      for(int n=0;n<cubNp;++n){
        //differentiate physical coordinates
        dfloat xr = 0.0, xs = 0.0;
        dfloat yr = 0.0, ys = 0.0;

        for(int m=0;m<Np;++m){
          xr += cubInterp[n*Np+m]*xre[m];
          xs += cubInterp[n*Np+m]*xse[m];
          yr += cubInterp[n*Np+m]*yre[m];
          ys += cubInterp[n*Np+m]*yse[m];
        }

        /* compute geometric factors for affine coordinate transform*/
        dfloat J = xr*ys - xs*yr;

        LIBP_ABORT("Negative J found at element " << e,
                   J<1e-8);

        dfloat rx =  ys/J;
        dfloat ry = -xs/J;
        dfloat sx = -yr/J;
        dfloat sy =  xr/J;

        dfloat JW = J*cubw[n];

        cubwJ[cubNp*e + n] = JW;

        /* store second order geometric factors */
        dlong base = Nggeo*(cubNp*e + n);
        cubggeo[base + G00ID] = JW*(rx*rx + ry*ry);
        cubggeo[base + G01ID] = JW*(rx*sx + ry*sy);
        cubggeo[base + G11ID] = JW*(sx*sx + sy*sy);
      }
    }

    o_cubwJ = platform.malloc<dfloat>(cubwJ);
    o_cubggeo = platform.malloc<dfloat>(cubggeo);
  }

}

} //namespace libp
