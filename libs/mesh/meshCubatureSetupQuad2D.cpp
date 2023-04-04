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

void mesh_t::CubatureSetupQuad2D(){

  /* Quadrature data */
  cubN = N;
  cubNq = cubN+1;
  cubNp = cubNq*cubNq;
  cubNfp = cubNq;

  // cubN+1 point Gauss-Legendre quadrature
  JacobiGQ(0, 0, cubN, cubr, cubw);

  // GLL to GL interpolation matrix
  InterpolationMatrix1D(N, gllz, cubr, cubInterp);

  //cubature project cubProject = cubInterp^T
  cubProject.malloc(cubNq*Nq);
  linAlg_t::matrixTranspose(cubNq, Nq, cubInterp, Nq, cubProject, cubNq);

  o_cubInterp  = platform.malloc<dfloat>(cubInterp);
  o_cubProject = platform.malloc<dfloat>(cubProject);

  //cubature derivates matrix, cubD: differentiate on cubature nodes
  Dmatrix1D(cubN, cubr, cubr, cubD);
  o_cubD = platform.malloc<dfloat>(cubD);

  // add compile time constants to kernels
  props["defines/" "p_cubN"]= cubN;
  props["defines/" "p_cubNq"]= cubNq;
  props["defines/" "p_cubNp"]= cubNp;
  props["defines/" "p_cubNfp"]= cubNfp;


  /*Geofactors*/
  cubwJ.malloc(Nelements*cubNp);
  // cubvgeo.malloc(Nelements*Nvgeo*cubNp);
  // cubsgeo.malloc(Nelements*Nsgeo*cubNfp*Nfaces);
  cubggeo.malloc(Nelements*Nggeo*cubNp);

  //temp arrays
  memory<dfloat> xre(Np);
  memory<dfloat> xse(Np);
  memory<dfloat> yre(Np);
  memory<dfloat> yse(Np);

  memory<dfloat> xre1(cubNq*Nq);
  memory<dfloat> xse1(cubNq*Nq);
  memory<dfloat> yre1(cubNq*Nq);
  memory<dfloat> yse1(cubNq*Nq);

  //geometric data for quadrature
  for(dlong e=0;e<Nelements;++e){ /* for each element */
    for(int j=0;j<Nq;++j){
      for(int i=0;i<Nq;++i){
        int n = i + j*Nq;

        //differentiate physical coordinates
        xre[n] = 0.0; xse[n] = 0.0;
        yre[n] = 0.0; yse[n] = 0.0;

        for(int m=0;m<Nq;++m){
          int idr = e*Np + j*Nq + m;
          int ids = e*Np + m*Nq + i;
          xre[n] += D[i*Nq+m]*x[idr];
          xse[n] += D[j*Nq+m]*x[ids];
          yre[n] += D[i*Nq+m]*y[idr];
          yse[n] += D[j*Nq+m]*y[ids];
        }
      }
    }

    //interpolate derivaties to cubature
    for(int j=0;j<Nq;++j){
      for(int i=0;i<cubNq;++i){
        xre1[j*cubNq+i] = 0.0; xse1[j*cubNq+i] = 0.0;
        yre1[j*cubNq+i] = 0.0; yse1[j*cubNq+i] = 0.0;
        for(int n=0;n<Nq;++n){
          xre1[j*cubNq+i] += cubInterp[i*Nq + n]*xre[j*Nq+n];
          xse1[j*cubNq+i] += cubInterp[i*Nq + n]*xse[j*Nq+n];
          yre1[j*cubNq+i] += cubInterp[i*Nq + n]*yre[j*Nq+n];
          yse1[j*cubNq+i] += cubInterp[i*Nq + n]*yse[j*Nq+n];
        }
      }
    }

    for(int j=0;j<cubNq;++j){
      for(int i=0;i<cubNq;++i){
        dfloat xr = 0.0, xs = 0.0;
        dfloat yr = 0.0, ys = 0.0;
        for(int n=0;n<Nq;++n){
          xr += cubInterp[j*Nq + n]*xre1[n*cubNq+i];
          xs += cubInterp[j*Nq + n]*xse1[n*cubNq+i];
          yr += cubInterp[j*Nq + n]*yre1[n*cubNq+i];
          ys += cubInterp[j*Nq + n]*yse1[n*cubNq+i];
        }

        /* compute geometric factors for affine coordinate transform*/
        dfloat J = xr*ys - xs*yr;

        LIBP_ABORT("Negative J found at element " << e,
                   J<1e-8);

        dfloat rx =  ys/J;
        dfloat ry = -xs/J;
        dfloat sx = -yr/J;
        dfloat sy =  xr/J;

        dfloat JW = J*cubw[i]*cubw[j];

        cubwJ[cubNp*e + i + j*cubNq] = JW;

        /* store second order geometric factors */
        dlong base = Nggeo*(cubNp*e + i + j*cubNq);
        cubggeo[base + G00ID] = JW*(rx*rx + ry*ry);
        cubggeo[base + G01ID] = JW*(rx*sx + ry*sy);
        cubggeo[base + G11ID] = JW*(sx*sx + sy*sy);
      }
    }
  }

  o_cubwJ = platform.malloc<dfloat>(cubwJ);
  o_cubggeo = platform.malloc<dfloat>(cubggeo);
}

} //namespace libp
