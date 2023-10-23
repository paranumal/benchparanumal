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

void mesh_t::CubatureSetupHex3D(){

  /* Quadrature data */
  int inCubNq = -1;
  settings.getSetting("CUBATURE SIZE", inCubNq);
  if(inCubNq==-1)
    cubNq = Nq;
  else
    cubNq = inCubNq;
  cubN = cubNq-1;

  cubNp = cubNq*cubNq*cubNq;
  cubNfp = cubNq*cubNq;

  // cubN+1 point Gauss-Legendre quadrature
  JacobiGQ(0, 0, cubN, cubr, cubw);

  // GLL to GL interpolation matrix
  InterpolationMatrix1D(N, gllz, cubr, cubInterp);

  //cubature project cubProject = cubInterp^T
  cubProject.malloc(cubNq*Nq);
  linAlg_t::matrixTranspose(cubNq, Nq, cubInterp, Nq, cubProject, cubNq);

  o_cubInterp  = platform.malloc<dfloat>(cubInterp);
  o_cubProject = platform.malloc<dfloat>(cubProject);

  if (settings.compareSetting("AFFINE MESH", "TRUE")) {
    // derivates matrix for affine elements: differentiate on modal basis
    // cubD = V^-1 * D * V
    //      = V^-1 * Vr
    memory<dfloat> Vr;
    GradVandermonde1D(N, gllz, Vr);

    Vandermonde1D(N, gllz, invV);
    linAlg_t::matrixInverse(Nq, invV);

    cubD.malloc(Nq*Nq);
    for(int n=0;n<Nq;++n){
      for(int m=0;m<Nq;++m){
        dfloat cubDnm = 0;
        for(int k=0;k<Nq;++k){
          cubDnm += invV[n*Nq+k]*Vr[k*Nq+m];
        }
        cubD[n*Nq + m] = cubDnm;
      }
    }

    o_cubD = platform.malloc<dfloat>(cubD);
    o_invV = platform.malloc<dfloat>(invV);

  } else {
    //cubature derivates matrix, cubD: differentiate on cubature nodes
    Dmatrix1D(cubN, cubr, cubr, cubD);
    o_cubD = platform.malloc<dfloat>(cubD);
  }

  // add compile time constants to kernels
  props["defines/" "p_cubN"]= cubN;
  props["defines/" "p_cubNq"]= cubNq;
  props["defines/" "p_cubNp"]= cubNp;
  props["defines/" "p_cubNfp"]= cubNfp;


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
    memory<dfloat> xte(Np);
    memory<dfloat> yre(Np);
    memory<dfloat> yse(Np);
    memory<dfloat> yte(Np);
    memory<dfloat> zre(Np);
    memory<dfloat> zse(Np);
    memory<dfloat> zte(Np);

    memory<dfloat> xre1(Nq*Nq*cubNq);
    memory<dfloat> xse1(Nq*Nq*cubNq);
    memory<dfloat> xte1(Nq*Nq*cubNq);
    memory<dfloat> yre1(Nq*Nq*cubNq);
    memory<dfloat> yse1(Nq*Nq*cubNq);
    memory<dfloat> yte1(Nq*Nq*cubNq);
    memory<dfloat> zre1(Nq*Nq*cubNq);
    memory<dfloat> zse1(Nq*Nq*cubNq);
    memory<dfloat> zte1(Nq*Nq*cubNq);

    memory<dfloat> xre2(Nq*cubNq*cubNq);
    memory<dfloat> xse2(Nq*cubNq*cubNq);
    memory<dfloat> xte2(Nq*cubNq*cubNq);
    memory<dfloat> yre2(Nq*cubNq*cubNq);
    memory<dfloat> yse2(Nq*cubNq*cubNq);
    memory<dfloat> yte2(Nq*cubNq*cubNq);
    memory<dfloat> zre2(Nq*cubNq*cubNq);
    memory<dfloat> zse2(Nq*cubNq*cubNq);
    memory<dfloat> zte2(Nq*cubNq*cubNq);

    //geometric data for quadrature
    for(dlong e=0;e<Nelements;++e){ /* for each element */
      for(int k=0;k<Nq;++k){
        for(int j=0;j<Nq;++j){
          for(int i=0;i<Nq;++i){
            int n = i + j*Nq + k*Nq*Nq;

            //differentiate physical coordinates
            xre[n] = 0; xse[n] = 0; xte[n] = 0;
            yre[n] = 0; yse[n] = 0; yte[n] = 0;
            zre[n] = 0; zse[n] = 0; zte[n] = 0;

            for(int m=0;m<Nq;++m){
              int idr = e*Np + k*Nq*Nq + j*Nq + m;
              int ids = e*Np + k*Nq*Nq + m*Nq + i;
              int idt = e*Np + m*Nq*Nq + j*Nq + i;
              xre[n] += D[i*Nq+m]*x[idr];
              xse[n] += D[j*Nq+m]*x[ids];
              xte[n] += D[k*Nq+m]*x[idt];
              yre[n] += D[i*Nq+m]*y[idr];
              yse[n] += D[j*Nq+m]*y[ids];
              yte[n] += D[k*Nq+m]*y[idt];
              zre[n] += D[i*Nq+m]*z[idr];
              zse[n] += D[j*Nq+m]*z[ids];
              zte[n] += D[k*Nq+m]*z[idt];
            }
          }
        }
      }

      //interpolate derivaties to cubature
      for(int k=0;k<Nq;++k){
        for(int j=0;j<Nq;++j){
          for(int i=0;i<cubNq;++i){
            dlong id = k*Nq*cubNq+j*cubNq+i;
            xre1[id] = 0.0; xse1[id] = 0.0;  xte1[id] = 0.0;
            yre1[id] = 0.0; yse1[id] = 0.0;  yte1[id] = 0.0;
            zre1[id] = 0.0; zse1[id] = 0.0;  zte1[id] = 0.0;
            for(int n=0;n<Nq;++n){
              dlong idn = k*Nq*Nq+j*Nq+n;
              xre1[id] += cubInterp[i*Nq + n]*xre[idn];
              xse1[id] += cubInterp[i*Nq + n]*xse[idn];
              xte1[id] += cubInterp[i*Nq + n]*xte[idn];
              yre1[id] += cubInterp[i*Nq + n]*yre[idn];
              yse1[id] += cubInterp[i*Nq + n]*yse[idn];
              yte1[id] += cubInterp[i*Nq + n]*yte[idn];
              zre1[id] += cubInterp[i*Nq + n]*zre[idn];
              zse1[id] += cubInterp[i*Nq + n]*zse[idn];
              zte1[id] += cubInterp[i*Nq + n]*zte[idn];
            }
          }
        }
      }

      for(int k=0;k<Nq;++k){
        for(int j=0;j<cubNq;++j){
          for(int i=0;i<cubNq;++i){
            dlong id = k*cubNq*cubNq+j*cubNq+i;
            xre2[id] = 0.0; xse2[id] = 0.0;  xte2[id] = 0.0;
            yre2[id] = 0.0; yse2[id] = 0.0;  yte2[id] = 0.0;
            zre2[id] = 0.0; zse2[id] = 0.0;  zte2[id] = 0.0;
            for(int n=0;n<Nq;++n){
              dlong idn = k*Nq*cubNq+n*cubNq+i;
              xre2[id] += cubInterp[j*Nq + n]*xre1[idn];
              xse2[id] += cubInterp[j*Nq + n]*xse1[idn];
              xte2[id] += cubInterp[j*Nq + n]*xte1[idn];
              yre2[id] += cubInterp[j*Nq + n]*yre1[idn];
              yse2[id] += cubInterp[j*Nq + n]*yse1[idn];
              yte2[id] += cubInterp[j*Nq + n]*yte1[idn];
              zre2[id] += cubInterp[j*Nq + n]*zre1[idn];
              zse2[id] += cubInterp[j*Nq + n]*zse1[idn];
              zte2[id] += cubInterp[j*Nq + n]*zte1[idn];
            }
          }
        }
      }

      for(int k=0;k<cubNq;++k){
        for(int j=0;j<cubNq;++j){
          for(int i=0;i<cubNq;++i){
            dfloat xr = 0.0, xs = 0.0, xt = 0.0;
            dfloat yr = 0.0, ys = 0.0, yt = 0.0;
            dfloat zr = 0.0, zs = 0.0, zt = 0.0;
            for(int n=0;n<Nq;++n){
              dlong idn = n*cubNq*cubNq+j*cubNq+i;
              xr += cubInterp[k*Nq + n]*xre2[idn];
              xs += cubInterp[k*Nq + n]*xse2[idn];
              xt += cubInterp[k*Nq + n]*xte2[idn];
              yr += cubInterp[k*Nq + n]*yre2[idn];
              ys += cubInterp[k*Nq + n]*yse2[idn];
              yt += cubInterp[k*Nq + n]*yte2[idn];
              zr += cubInterp[k*Nq + n]*zre2[idn];
              zs += cubInterp[k*Nq + n]*zse2[idn];
              zt += cubInterp[k*Nq + n]*zte2[idn];
            }

            /* compute geometric factors for affine coordinate transform*/
            dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);

            LIBP_ABORT("Negative J found at element " << e,
                       J<1e-8);

            dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
            dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
            dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;

            dfloat JW = J*cubw[i]*cubw[j]*cubw[k];

            cubwJ[cubNp*e + i + j*cubNq + k*cubNq*cubNq] = JW;

            /* store second order geometric factors */
            dlong base = Nggeo*(cubNp*e + i + j*cubNq + k*cubNq*cubNq);
            cubggeo[base + G00ID] = JW*(rx*rx + ry*ry + rz*rz);
            cubggeo[base + G01ID] = JW*(rx*sx + ry*sy + rz*sz);
            cubggeo[base + G02ID] = JW*(rx*tx + ry*ty + rz*tz);
            cubggeo[base + G11ID] = JW*(sx*sx + sy*sy + sz*sz);
            cubggeo[base + G12ID] = JW*(sx*tx + sy*ty + sz*tz);
            cubggeo[base + G22ID] = JW*(tx*tx + ty*ty + tz*tz);
          }
        }
      }
    }
    o_cubwJ = platform.malloc<dfloat>(cubwJ);
    o_cubggeo = platform.malloc<dfloat>(cubggeo);
  }

}

} //namespace libp
