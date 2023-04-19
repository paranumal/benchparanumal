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

void mesh_t::ReferenceNodesHex3D(){

  Nq = N+1;
  Nfp = Nq*Nq;
  Np = Nq*Nq*Nq;

  /* Nodal Data */
  NodesHex3D(N, r, s, t);
  FaceNodesHex3D(N, r, s, t, faceNodes);
  VertexNodesHex3D(N, r, s, t, vertexNodes);

  //GLL quadrature
  JacobiGLL(N, gllz, gllw);
  o_gllw = platform.malloc<dfloat>(gllw);

  memory<dfloat> V;
  Vandermonde1D(N, gllz, V);

  //Mass matrix
  MassMatrix1D(Nq, V, MM);
  o_MM = platform.malloc<dfloat>(MM); //MM is symmetric

  // D matrix
  Dmatrix1D(N, gllz, gllz, D);
  o_D = platform.malloc<dfloat>(D);

  /* Plotting data */
  plotN = N;
  plotNelements = 6*plotN*plotN*plotN;
  plotNverts = 4;
  EquispacedEToVHex3D(plotN, plotEToV);

  props["defines/" "p_dim"]= dim;
  props["defines/" "p_N"]= N;
  props["defines/" "p_Nq"]= Nq;
  props["defines/" "p_Np"]= Np;
  props["defines/" "p_Nfp"]= Nfp;
  props["defines/" "p_Nfaces"]= Nfaces;
  props["defines/" "p_NfacesNfp"]= Nfp*Nfaces;
}

} //namespace libp
