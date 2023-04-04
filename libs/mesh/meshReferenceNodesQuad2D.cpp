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

void mesh_t::ReferenceNodesQuad2D(){

  Nfp = N+1;
  Nq = (N+1);
  Np = (N+1)*(N+1);

  /* Nodal Data */
  NodesQuad2D(N, r, s);
  FaceNodesQuad2D(N, r, s, faceNodes);
  VertexNodesQuad2D(N, r, s, vertexNodes);

  //GLL quadrature
  JacobiGLL(N, gllz, gllw);

  // D matrix
  Dmatrix1D(N, gllz, gllz, D);
  o_D = platform.malloc<dfloat>(D);

  /* Plotting data */
  plotN = N;
  plotNelements = 2*plotN*plotN;
  plotNverts = 3;
  EquispacedEToVQuad2D(plotN, plotEToV);

  props["defines/" "p_dim"]= dim;
  props["defines/" "p_N"]= N;
  props["defines/" "p_Nq"]= Nq;
  props["defines/" "p_Np"]= Np;
  props["defines/" "p_Nfp"]= Nfp;
  props["defines/" "p_Nfaces"]= Nfaces;
  props["defines/" "p_NfacesNfp"]= Nfp*Nfaces;
}

} //namespace libp
