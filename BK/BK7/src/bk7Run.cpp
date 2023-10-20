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

#include "bk7.hpp"
#include "timer.hpp"

void bk7_t::Run(){

  //create occa buffers
  dlong offset = mesh.Np*mesh.Nelements;
  dlong cubatureOffset = mesh.cubNp*mesh.Nelements;
  dlong NUoffset = 0;
  
  deviceMemory<dfloat> o_NU   = platform.malloc<dfloat>(offset*Nfields);
  deviceMemory<dfloat> o_Ud   = platform.malloc<dfloat>(offset*Nfields);
  deviceMemory<dfloat> o_conv = platform.malloc<dfloat>(cubatureOffset*Nfields*Next); // interpolated convection velocity

  deviceMemory<dfloat> o_invLumpedMassMatrix = platform.malloc<dfloat>(offset);
  
  dfloat c0 = 1., c1 = 2., c2 = 3.;
  
  int Ntests = 50;

  for(int n=0;n<5;++n){ //warmup
    operatorKernel(mesh.NlocalGatherElements,
                   mesh.o_localGatherElementList,
                   mesh.o_cubD,
                   mesh.o_cubInterp,
		   offset,
		   cubatureOffset,
		   NUoffset,
		   o_invLumpedMassMatrix,
		   c0,
		   c1,
		   c2,
		   o_conv,
		   o_Ud,
		   o_NU);
  }

  timePoint_t start = GlobalPlatformTime(platform);
  for(int n=0;n<Ntests;++n){

    operatorKernel(mesh.NlocalGatherElements,
                   mesh.o_localGatherElementList,
                   mesh.o_cubD,
                   mesh.o_cubInterp,
		   offset,
		   cubatureOffset,
		   NUoffset,
		   o_invLumpedMassMatrix,
		   c0,
		   c1,
		   c2,
		   o_conv,
		   o_Ud,
		   o_NU);

  }
  
  timePoint_t end = GlobalPlatformTime(platform);
  double elapsedTime = ElapsedTime(start,end)/Ntests;

  int Np = mesh.Np, Nq = mesh.Nq, cubNq = mesh.cubNq, cubNp = mesh.cubNp;

  hlong Ndofs = ogs.NgatherGlobal;

  size_t Nbytes = 0;
  switch (mesh.elementType) {
  case mesh_t::TRIANGLES:
    break;
  case mesh_t::TETRAHEDRA:
    break;
  case mesh_t::QUADRILATERALS:
    break;
  case mesh_t::HEXAHEDRA:
    Nbytes = (sizeof(dlong) + /* element index */
	      cubNp*mesh.dim*Next*sizeof(dfloat) + /* extrap */
	      Np*sizeof(dfloat) + /* inv mass */
	      Np*mesh.dim*sizeof(dfloat) + /* velocity in */
	      Np*mesh.dim*sizeof(dfloat) /* NU out */
	      )*mesh.NelementsGlobal;
    break;
  }
  
  size_t Nflops=0;
  switch (mesh.elementType) {
  case mesh_t::TRIANGLES:
    break;
  case mesh_t::TETRAHEDRA:
    break;
  case mesh_t::QUADRILATERALS:
    break;
  case mesh_t::HEXAHEDRA:
    // interp + projection + 
    Nflops =(
	     mesh.dim*2*2*(cubNq*Nq*Nq*Nq + cubNq*cubNq*Nq*Nq + cubNq*cubNq*cubNq*Nq) +  /* interp + project */
	     mesh.dim*2*mesh.dim*(cubNq*cubNq*cubNq*cubNq) +  /* differentiation per field */
	     mesh.Np*mesh.dim
	     )*mesh.NelementsGlobal;
    break;
  }
  
  if (mesh.rank==0){
    std::string suffix("Element=");
    switch (mesh.elementType) {
    case mesh_t::TRIANGLES:
      suffix += "Tri";
      break;
    case mesh_t::TETRAHEDRA:
      suffix += "Tet";
      break;
    case mesh_t::QUADRILATERALS:
      suffix += "Quad";
      break;
    case mesh_t::HEXAHEDRA:
      suffix += "Hex";
      break;
    }
    printf("BK7: N=%2d, DOFs=" hlongFormat ", elapsed=%4.4f, time per DOF=%1.2e, avg BW (GB/s)=%6.1f, avg GFLOPs=%6.1f, DOFs/ranks*time=%1.2e, %s \n",
	   mesh.N,
	   Ndofs,
	   elapsedTime,
	   elapsedTime/(Ndofs),
	   Nbytes/(1.0e9 * elapsedTime),
	   Nflops/(1.0e9 * elapsedTime),
	   Ndofs/(mesh.size*elapsedTime),
	   suffix.c_str());
  
  }
}
