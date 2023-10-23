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
#include <unistd.h>

void bk7_t::Run(){

  Next = 2;
  
  //create occa buffers
  dlong offset = mesh.Np*mesh.Nelements;
  dlong cubatureOffset = mesh.cubNp*mesh.Nelements;
  dlong NUoffset = 0;

  memory<dfloat> h_NU(offset*Nfields);
  memory<dfloat> h_Ud(offset*Nfields);
  memory<dfloat> h_conv(cubatureOffset*Nfields*Next);
  memory<dfloat> h_invLumpedMassMatrix(offset);

  srand48(123456);
  for(int n=0;n<offset*Nfields;++n){
    h_NU[n] = drand48();
    h_Ud[n] = drand48();
  }
  for(int n=0;n<offset;++n){
    h_invLumpedMassMatrix[n] = 1.; //drand48();
  }

  for(int n=0;n<cubatureOffset*Nfields*Next;++n){
    h_conv[n] = drand48();
  }
  
  deviceMemory<dfloat> o_NU   = platform.malloc<dfloat>(offset*Nfields, h_NU);
  deviceMemory<dfloat> o_Ud   = platform.malloc<dfloat>(offset*Nfields, h_Ud);
  deviceMemory<dfloat> o_cubatureScratch = platform.malloc<dfloat>(cubatureOffset*Nfields*Next, h_conv);
  deviceMemory<dfloat> o_conv = platform.malloc<dfloat>(cubatureOffset*Nfields*Next, h_conv); // interpolated convection velocity

  deviceMemory<dfloat> o_invLumpedMassMatrix = platform.malloc<dfloat>(offset, h_invLumpedMassMatrix);

  // set kernel name suffix
  char *suffix;
  if(mesh.elementType==mesh_t::TRIANGLES)
    suffix = strdup("Tri2D");
  else if(mesh.elementType==mesh_t::QUADRILATERALS)
    suffix = strdup("Quad2D");
  else if(mesh.elementType==mesh_t::TETRAHEDRA)
    suffix = strdup("Tet3D");
  else //if(mesh.elementType==mesh_t::HEXAHEDRA)
    suffix = strdup("Hex3D");
  
  dfloat c0 = 1., c1 = 2., c2 = 3.;
  
  int minKnl = 0, skipKnl = 1, maxKnl = 21;
  for(int knl=minKnl;knl<=maxKnl;knl+=skipKnl){

    // OCCA build stuff
    properties_t kernelInfo = mesh.props; //copy base occa properties
    
    kernelInfo["defines/" "p_Next"]= Next;
    kernelInfo["defines/" "p_Nfields"]= Nfields;
    kernelInfo["defines/" "p_NVfields"]= mesh.dim;
    kernelInfo["defines/" "p_knl"]= knl;
    
    kernelInfo["includes"] += "constantInterpolationMatrices.h";
    kernelInfo["includes"] += "constantDifferentiationMatrices.h";
    
    char fileName[BUFSIZ], kernelName[BUFSIZ];
    
    // Ax kernel
    sprintf(fileName,  LIBP_DIR "/okl/bp7Ax%s.okl", suffix);
    sprintf(kernelName, "bp7Ax%s", suffix);
    
    occa::kernel opKernel = platform.buildKernel(fileName, kernelName,
						 kernelInfo);

    if(knl!=20){
      // checksum test
      opKernel(mesh.NlocalGatherElements,
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
	       o_NU,
	       o_cubatureScratch);    
    }
    else{
      for(int dim=0;dim<3;++dim){
	opKernel(dim,
		 mesh.NlocalGatherElements,
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
		 o_NU,
		 o_cubatureScratch);
      }

    }
    usleep(1000000);
      
    dfloat checksum = platform.linAlg().norm2(offset*Nfields, o_NU, mesh.comm);
    
    for(int n=0;n<3;++n){ //warmup

      if(knl!=20){
	opKernel(mesh.NlocalGatherElements,
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
		 o_NU,
		 o_cubatureScratch);

      }else{
	for(int dim=0;dim<3;++dim){
	  opKernel(dim,
		   mesh.NlocalGatherElements,
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
		   o_NU,
		   o_cubatureScratch);
	}
      }
    }
    
    int Nrepeats = 1;
    int Ntests = 10;
    
	
    for(int rep=0;rep<Nrepeats;++rep){
      
      timePoint_t start = GlobalPlatformTime(platform);

      for(int n=0;n<Ntests;++n){

	if(knl!=20)
	  opKernel(mesh.NlocalGatherElements,
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
		   o_NU,
		   o_cubatureScratch);
	else{
	  for(int dim=0;dim<3;++dim){
	    opKernel(dim,
		     mesh.NlocalGatherElements,
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
		     o_NU,
		     o_cubatureScratch);
	  }
	}

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
	printf("BK7: Kernel=%d, Rep=%d, N=%2d, cubNq=%2d, DOFs=" hlongFormat ", elapsed=%4.4e, time per DOF=%1.2e, avg BW (GB/s)=%6.1f, avg GFLOPs=%6.1f, DOFs/ranks*time=%1.2e, checksum=%15.14e %s \n",
	       knl,
	       rep,
	       mesh.N,
	       mesh.cubNq,
	       Ndofs,
	       elapsedTime,
	       elapsedTime/(Ndofs),
	       Nbytes/(1.0e9 * elapsedTime),
	       Nflops/(1.0e9 * elapsedTime),
	       Ndofs/(mesh.size*elapsedTime),
	       checksum,
	       suffix);
      }
    }
  }
}
