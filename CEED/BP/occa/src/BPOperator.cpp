/*

  The MIT License (MIT)

  Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#include "BP.hpp"

void runBPKernel(BP_t *BP,  dfloat lambda, dfloat mu,
		 hlong Nelements, occa::memory &o_elementList,
		 occa::memory &o_q, occa::memory &o_Aq){

  if(Nelements){
    setupAide &options = BP->options;
    mesh_t *mesh = BP->mesh;

    occa::kernel &BPKernel = BP->BPKernel[BP->BPid];
    
    int combineDot = 0;
    combineDot = options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

    dlong offset = mesh->Np*mesh->Nelements; // TW: check this
    
    if(!combineDot){
      switch(BP->BPid){
      case 1:
	BPKernel(Nelements, o_elementList, mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq);
	break;
      case 2:
	BPKernel(Nelements, o_elementList, offset, mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq);
	break;
      case 3:
	BPKernel(Nelements, o_elementList, mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq);	
	break;
      case 4:
	BPKernel(Nelements, o_elementList, offset, mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq);	
	break;
      case 5:

	mesh->device.finish();
#if USE_CUDA_NATIVE==1
	printf("ENTERING NATIVE CUDA KERNEL\n");
	BK5(Nelements,
	    mesh->Nq,
	    lambda,
	    (dfloat*) mesh->o_ggeo.ptr(),
	    BP->c_DofToDofD,
	    BP->c_oddDofToDofD,
	    BP->c_evenDofToDofD,
	    (dfloat*) o_q.ptr(),
	    (dfloat*) o_Aq.ptr(),
	    BP->knlId);
	printf("LEAVING NATIVE CUDA KERNEL\n");
#else
	BPKernel(Nelements, o_elementList, mesh->o_ggeo, mesh->o_D, lambda, o_q, o_Aq);	
#endif

	
	break;
      case 6:
	BPKernel(Nelements, o_elementList, offset, mesh->o_ggeo, mesh->o_D, lambda, o_q, o_Aq);
	break;
      case 9:
	BPKernel(Nelements, o_elementList, offset, mesh->o_vgeo, mesh->o_D, mesh->o_filterMatrix, lambda, mu, o_q, o_Aq);
	break;
#if 0
      case 10:
	BPKernel(Nelements, o_elementList, mesh->o_cubggeo, mesh->o_cubInterpTet, o_q, o_Aq);
	break;
#endif
      }

    }
    
    if(combineDot){
      switch(BP->BPid){
      case 1:  
	BPKernel(Nelements, o_elementList, mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq,  BP->o_tmpAtomic);
	break;
      case 2:
	BPKernel(Nelements, o_elementList, offset, mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq, BP->o_tmpAtomic);
	break;
      case 3:
	BPKernel(Nelements, o_elementList, mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq,
		 BP->o_tmpAtomic);
	break;
      case 4:
	BPKernel(Nelements, o_elementList, offset, mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq,
		 BP->o_tmpAtomic);
	break;
      case 5:
	BPKernel(Nelements, o_elementList, mesh->o_ggeo, mesh->o_D, lambda, o_q, o_Aq, BP->o_tmpAtomic);
	break;
      case 6:
	BPKernel(Nelements, o_elementList, offset, mesh->o_ggeo, mesh->o_D, lambda, o_q, o_Aq, BP->o_tmpAtomic);
	break;
      case 9:
	BPKernel(Nelements, o_elementList, offset, mesh->o_vgeo, mesh->o_D, mesh->o_filterMatrix, lambda, mu, o_q, o_Aq, BP->o_tmpAtomic);
	break;
      }
    }
  }
}

dfloat BPOperator(BP_t *BP, dfloat lambda, dfloat mu, occa::memory &o_q, occa::memory &o_Aq,
		  const char *precision, occa::streamTag *start, occa::streamTag *end){

  mesh_t *mesh = BP->mesh;
  setupAide &options = BP->options;
  ogs_t *ogs = BP->ogs;

  int combineDot = 0;
  combineDot = options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

  if(combineDot)
    BP->o_zeroAtomic.copyTo(BP->o_tmpAtomic);

  dlong offset = mesh->Np*(mesh->Nelements+mesh->totalHaloPairs);
  
  int BPid = BP->BPid;
  
  runBPKernel(BP, lambda, mu, mesh->NglobalGatherElements, mesh->o_globalGatherElementList, o_q, o_Aq);
  
  if(BP->Nfields==1)
    ogsGatherScatterStart(o_Aq, ogsDfloat, ogsAdd, ogs);
  else
    ogsGatherScatterManyStart(o_Aq, BP->Nfields, offset, ogsDfloat, ogsAdd, ogs);

  if(start)
    *start = BP->mesh->device.tagStream();
  
  runBPKernel(BP, lambda, mu, mesh->NlocalGatherElements, mesh->o_localGatherElementList, o_q, o_Aq);

  if(end)
    *end = BP->mesh->device.tagStream();
  
  // finalize gather using local and global contributions
#if 1
  if(BP->Nfields==1)
    ogsGatherScatterFinish(o_Aq, ogsDfloat, ogsAdd, ogs);
  else
    ogsGatherScatterManyFinish(o_Aq, BP->Nfields, offset, ogsDfloat, ogsAdd, ogs);
#endif
  
  dfloat pAp = 0;

  if(!combineDot){
    // switch internal based on Nfields
    pAp =  BPWeightedInnerProduct(BP, BP->o_invDegree, o_q, o_Aq);
  }else{
    BP->o_tmpAtomic.copyTo(BP->tmpAtomic);
    
    MPI_Allreduce(BP->tmpAtomic, &pAp, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
  }
  
  return pAp;
}


