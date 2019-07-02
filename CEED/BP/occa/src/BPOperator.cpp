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

void runBPKernel(BP_t *BP,  dfloat lambda,
		 hlong Nelements, occa::memory &o_elementList,
		 occa::memory &o_q, occa::memory &o_Aq){

  if(Nelements){
    setupAide &options = BP->options;
    mesh_t *mesh = BP->mesh;

    occa::kernel &BPKernel = BP->BPKernel;
    
    int combineDot = 0;
    combineDot = options.compareArgs("COMBINE DOT PRODUCT", "TRUE");
    
    if(!combineDot){
      switch(BP->BPid){
      case 1:
	BPKernel(Nelements, o_elementList, mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq);
	break;
      case 3:
	BPKernel(Nelements, o_elementList, mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq);	
	break;
      case 5:
	BPKernel(Nelements, o_elementList, mesh->o_ggeo, mesh->o_D, lambda, o_q, o_Aq);
	break;
      }
    }
    
    if(combineDot){
      switch(BP->BPid){
      case 1:  
	BPKernel(Nelements, o_elementList, mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq,  BP->o_tmpAtomic);
	break;
      case 3:
	BPKernel(Nelements, o_elementList, mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq,
		 BP->o_tmpAtomic);
	break;
      case 5:
	BPKernel(Nelements, o_elementList, mesh->o_ggeo, mesh->o_D, lambda, o_q, o_Aq, BP->o_tmpAtomic);
	break;
      }
    }
  }
}

dfloat BPOperator(BP_t *BP, dfloat lambda, occa::memory &o_q, occa::memory &o_Aq, const char *precision){

  mesh_t *mesh = BP->mesh;
  setupAide &options = BP->options;
  ogs_t *ogs = BP->ogs;

  int combineDot = 0;
  combineDot = options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

  if(combineDot){
    BP->o_zeroAtomic.copyTo(BP->o_tmpAtomic);
  }

  int BPid = BP->BPid;

  if(mesh->NglobalGatherElements) {
    runBPKernel(BP, lambda, mesh->NglobalGatherElements, mesh->o_globalGatherElementList, o_q, o_Aq);
  }

  ogsGatherScatterStart(o_Aq, ogsDfloat, ogsAdd, ogs);
    
  if(mesh->NlocalGatherElements){
    runBPKernel(BP, lambda, mesh->NlocalGatherElements, mesh->o_localGatherElementList, o_q, o_Aq);
  }
  
  // finalize gather using local and global contributions
  ogsGatherScatterFinish(o_Aq, ogsDfloat, ogsAdd, ogs);
  
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


