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

dfloat BPOperator(BP_t *BP, int mode, dfloat lambda, occa::memory &o_q, occa::memory &o_Aq, const char *precision){

  mesh_t *mesh = BP->mesh;
  setupAide &options = BP->options;
  ogs_t *ogs = BP->ogs;

  int combineDot = 0;
  combineDot = options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

  if(combineDot){
    BP->o_zeroAtomic.copyTo(BP->o_tmpAtomic);
  }
  
  if(mesh->NglobalGatherElements) {

    if(options.compareArgs("BENCHMARK", "BP1")){

      if(!combineDot){
	occa::kernel &BP1Kernel = BP->BP1Kernels[mode];
	
	BP1Kernel(mesh->NglobalGatherElements, mesh->o_globalGatherElementList,
		  mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq);

      }else{
	occa::kernel &BP1DotKernel = BP->BP1DotKernels[mode];
	
	BP1DotKernel(mesh->NglobalGatherElements, mesh->o_globalGatherElementList,
		     mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq,
		     BP->o_tmpAtomic);
      }
    }
    
    if(options.compareArgs("BENCHMARK", "BP3")){

      if(!combineDot){
	occa::kernel &BP3Kernel = BP->BP3Kernels[mode];
	
	BP3Kernel(mesh->NglobalGatherElements, mesh->o_globalGatherElementList,
		  mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq);
      }else{
	occa::kernel &BP3DotKernel = BP->BP3DotKernels[mode];
	
	BP3DotKernel(mesh->NglobalGatherElements, mesh->o_globalGatherElementList,
		     mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq,
		     BP->o_tmpAtomic);
      }
    }    
    
    if(options.compareArgs("BENCHMARK", "BP5")){

      if(!combineDot){
	occa::kernel &BP5Kernel = BP->BP3Kernels[mode];      

	BP5Kernel(mesh->NglobalGatherElements, mesh->o_globalGatherElementList,
		  mesh->o_ggeo, mesh->o_D, lambda, o_q, o_Aq);
      }else{
	occa::kernel &BP5DotKernel = BP->BP5DotKernels[mode];

	BP5DotKernel(mesh->NglobalGatherElements, mesh->o_globalGatherElementList,
		     mesh->o_ggeo, mesh->o_D, lambda, o_q, o_Aq, BP->o_tmpAtomic);
      }
    }
  }

  ogsGatherScatterStart(o_Aq, ogsDfloat, ogsAdd, ogs);
    
  if(mesh->NlocalGatherElements){
      
    if(options.compareArgs("BENCHMARK", "BP1")){

      if(!combineDot){
	occa::kernel &BP1Kernel = BP->BP1Kernels[mode];
	
	BP1Kernel(mesh->NlocalGatherElements, mesh->o_localGatherElementList,
		  mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq);
      }else{
	occa::kernel &BP1DotKernel = BP->BP1DotKernels[mode];
	
	BP1DotKernel(mesh->NlocalGatherElements, mesh->o_localGatherElementList,
		     mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq,
		     BP->o_tmpAtomic);
      }
      
    }
    
    if(options.compareArgs("BENCHMARK", "BP3")){

      if(!combineDot){
	occa::kernel &BP3Kernel = BP->BP3Kernels[mode];

	BP3Kernel(mesh->NlocalGatherElements, mesh->o_localGatherElementList,
		  mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq);
      }else{
	occa::kernel &BP3DotKernel = BP->BP3DotKernels[mode];

	BP3DotKernel(mesh->NlocalGatherElements, mesh->o_localGatherElementList,
		     mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq,
		     BP->o_tmpAtomic);
      }
    }
      
    if(options.compareArgs("BENCHMARK", "BP5")){
      if(!combineDot){
	occa::kernel &BP5Kernel = BP->BP5Kernels[mode];
	
	BP5Kernel(mesh->NlocalGatherElements, mesh->o_localGatherElementList,
		  mesh->o_ggeo, mesh->o_D,lambda, o_q, o_Aq);
      }else{
	occa::kernel &BP5DotKernel = BP->BP5DotKernels[mode];
	
	BP5DotKernel(mesh->NlocalGatherElements, mesh->o_localGatherElementList,
		     mesh->o_ggeo, mesh->o_D,lambda, o_q, o_Aq,
		     BP->o_tmpAtomic);
      }
    }
  }
  
  // finalize gather using local and global contributions
  ogsGatherScatterFinish(o_Aq, ogsDfloat, ogsAdd, ogs);
    
  dfloat pAp = 0;

  if(!combineDot){
    pAp =  BPWeightedInnerProduct(BP, BP->o_invDegree, o_q, o_Aq);
  }else{
    BP->o_tmpAtomic.copyTo(BP->tmpAtomic);
    
    // WATCH OUT FOR VECTOR HERE
    MPI_Allreduce(BP->tmpAtomic, &pAp, BP->Nfields, MPI_DFLOAT, MPI_SUM, mesh->comm);
  }
  
  return pAp;
}


