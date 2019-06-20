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

void BPOperator(BP_t *BP, dfloat lambda, occa::memory &o_q, occa::memory &o_Aq, const char *precision){

  mesh_t *mesh = BP->mesh;
  setupAide &options = BP->options;

  int enableGatherScatters = 1;
  int enableReductions = 1;
  int continuous = options.compareArgs("DISCRETIZATION", "CONTINUOUS");
  
  options.getArgs("DEBUG ENABLE REDUCTIONS", enableReductions);
  options.getArgs("DEBUG ENABLE OGS", enableGatherScatters);

  //  printf("generalOperator: gathers = %d, reductions = %d, cts = %d, serial = %d, ipdg = %d\n",
  //	 enableGatherScatters, enableReductions, continuous, serial, ipdg);
  
  dfloat *sendBuffer = BP->sendBuffer;
  dfloat *recvBuffer = BP->recvBuffer;

  dfloat alpha = 0., alphaG = 0.;
  dlong Nblock = BP->Nblock;
  dfloat *tmp = BP->tmp;
  occa::memory &o_tmp = BP->o_tmp;

  if(continuous){

    ogs_t *ogs = BP->ogs;

    occa::kernel &BP1Kernel = BP->BP1Kernel;
    occa::kernel &BP3Kernel = BP->BP3Kernel;
    occa::kernel &BP5Kernel = BP->BP5Kernel;
    
    if(mesh->NglobalGatherElements) {

      if(options.compareArgs("BENCHMARK", "BP1"))
	BP1Kernel(mesh->NglobalGatherElements, mesh->o_globalGatherElementList,
		  mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq);

      if(options.compareArgs("BENCHMARK", "BP3"))
	BP3Kernel(mesh->NglobalGatherElements, mesh->o_globalGatherElementList,
		  mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq);
      
      if(options.compareArgs("BENCHMARK", "BP5"))
	BP5Kernel(mesh->NglobalGatherElements, mesh->o_globalGatherElementList,
		  mesh->o_ggeo, mesh->o_D, lambda, o_q, o_Aq);
    }
    
    if(enableGatherScatters)
      ogsGatherScatterStart(o_Aq, ogsDfloat, ogsAdd, ogs);
    
    if(mesh->NlocalGatherElements){

      if(options.compareArgs("BENCHMARK", "BP1")){
	BP1Kernel(mesh->NlocalGatherElements, mesh->o_localGatherElementList,
		  mesh->o_cubggeo, mesh->o_cubInterp, o_q, o_Aq);
      }

      if(options.compareArgs("BENCHMARK", "BP3")){
	BP3Kernel(mesh->NlocalGatherElements, mesh->o_localGatherElementList,
      		  mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq);
      }
      
      if(options.compareArgs("BENCHMARK", "BP5")){
	BP5Kernel(mesh->NlocalGatherElements, mesh->o_localGatherElementList,
		  mesh->o_ggeo, mesh->o_D,lambda, o_q, o_Aq);
      }
    }
    
    // finalize gather using local and global contributions
    if(enableGatherScatters==1)
      ogsGatherScatterFinish(o_Aq, ogsDfloat, ogsAdd, ogs);

#if USE_NULL_BOOST==1
    if(BP->allNeumann) {
      
      BP->innerProductKernel(mesh->Nelements*mesh->Np, BP->o_invDegree, o_q, o_tmp);
      o_tmp.copyTo(tmp);
      
      for(dlong n=0;n<Nblock;++n)
        alpha += tmp[n];

      MPI_Allreduce(&alpha, &alphaG, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
      alphaG *= BP->allNeumannPenalty*BP->allNeumannScale*BP->allNeumannScale;

      mesh->addScalarKernel(mesh->Nelements*mesh->Np, alphaG, o_Aq);
    }
#endif
    
  }
}

