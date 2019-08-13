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

void runBPGlobalKernel(BP_t *BP,  dfloat lambda, dfloat mu,
		       hlong Nelements, occa::memory &o_elementList,
		       occa::memory &o_localizedIds,
		       occa::memory &o_q, occa::memory &o_Aq){
  if(Nelements){
    setupAide &options = BP->options;
    mesh_t *mesh = BP->mesh;

    occa::kernel &BPKernel = BP->BPKernelGlobal[BP->BPid];

    BPKernel(Nelements, o_elementList, mesh->o_localizedIds,
	     mesh->o_cubggeo, mesh->o_cubD, mesh->o_cubInterp, lambda, o_q, o_Aq);
  }
}

dfloat BPOperatorGlobal(BP_t *BP, dfloat lambda, dfloat mu, occa::memory &o_q, occa::memory &o_Aq,
			const char *precision, occa::streamTag *start, occa::streamTag *end){

  mesh_t *mesh = BP->mesh;
  setupAide &options = BP->options;
  ogs_t *ogs = BP->ogs;

  dlong Ndof = mesh->Nlocalized;
  
  dlong offset = mesh->Np*(mesh->Nelements+mesh->totalHaloPairs);
  
  int BPid = BP->BPid;

  BP->vecZeroKernel(mesh->Nlocalized*BP->Nfields, o_Aq);

  // localized ids needs to give global ids relative to local storage
  runBPGlobalKernel(BP, lambda, mu, mesh->NglobalGatherElements, mesh->o_globalGatherElementList,
		    mesh->o_localizedIds, o_q, o_Aq);
  
#if 0
  if(BP->Nfields==1)
    ogsGatherScatterStart(o_Aq, ogsDfloat, ogsAdd, ogs);
  else
    ogsGatherScatterManyStart(o_Aq, BP->Nfields, offset, ogsDfloat, ogsAdd, ogs);
#endif
  
  if(start)
    *start = BP->mesh->device.tagStream();

  // really need to transfer to o_someTmp then copy Nlocalized to o_Aq
  runBPGlobalKernel(BP, lambda, mu, mesh->NlocalGatherElements, mesh->o_localGatherElementList,
		    mesh->o_localizedIds, o_q, o_Aq);
  
  if(end)
    *end = BP->mesh->device.tagStream();
  
  // finalize gather using local and global contributions
#if 0
  if(BP->Nfields==1)
    ogsGatherScatterFinish(o_Aq, ogsDfloat, ogsAdd, ogs);
  else
    ogsGatherScatterManyFinish(o_Aq, BP->Nfields, offset, ogsDfloat, ogsAdd, ogs);
#endif
  
  //  dfloat pAp = BPInnerProduct(BP, Ndof, Ndof, o_q, o_Aq);
  dfloat pAp = BPAtomicInnerProduct(BP, Ndof, o_q, o_Aq);
  
  return pAp;
}

