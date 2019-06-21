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

#include "omp.h"
#include <unistd.h>
#include "BP.hpp"

void reportMemoryUsage(occa::device &device, const char *mess);

BP_t *BPSetup(mesh_t *mesh, dfloat lambda, occa::properties &kernelInfo, setupAide options){

  BP_t *BP = (BP_t*) calloc(1, sizeof(BP_t));

  options.getArgs("MESH DIMENSION", BP->dim);
  options.getArgs("ELEMENT TYPE", BP->elementType);
  BP->mesh = mesh;
  BP->options = options;

  BP->Nfields = 1;
  
  // defaults for conjugate gradient
  int enableGatherScatters = 1;
  int enableReductions = 1; 
  int flexible = 1; 
  int verbose = 0;
  
  int serial = options.compareArgs("THREAD MODEL", "Serial");

  int continuous = options.compareArgs("DISCRETIZATION", "CONTINUOUS");
  int ipdg = options.compareArgs("DISCRETIZATION", "IPDG");

  options.getArgs("DEBUG ENABLE REDUCTIONS", enableReductions);
  options.getArgs("DEBUG ENABLE OGS", enableGatherScatters);
  
  flexible = options.compareArgs("KRYLOV SOLVER", "FLEXIBLE");
  verbose  = options.compareArgs("VERBOSE", "TRUE");

  if(mesh->rank==0 && verbose==1){
    printf("CG OPTIONS: enableReductions=%d, enableGatherScatters=%d, flexible=%d, verbose=%d, ipdg=%d, continuous=%d, serial=%d\n",
	   enableGatherScatters, 
	   enableReductions,
	   flexible,
	   verbose,
	   ipdg,
	   continuous,
	   serial);
  }

  if (mesh->rank==0)
    reportMemoryUsage(mesh->device, "after occa setup");

  BPSolveSetup(BP, lambda, kernelInfo);

  dlong Nall = mesh->Np*(mesh->Nelements+mesh->totalHaloPairs);
  BP->r   = (dfloat*) calloc(Nall,   sizeof(dfloat));
  BP->x   = (dfloat*) calloc(Nall,   sizeof(dfloat));
  BP->q   = (dfloat*) calloc(Nall,   sizeof(dfloat));

  // load forcing into r
  int BP1 = options.compareArgs("BENCHMARK", "BP1");
  int BP3 = options.compareArgs("BENCHMARK", "BP3");
  int BP5 = options.compareArgs("BENCHMARK", "BP5");

  for(dlong e=0;e<mesh->Nelements;++e){
    for(int n=0;n<mesh->Np;++n){

      dfloat JW = mesh->ggeo[mesh->Np*(e*mesh->Nggeo + GWJID) + n];

      dlong id = n+e*mesh->Np;
      dfloat xn = mesh->x[id];
      dfloat yn = mesh->y[id];
      dfloat zn = mesh->z[id];
      
      dfloat mode = 1;

     // mass projection rhs
      if(BP1)
	BP->r[id] =
	  JW*cos(mode*M_PI*xn)*cos(mode*M_PI*yn)*cos(mode*M_PI*zn);
								     
      // stiffness solve rhs
      if(BP3 || BP5)
	BP->r[id] =
	  JW*(3*mode*mode*M_PI*M_PI+lambda)*cos(mode*M_PI*xn)*cos(mode*M_PI*yn)*cos(mode*M_PI*zn);
      
      BP->x[id] = 0;
    }
  }

  //copy to occa buffers
  BP->o_r   = mesh->device.malloc(Nall*sizeof(dfloat), BP->r);
  BP->o_x   = mesh->device.malloc(Nall*sizeof(dfloat), BP->x);
  
  char *suffix = strdup("Hex3D");
  
  // gather-scatter
  if(options.compareArgs("DISCRETIZATION","CONTINUOUS")){
    ogsGatherScatter(BP->o_r, ogsDfloat, ogsAdd, mesh->ogs);
  }
  
  return BP;
}


void BPSolveSetup(BP_t *BP, dfloat lambda, occa::properties &kernelInfo){

  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;

  dlong Ntotal = mesh->Np*mesh->Nelements;

  dlong Nhalo = mesh->Np*mesh->totalHaloPairs;
  dlong Nall   = Ntotal + Nhalo;

  dlong Nblock  = mymax(1,(Ntotal+blockSize-1)/blockSize);
  dlong Nblock2 = mymax(1,(Nblock+blockSize-1)/blockSize);

  dlong NthreadsUpdatePCG = 256;
  dlong NblocksUpdatePCG = mymin((Ntotal+NthreadsUpdatePCG-1)/NthreadsUpdatePCG, 160);
 
  BP->NthreadsUpdatePCG = NthreadsUpdatePCG;
  BP->NblocksUpdatePCG = NblocksUpdatePCG;
  
  //tau
  BP->tau = 2.0*(mesh->N+1)*(mesh->N+3);

  BP->p   = (dfloat*) calloc(Nall,   sizeof(dfloat));
  BP->z   = (dfloat*) calloc(Nall,   sizeof(dfloat));
  BP->Ax  = (dfloat*) calloc(Nall,   sizeof(dfloat));
  BP->Ap  = (dfloat*) calloc(Nall,   sizeof(dfloat));
  BP->tmp = (dfloat*) calloc(Nblock, sizeof(dfloat));

  BP->o_p   = mesh->device.malloc(Nall*sizeof(dfloat), BP->p);
  BP->o_rtmp= mesh->device.malloc(Nall*sizeof(dfloat), BP->p);
  BP->o_z   = mesh->device.malloc(Nall*sizeof(dfloat), BP->z);

  BP->o_res = mesh->device.malloc(Nall*sizeof(dfloat), BP->z);
  BP->o_Sres = mesh->device.malloc(Nall*sizeof(dfloat), BP->z);
  BP->o_Ax  = mesh->device.malloc(Nall*sizeof(dfloat), BP->p);
  BP->o_Ap  = mesh->device.malloc(Nall*sizeof(dfloat), BP->Ap);
  BP->o_tmp = mesh->device.malloc(Nblock*sizeof(dfloat), BP->tmp);
  BP->o_tmp2 = mesh->device.malloc(Nblock2*sizeof(dfloat), BP->tmp);

  BP->tmpNormr = (dfloat*) calloc(BP->NblocksUpdatePCG,sizeof(dfloat));
  BP->o_tmpNormr = mesh->device.malloc(BP->NblocksUpdatePCG*sizeof(dfloat), BP->tmpNormr);

  BP->tmpAtomic = (dfloat*) calloc(BP->Nfields,sizeof(dfloat));
  BP->o_tmpAtomic = mesh->device.malloc(BP->Nfields*sizeof(dfloat), BP->tmpAtomic);
  BP->o_zeroAtomic = mesh->device.malloc(BP->Nfields*sizeof(dfloat), BP->tmpAtomic);
  
  //setup async halo stream
  BP->defaultStream = mesh->defaultStream;
  BP->dataStream = mesh->dataStream;

  dlong Nbytes = mesh->totalHaloPairs*mesh->Np*sizeof(dfloat);
  if(Nbytes>0){
    BP->sendBuffer = (dfloat*) occaHostMallocPinned(mesh->device, Nbytes, NULL, BP->o_sendBuffer, BP->h_sendBuffer);
    BP->recvBuffer = (dfloat*) occaHostMallocPinned(mesh->device, Nbytes, NULL, BP->o_recvBuffer, BP->h_recvBuffer);
  }else{
    BP->sendBuffer = NULL;
    BP->recvBuffer = NULL;
  }
  mesh->device.setStream(BP->defaultStream);

  BP->type = strdup(dfloatString);

  BP->Nblock = Nblock;
  BP->Nblock2 = Nblock2;

  // count total number of elements
  hlong NelementsLocal = mesh->Nelements;
  hlong NelementsGlobal = 0;

  MPI_Allreduce(&NelementsLocal, &NelementsGlobal, 1, MPI_HLONG, MPI_SUM, mesh->comm);

  BP->NelementsGlobal = NelementsGlobal;

  //check all the bounaries for a Dirichlet
  bool allNeumann = (lambda==0) ? true :false;
  BP->allNeumannPenalty = 1.;
  hlong localElements = (hlong) mesh->Nelements;
  hlong totalElements = 0;
  MPI_Allreduce(&localElements, &totalElements, 1, MPI_HLONG, MPI_SUM, mesh->comm);
  BP->allNeumannScale = 1./sqrt((dfloat)mesh->Np*totalElements);

  BP->EToB = (int *) calloc(mesh->Nelements*mesh->Nfaces,sizeof(int));

  // !!!!!! Removed MPI::BOOL since some mpi versions complains about it !!!!!
  int lallNeumann, gallNeumann;
  lallNeumann = allNeumann ? 0:1;
  MPI_Allreduce(&lallNeumann, &gallNeumann, 1, MPI_INT, MPI_SUM, mesh->comm);
  BP->allNeumann = (gallNeumann>0) ? false: true;

  // MPI_Allreduce(&allNeumann, &(BP->allNeumann), 1, MPI::BOOL, MPI_LAND, mesh->comm);
  //  if (mesh->rank==0&& options.compareArgs("VERBOSE","TRUE"))
  printf("allNeumann = %d \n", BP->allNeumann);

  //copy boundary flags
  BP->o_EToB = mesh->device.malloc(mesh->Nelements*mesh->Nfaces*sizeof(int), BP->EToB);

  //setup an unmasked gs handle
  int verbose = options.compareArgs("VERBOSE","TRUE") ? 1:0;
  meshParallelGatherScatterSetup(mesh, Ntotal, mesh->globalIds, mesh->comm, verbose);

  //make a masked version of the global id numbering
  mesh->maskedGlobalIds = (hlong *) calloc(Ntotal,sizeof(hlong));
  memcpy(mesh->maskedGlobalIds, mesh->globalIds, Ntotal*sizeof(hlong));

  //use the masked ids to make another gs handle
  BP->ogs = ogsSetup(Ntotal, mesh->maskedGlobalIds, mesh->comm, verbose, mesh->device);
  BP->o_invDegree = BP->ogs->o_invDegree;

  if(mesh->device.mode()=="CUDA"){ // add backend compiler optimization for CUDA
    kernelInfo["compiler_flags"] += "-Xptxas -dlcm=ca";
  }

  if(mesh->device.mode()=="Serial")
    kernelInfo["compiler_flags"] += "-g";

  // set kernel name suffix
  char *suffix = strdup("Hex3D");

  char fileName[BUFSIZ], kernelName[BUFSIZ];

  kernelInfo["defines/" "p_blockSize"]= blockSize;
  
  for (int r=0;r<2;r++){
    if ((r==0 && mesh->rank==0) || (r==1 && mesh->rank>0)) {      

      //mesh kernels
      mesh->haloExtractKernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl",
                                       "meshHaloExtract3D",
                                       kernelInfo);

      mesh->addScalarKernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl",
                   "addScalar",
                   kernelInfo);

      mesh->maskKernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl",
                   "mask",
                   kernelInfo);



      mesh->sumKernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl",
                   "sum",
                   kernelInfo);

      BP->weightedInnerProduct1Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl",
                                       "weightedInnerProduct1",
                                       kernelInfo);

      BP->weightedInnerProduct2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl",
                                       "weightedInnerProduct2",
                                       kernelInfo);

      BP->innerProductKernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl",
                                       "innerProduct",
                                       kernelInfo);

      BP->weightedNorm2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl",
                                           "weightedNorm2",
                                           kernelInfo);

      BP->norm2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl",
                                           "norm2",
                                           kernelInfo);

      BP->scaledAddKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl",
                                         "scaledAdd",
                                         kernelInfo);

      BP->dotMultiplyKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl",
                                         "dotMultiply",
                                         kernelInfo);

      BP->dotMultiplyAddKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl",
                                         "dotMultiplyAdd",
                                         kernelInfo);

      BP->dotDivideKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl",
                                         "dotDivide",
                                         kernelInfo);

      // add custom defines

      kernelInfo["defines/" "p_Nfields"]= BP->Nfields;
      
      kernelInfo["defines/" "p_NpP"]= (mesh->Np+mesh->Nfp*mesh->Nfaces);
      kernelInfo["defines/" "p_Nverts"]= mesh->Nverts;

      int Nmax = mymax(mesh->Np, mesh->Nfaces*mesh->Nfp);
      kernelInfo["defines/" "p_Nmax"]= Nmax;

      int maxNodes = mymax(mesh->Np, (mesh->Nfp*mesh->Nfaces));
      kernelInfo["defines/" "p_maxNodes"]= maxNodes;

      int NblockV = mymax(1,maxNthreads/mesh->Np); // works for CUDA
      int NnodesV = 1; //hard coded for now
      kernelInfo["defines/" "p_NblockV"]= NblockV;
      kernelInfo["defines/" "p_NnodesV"]= NnodesV;

      int NblockS = mymax(1,maxNthreads/maxNodes); // works for CUDA
      kernelInfo["defines/" "p_NblockS"]= NblockS;

      int NblockP = mymax(1,maxNthreads/(4*mesh->Np)); // get close to maxNthreads threads
      kernelInfo["defines/" "p_NblockP"]= NblockP;

      int NblockG;
      if(mesh->Np<=32) NblockG = ( 32/mesh->Np );
      else NblockG = maxNthreads/mesh->Np;
      kernelInfo["defines/" "p_NblockG"]= NblockG;

      kernelInfo["defines/" "p_halfC"]= (int)((mesh->cubNq+1)/2);
      kernelInfo["defines/" "p_halfN"]= (int)((mesh->Nq+1)/2);

      kernelInfo["defines/" "p_NthreadsUpdatePCG"] = (int) NthreadsUpdatePCG; // WARNING SHOULD BE MULTIPLE OF 32
      kernelInfo["defines/" "p_NwarpsUpdatePCG"] = (int) (NthreadsUpdatePCG/32); // WARNING: CUDA SPECIFIC

      BP->BP1Kernel = mesh->device.buildKernel(DBP "/okl/BP1.okl", "BP1", kernelInfo);
      BP->BP3Kernel = mesh->device.buildKernel(DBP "/okl/BP3.okl", "BP3", kernelInfo);
      BP->BP5Kernel = mesh->device.buildKernel(DBP "/okl/BP5.okl", "BP5", kernelInfo);

      BP->BP1DotKernel = mesh->device.buildKernel(DBP "/okl/BP1.okl", "BP1Dot", kernelInfo);
      BP->BP3DotKernel = mesh->device.buildKernel(DBP "/okl/BP3.okl", "BP3Dot", kernelInfo);
      BP->BP5DotKernel = mesh->device.buildKernel(DBP "/okl/BP5.okl", "BP5Dot", kernelInfo);
      
      // combined PCG update and r.r kernel
      BP->updatePCGKernel =
	mesh->device.buildKernel(DBP "/okl/BPUpdatePCG.okl", "BPUpdatePCG", kernelInfo);

      occa::kernel nothingKernel = mesh->device.buildKernel(DBP "/okl/utils.okl", "nothingKernel", kernelInfo);
      nothingKernel();
     
      MPI_Barrier(mesh->comm);
    }
  }

  // TW: WARNING C0 appropriate only
  mesh->sumKernel(mesh->Nelements*mesh->Np, BP->o_invDegree, BP->o_tmp);
  BP->o_tmp.copyTo(BP->tmp);

  dfloat nullProjectWeightLocal = 0;
  dfloat nullProjectWeightGlobal = 0;
  for(dlong n=0;n<BP->Nblock;++n)
    nullProjectWeightLocal += BP->tmp[n];
  
  MPI_Allreduce(&nullProjectWeightLocal, &nullProjectWeightGlobal, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
  
  BP->nullProjectWeightGlobal = 1./nullProjectWeightGlobal;


}
