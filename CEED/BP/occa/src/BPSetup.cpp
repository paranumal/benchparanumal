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

BP_t *BPSetup(mesh_t *mesh, dfloat lambda, dfloat mu, occa::properties &kernelInfo, setupAide &options){

  BP_t *BP = new BP_t();

  // load forcing into r
  int BP1 = options.compareArgs("BENCHMARK", "BP1");
  int BP2 = options.compareArgs("BENCHMARK", "BP2");
  int BP3 = options.compareArgs("BENCHMARK", "BP3");
  int BP4 = options.compareArgs("BENCHMARK", "BP4");
  int BP5 = options.compareArgs("BENCHMARK", "BP5");
  int BP6 = options.compareArgs("BENCHMARK", "BP6");
  int BP9 = options.compareArgs("BENCHMARK", "BP9");
  int BP10 = options.compareArgs("BENCHMARK", "BP10");

  if(BP9)
    options.setArgs("KRYLOV SOLVER", "MINRES");

  printf("BP0:10=%d,%d,%d,%d,%d,%d,%d,%d\n",
	 BP1, BP2, BP3, BP4, BP5, BP6, BP9, BP10);
  
  BP->BPid =
    1*(BP1==1) + 2*(BP2==1) + 3*(BP3==1) + 4*(BP4==1) +
    5*(BP5==1) + 6*(BP6==1) + 9*(BP9==1) + 10*(BP10==1);

  if(BP1 || BP3 || BP5)
    BP->Nfields = 1;
  else if(BP2 || BP4 || BP6)
    BP->Nfields = 3;
  else if(BP9)
    BP->Nfields = 4; // BP9 STOKES WITH EQUAL ORDER 
  else if(BP10)
    BP->Nfields = 1;
  else{
    printf("BP%d is not supported\n", BP->BPid);
    exit(-1);
  }

  options.getArgs("MESH DIMENSION", BP->dim);
  options.getArgs("ELEMENT TYPE", BP->elementType);
  BP->mesh = mesh;
  BP->options = options;
  
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

  printf("ENTERING BPSOLVESETUP\n");
  BPSolveSetup(BP, lambda, mu, kernelInfo);

  dlong Ndof = mesh->Np*(mesh->Nelements+mesh->totalHaloPairs);
  dlong Nall = BP->Nfields*Ndof;
  BP->r   = (dfloat*) calloc(Nall,   sizeof(dfloat));
  BP->x   = (dfloat*) calloc(Nall,   sizeof(dfloat));
  BP->q   = (dfloat*) calloc(Nall,   sizeof(dfloat));

#if 0
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int n=0;n<mesh->Np;++n){

      dfloat JW = mesh->ggeo[mesh->Np*(e*mesh->Nggeo + GWJID) + n];

      dlong id = n+e*mesh->Np;
      dfloat xn = mesh->x[id];
      dfloat yn = mesh->y[id];
      dfloat zn = mesh->z[id];
      
      dfloat mode = 1;

      for(int fld=0;fld<BP->Nfields;++fld){
	dlong fldid = id + fld*Ndof;
	
	// mass projection rhs
	BP->r[fldid] =
	  (3.*M_PI*M_PI*mode*mode+lambda)*JW*cos(mode*M_PI*xn)*cos(mode*M_PI*yn)*cos(mode*M_PI*zn);

	BP->x[fldid] = 0;
      }
    }
  }
#else
  int cubNp = mesh->cubNp;

  dfloat cubrhs[BP->Nfields][cubNp];
  dfloat *cubx = (dfloat*) calloc(cubNp, sizeof(dfloat));
  dfloat *cuby = (dfloat*) calloc(cubNp, sizeof(dfloat));
  dfloat *cubz = (dfloat*) calloc(cubNp, sizeof(dfloat));
  dfloat *cubInterpT;
  dfloat *cubInterp3DT;

  cubInterpT = (dfloat*) calloc(mesh->cubNq*mesh->Nq, sizeof(dfloat));

  // BUILD 3D version here too
  for(int i=0;i<mesh->cubNq;++i){
    for(int a=0;a<mesh->Nq;++a){
      cubInterpT[a*mesh->cubNq + i] = mesh->cubInterp[i*mesh->Nq+a];
    }
  }

  if(mesh->elementType==TETRAHEDRA){
    cubInterp3DT = (dfloat*) calloc(mesh->cubNp*mesh->Np, sizeof(dfloat));
    
    // BUILD 3D version here too
    for(int i=0;i<mesh->cubNp;++i){
      for(int a=0;a<mesh->Np;++a){
	cubInterp3DT[a*mesh->cubNp + i] = mesh->cubInterp3D[i*mesh->Np+a];
      }
    }
  }

  for(dlong e=0;e<mesh->Nelements;++e){

    if(mesh->elementType==HEXAHEDRA){
      meshInterpolateHex3D(mesh->cubInterp, mesh->x+e*mesh->Np, mesh->Nq, cubx, mesh->cubNq);
      meshInterpolateHex3D(mesh->cubInterp, mesh->y+e*mesh->Np, mesh->Nq, cuby, mesh->cubNq);
      meshInterpolateHex3D(mesh->cubInterp, mesh->z+e*mesh->Np, mesh->Nq, cubz, mesh->cubNq);
    }
    else{
      meshInterpolateTet3D(mesh->cubInterp3D, mesh->x+e*mesh->Np, mesh->Np, cubx, mesh->cubNp);
      meshInterpolateTet3D(mesh->cubInterp3D, mesh->y+e*mesh->Np, mesh->Np, cuby, mesh->cubNp);
      meshInterpolateTet3D(mesh->cubInterp3D, mesh->z+e*mesh->Np, mesh->Np, cubz, mesh->cubNp);
    }

    for(int n=0;n<cubNp;++n){
      
      dfloat JW = mesh->cubggeo[cubNp*(e*mesh->Nggeo + GWJID) + n];
      
      dfloat xn = cubx[n];
      dfloat yn = cuby[n];
      dfloat zn = cubz[n];
      
      dfloat mode = 1;

      for(int fld=0;fld<BP->Nfields;++fld){
	// mass projection rhs
	cubrhs[fld][n] =
	  (3.*M_PI*M_PI*mode*mode+lambda)*JW*cos(mode*M_PI*xn)*cos(mode*M_PI*yn)*cos(mode*M_PI*zn);
      }
      if(BP->BPid==9) // hack
	cubrhs[BP->Nfields-1][n] = 0;


      //      printf("%e %e %e: %e\n", xn, yn, zn, cubrhs[0][n]);
    }
    
    for(int fld=0;fld<BP->Nfields;++fld){
      for(int n=0;n<mesh->Np;++n){
	dlong fldid = n + fld*Ndof + e*mesh->Np;
	BP->x[fldid] = 0;
      }

      if(mesh->elementType==HEXAHEDRA){
	meshInterpolateHex3D(cubInterpT, cubrhs[fld], mesh->cubNq, BP->r + e*mesh->Np + fld*Ndof, mesh->Nq);
      }
      else{
	meshInterpolateTet3D(cubInterp3DT, cubrhs[fld], mesh->cubNp, BP->r + e*mesh->Np + fld*Ndof, mesh->Np);
#if 0
	for(int n=0;n<mesh->Np;++n){
	  printf("% e ", BP->r[e*mesh->Np+n]);
	}
	printf("\n r:");
#endif
      }
    }
  }
#endif

  //copy to occa buffers

  int useGlobal  = options.compareArgs("USE GLOBAL STORAGE", "TRUE");
  if(!useGlobal){
    BP->o_r   = mesh->device.malloc(Nall*sizeof(dfloat), BP->r);
    BP->o_x   = mesh->device.malloc(Nall*sizeof(dfloat), BP->x);
    
    char *suffix = strdup("Hex3D");
    
    // gather-scatterd
    if(options.compareArgs("DISCRETIZATION","CONTINUOUS")){
      if(BP->Nfields == 1)
	ogsGatherScatter(BP->o_r, ogsDfloat, ogsAdd, mesh->ogs);
      else
	ogsGatherScatterMany(BP->o_r, BP->Nfields, Ndof, ogsDfloat, ogsAdd, mesh->ogs);
    }
  }
  else{
    BP->o_r  = mesh->device.malloc(Nall*sizeof(dfloat), BP->r);
    BP->o_gr = mesh->device.malloc(BP->Nfields*mesh->Nlocalized*sizeof(dfloat), BP->x); //ZERO !
    BP->o_x  = mesh->device.malloc(BP->Nfields*mesh->Nlocalized*sizeof(dfloat), BP->x);

    BP->vecAtomicMultipleGatherKernel(mesh->Np*mesh->Nelements, mesh->Nlocalized,
				      mesh->o_localizedIds, BP->o_r, BP->o_gr);
  }

  if (mesh->rank==0)
    reportMemoryUsage(mesh->device, "after BP setup");

  
  return BP;
}


void BPSolveSetup(BP_t *BP, dfloat lambda, dfloat mu, occa::properties &kernelInfo){

  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;

  int knlId = 0;
  options.getArgs("KERNEL ID", knlId);
  BP->knlId = knlId;
  
  dlong Ntotal = mesh->Np*mesh->Nelements;
  dlong Nhalo  = mesh->Np*mesh->totalHaloPairs;
  dlong Nall   = (Ntotal + Nhalo)*BP->Nfields;
  
  dlong Nblock  = mymax(1,(Ntotal+blockSize-1)/blockSize);
  dlong Nblock2 = mymax(1,(Nblock+blockSize-1)/blockSize);

  dlong NthreadsUpdatePCG = 1024; // was 256
  dlong NblocksUpdatePCG = mymin((Ntotal+NthreadsUpdatePCG-1)/NthreadsUpdatePCG, 640);
  //  dlong NblocksUpdatePCG = (Ntotal+NthreadsUpdatePCG-1)/NthreadsUpdatePCG;x
 
  BP->NthreadsUpdatePCG = NthreadsUpdatePCG;
  BP->NblocksUpdatePCG = NblocksUpdatePCG;

  BP->NsolveWorkspace = 10;
  BP->solveWorkspace = (dfloat*) calloc(Nall*BP->NsolveWorkspace, sizeof(dfloat));
  
  //  BP->o_solveWorkspace  = mesh->device.malloc(Nall*BP->NsolveWorkspace*sizeof(dfloat), BP->solveWorkspace);

  BP->o_solveWorkspace = new occa::memory[BP->NsolveWorkspace];
  for(int wk=0;wk<BP->NsolveWorkspace;++wk)
    BP->o_solveWorkspace[wk]  =
      mesh->device.malloc(Nall*sizeof(dfloat), BP->solveWorkspace);


  BP->tmp  = (dfloat*) calloc(Nblock, sizeof(dfloat));
  //  BP->tmp2 = (dfloat*) calloc(Nblock2, sizeof(dfloat));
  
  BP->o_tmp = mesh->device.malloc(Nblock*sizeof(dfloat), BP->tmp);
  BP->o_tmp2 = mesh->device.malloc(Nblock2*sizeof(dfloat), BP->tmp);

  BP->tmpNormr = (dfloat*) calloc(BP->NblocksUpdatePCG,sizeof(dfloat));
  BP->o_tmpNormr = mesh->device.malloc(BP->NblocksUpdatePCG*sizeof(dfloat), BP->tmpNormr);

  BP->tmpAtomic = (dfloat*) calloc(1,sizeof(dfloat));
  BP->o_tmpAtomic = mesh->device.malloc(1*sizeof(dfloat), BP->tmpAtomic);
  BP->o_zeroAtomic = mesh->device.malloc(1*sizeof(dfloat), BP->tmpAtomic);
  
  //setup async halo stream
  BP->defaultStream = mesh->defaultStream;
  BP->dataStream = mesh->dataStream;

  dlong Nbytes = BP->Nfields*mesh->totalHaloPairs*mesh->Np*sizeof(dfloat);
  if(Nbytes>0){
    BP->sendBuffer =
      (dfloat*) occaHostMallocPinned(mesh->device, Nbytes, NULL, BP->o_sendBuffer, BP->h_sendBuffer);
    BP->recvBuffer =
      (dfloat*) occaHostMallocPinned(mesh->device, Nbytes, NULL, BP->o_recvBuffer, BP->h_recvBuffer);
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

  //setup an unmasked gs handlex
  int verbose = options.compareArgs("VERBOSE","TRUE") ? 1:0;
  meshParallelGatherScatterSetup(mesh, Ntotal, mesh->globalIds, mesh->comm, verbose);

  //make a masked version of the global id numbering
  mesh->maskedGlobalIds = (hlong *) calloc(Ntotal,sizeof(hlong));
  memcpy(mesh->maskedGlobalIds, mesh->globalIds, Ntotal*sizeof(hlong));

  //use the masked ids to make another gs handle
  BP->ogs = ogsSetup(Ntotal, mesh->maskedGlobalIds, mesh->comm, verbose, mesh->device);
  BP->o_invDegree = BP->ogs->o_invDegree;

  // set kernel name suffix
  char fileName[BUFSIZ], kernelName[BUFSIZ];

  kernelInfo["defines/" "p_blockSize"]= blockSize;
  kernelInfo["defines/" "p_Nfields"]= BP->Nfields;

  printf("ENTERING KERNEL BUILDS\n");
  
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
        mesh->device.buildKernel(DBP "/okl/utils.okl", "weightedInnerProduct1", kernelInfo);

      BP->weightedInnerProduct2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl", "weightedInnerProduct2", kernelInfo);
      
      BP->weightedMultipleInnerProduct2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl", "weightedMultipleInnerProduct2", kernelInfo);

      BP->multipleInnerProduct2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl", "multipleInnerProduct2", kernelInfo);

      BP->innerProductKernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl", "innerProduct", kernelInfo);

      BP->innerProduct2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl", "innerProduct", kernelInfo);

      BP->weightedNorm2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl", "weightedNorm2", kernelInfo);

      BP->weightedMultipleNorm2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl", "weightedMultipleNorm2", kernelInfo);

      BP->norm2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl", "norm2", kernelInfo);

      BP->multipleNorm2Kernel =
        mesh->device.buildKernel(DBP "/okl/utils.okl", "multipleNorm2", kernelInfo);

      
      BP->scaledAddKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl", "scaledAdd", kernelInfo);

      BP->dotMultiplyKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl", "dotMultiply", kernelInfo);

      BP->dotMultiplyAddKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl", "dotMultiplyAdd", kernelInfo);

      BP->dotDivideKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl", "dotDivide", kernelInfo);

      BP->vecZeroKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl", "vecZero", kernelInfo);

      BP->vecScaleKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl", "vecScale", kernelInfo);

      BP->vecCopyKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl", "vecCopy", kernelInfo);

      BP->vecAtomicGatherKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl", "vecAtomicGather", kernelInfo);

      BP->vecAtomicMultipleGatherKernel =
          mesh->device.buildKernel(DBP "/okl/utils.okl", "vecAtomicMultipleGather", kernelInfo);

      BP->vecAtomicInnerProductKernel =
	mesh->device.buildKernel(DBP "/okl/utils.okl", "vecAtomicInnerProduct", kernelInfo);

      

      BP->vecScatterKernel =
	mesh->device.buildKernel(DBP "/okl/utils.okl", "vecScatter", kernelInfo);
      
      BP->vecMultipleScatterKernel =
	mesh->device.buildKernel(DBP "/okl/utils.okl", "vecMultipleScatter", kernelInfo);

      
      // add custom defines
      kernelInfo["defines/" "p_NpTet"]= mesh->Np;
      
      kernelInfo["defines/" "p_NpP"]= (mesh->Np+mesh->Nfp*mesh->Nfaces);
      kernelInfo["defines/" "p_Nverts"]= mesh->Nverts;

      int Nmax = mymax(mesh->Np, mesh->Nfaces*mesh->Nfp);
      int maxNodes = mymax(mesh->Np, (mesh->Nfp*mesh->Nfaces));
      int NblockV = mymax(1,maxNthreads/mesh->Np); // works for CUDA
      int NnodesV = 1; //hard coded for now
      int NblockS = mymax(1,maxNthreads/maxNodes); // works for CUDA
      int NblockP = mymax(1,maxNthreads/(4*mesh->Np)); // get close to maxNthreads threads
      int NblockG;
      if(mesh->Np<=32) NblockG = ( 32/mesh->Np );
      else NblockG = maxNthreads/mesh->Np;
      
      kernelInfo["defines/" "p_Nmax"]= Nmax;
      kernelInfo["defines/" "p_maxNodes"]= maxNodes;
      kernelInfo["defines/" "p_NblockV"]= NblockV;
      kernelInfo["defines/" "p_NnodesV"]= NnodesV;
      kernelInfo["defines/" "p_NblockS"]= NblockS;
      kernelInfo["defines/" "p_NblockP"]= NblockP;
      kernelInfo["defines/" "p_NblockG"]= NblockG;

      kernelInfo["defines/" "p_halfC"]= (int)((mesh->cubNq+1)/2);
      kernelInfo["defines/" "p_halfN"]= (int)((mesh->Nq+1)/2);

      kernelInfo["defines/" "p_NthreadsUpdatePCG"] = (int) NthreadsUpdatePCG; // WARNING SHOULD BE MULTIPLE OF 32
      kernelInfo["defines/" "p_NwarpsUpdatePCG"] = (int) (NthreadsUpdatePCG/32); // WARNING: CUDA SPECIFIC

      char kernelName[BUFSIZ], fileName[BUFSIZ];

      BP->BPKernel = (occa::kernel*) new occa::kernel[20];
      BP->BPKernelGlobal = (occa::kernel*) new occa::kernel[20];

      //      for(int bpid=1;bpid<=6;++bpid){
      int bpid = BP->BPid;
      int useGlobal = 0;
      int combineDot = 0;

      useGlobal  = options.compareArgs("USE GLOBAL STORAGE", "TRUE");
      combineDot = options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

      printf("useGlobal=%d\n", useGlobal);

      if(!useGlobal){
	printf("BUILDING LOCAL STORAGE KERNELS\n");
	sprintf(fileName, "%s/okl/BP%d.okl", DBP, bpid);

	if(!combineDot)
	  sprintf(kernelName, "BP%d_v%d", bpid, knlId);
	else
	  sprintf(kernelName, "BP%dDot_v%d", bpid, knlId);
      
	BP->BPKernel[bpid] = mesh->device.buildKernel(fileName, kernelName, kernelInfo);
      }
      else{
	printf("BUILDING GLOBAL KERNELS\n");
	
	sprintf(fileName, "%s/okl/BP%dGlobal.okl", DBP, bpid);
	
	if(!combineDot)
	  sprintf(kernelName, "BP%dGlobal_v%d", bpid, knlId);
	else
	  sprintf(kernelName, "BP%dDotGlobal_v%d", bpid, knlId);
	
	BP->BPKernelGlobal[bpid] = mesh->device.buildKernel(fileName, kernelName, kernelInfo);
      }
      
      printf("Loaded: %s from %s\n", kernelName, fileName);
      
      // combined PCG update and r.r kernel
      BP->updatePCGKernel =
	mesh->device.buildKernel(DBP "/okl/BPUpdatePCG.okl", "BPUpdatePCG", kernelInfo);

      BP->updateMultiplePCGKernel =
	mesh->device.buildKernel(DBP "/okl/BPUpdatePCG.okl", "BPMultipleUpdatePCG", kernelInfo);

      BP->updateMINRESKernel =
	mesh->device.buildKernel(DBP "/okl/BPUpdateMINRES.okl", "BPUpdateMINRES", kernelInfo);

      BP->updatePCGGlobalKernel =
	mesh->device.buildKernel(DBP "/okl/BPUpdatePCG.okl", "BPUpdatePCGGlobal", kernelInfo);

      BP->updateMultiplePCGGlobalKernel =
	mesh->device.buildKernel(DBP "/okl/BPUpdatePCG.okl", "BPMultipleUpdatePCGGlobal", kernelInfo);

#if 0
      BP->filterKernel =
	mesh->device.buildKernel(DBP "/okl/BP9.okl", "BPfilter", kernelInfo);
#endif
      
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

#if USE_CUDA_NATIVE==1
  // do NATIVE CUDA STUFF
  BK5Setup(mesh->Nelements,
	   mesh->Nq,
	   mesh->D, 
	   &(BP->c_DofToDofD),
	   &(BP->c_oddDofToDofD),
	   &(BP->c_evenDofToDofD));
#endif
}
