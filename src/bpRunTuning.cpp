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

#include "bp.hpp"
#include "timer.hpp"
#include "parameters.hpp"

int NkernelsFn(mesh_t& mesh, bool affine, int P);
int maxElementsPerBlockFn(mesh_t& mesh, bool affine, int P, int k);
int maxElementsPerThreadFn(mesh_t& mesh, bool affine, int P, int k);
int blockSizeFn(mesh_t& mesh, bool affine, int P, int k, int elementsPerBlock);
size_t shmemUse(mesh_t& mesh, bool affine, int P, int k, int elementsPerBlock, int elementsPerThread);

void bp_t::RunTuning(){

  // build OCCA kernels
  properties_t kernelInfo = mesh.props; //copy base occa properties
  kernelInfo["defines/" "p_Nfields"]= Nfields;

  //rhs kernel
  forcingKernel = platform.buildKernel(rhsFileName(),
                                       rhsKernelName(),
                                       kernelInfo);

  //create occa buffers
  dlong Ngather = ogs.Ngather + gHalo.Nhalo;
  deviceMemory<dfloat> o_q  = platform.malloc<dfloat>(Ngather);

  //populate x with a typical rhs (use Aq as temp storage)
  dfloat zero = 0.0;
  forcingKernel(mesh.Nelements,
                mesh.o_wJ,
                mesh.o_gllw,
                mesh.o_MM,
                mesh.o_x,
                mesh.o_y,
                mesh.o_z,
                zero,
                o_AqL);

  // gather rhs
  ogs.Gather(o_q, o_AqL, 1, ogs::Add, ogs::Trans);

  std::string name = "BK" + std::to_string(problemNumber);

  int Ntests = 50;

  int wavesize = 1;
  if (platform.device.mode() == "CUDA") wavesize = 32;
  if (platform.device.mode() == "HIP") wavesize = 64;

  size_t shmemLimit = 64*1024; //64 KB
  if (platform.device.mode() == "CUDA") shmemLimit = 48*1024; //48 KB

  double bestKernelBW = 0;
  // double bestKernelGflops = 0;
  int bestKernel = 0;
  int bestKernelElementsPerBlock = 0;
  int bestKernelElementsPerThread = 0;

  bool affine = settings.compareSetting("AFFINE MESH", "TRUE");

  int Nkernels = NkernelsFn(mesh, affine, problemNumber);

  for (int k=2;k<Nkernels;++k) {
    // if (mesh.rank==0) { printf("Testing Kernel %d\n", k);}

    double maxBW = 0;
    double maxGflops = 0;
    int bestElementsPerBlock = 0;
    int bestElementsPerThread = 0;

    int maxElementsPerBlock = maxElementsPerBlockFn(mesh, affine, problemNumber, k);
    int maxElementsPerThread = maxElementsPerThreadFn(mesh, affine, problemNumber, k);

    properties_t props = kernelInfo;
    props["defines/KERNEL_NUMBER"] = k;


    for (int elementsPerBlock=1;elementsPerBlock<=maxElementsPerBlock;elementsPerBlock++) {

      //Count number of waves
      int Nwaves = (blockSizeFn(mesh, affine, problemNumber, k, elementsPerBlock)+wavesize-1)/wavesize;
      //increment elementsPerBlock to fill wave count
      if (maxElementsPerBlock>1)
        while ((blockSizeFn(mesh, affine, problemNumber, k, elementsPerBlock+1)+wavesize-1)/wavesize == Nwaves) elementsPerBlock++;

      int blockSize = blockSizeFn(mesh, affine, problemNumber, k, elementsPerBlock);
      if (blockSize>1024) break;

      for (int elementsPerThread=1; elementsPerThread<=maxElementsPerThread;elementsPerThread++) {

        //Check shmem use
        size_t shmem = shmemUse(mesh, affine, problemNumber, k,
                                elementsPerBlock, elementsPerThread);
        if (shmem > shmemLimit) continue;

        if (maxElementsPerBlock>1)  props["defines/p_NelementsPerBlk"] = elementsPerBlock;
        if (maxElementsPerThread>1) props["defines/p_NelementsPerThd"] = elementsPerThread;

        // Ax kernel
        operatorKernel = platform.buildKernel(AxFileName(),
                                              AxKernelName(),
                                              props);

        for(int n=0;n<5;++n){ //warmup
          LocalOperator(o_q, o_AqL);
        }

        timePoint_t start = GlobalPlatformTime(platform);
        for(int n=0;n<Ntests;++n){
          LocalOperator(o_q, o_AqL);
        }
        timePoint_t end = GlobalPlatformTime(platform);
        double elapsedTime = ElapsedTime(start,end)/Ntests;

        size_t Nbytes = AxBytesMoved();
        size_t Nflops = AxFLOPs();

        double BW = Nbytes/(1.0e9 * elapsedTime);
        double gflops = Nflops/(1.0e9 * elapsedTime);


        std::string suffix = "Element=" + mesh.elementName();
        if (affine) suffix += ", Affine";

        if (mesh.rank==0 && settings.compareSetting("VERBOSE", "TRUE")){
          hlong Ndofs = ogs.NgatherGlobal;
          printf("%s: N=%2d, DOFs=" hlongFormat ", elapsed=%4.4f, avg BW (GB/s)=%6.1f, avg GFLOPs=%6.1f, %s, blocksize=%d",
                 name.c_str(),
                 mesh.N,
                 Ndofs,
                 elapsedTime,
                 BW,
                 gflops,
                 suffix.c_str(),
                 blockSize);
          if (maxElementsPerBlock>1)  printf(", elementsPerBlock = %d", elementsPerBlock);
          if (maxElementsPerThread>1) printf(", elementsPerThread = %d", elementsPerThread);
          printf("\n");
        }

        if (BW > maxBW) {
          maxBW = BW;
          maxGflops = gflops;
          bestElementsPerBlock = elementsPerBlock;
          bestElementsPerThread = elementsPerThread;
        }
      }
    }
    if (mesh.rank==0){
      printf("%s: Kernel=%d, N=%2d, BW = %6.1f, GFLOPS = %6.1f", name.c_str(), k, mesh.N, maxBW, maxGflops);
      if (maxElementsPerBlock>1)  printf(", bestElementsPerBlock = %d", bestElementsPerBlock);
      if (maxElementsPerThread>1) printf(", bestElementsPerThread = %d", bestElementsPerThread);
      printf("\n");
    }

    if (maxBW > bestKernelBW) {
      bestKernel = k;
      bestKernelBW = maxBW;
      // bestKernelGflops = maxGflops;
      bestKernelElementsPerBlock = bestElementsPerBlock;
      bestKernelElementsPerThread = bestElementsPerThread;
    }
  }

  std::string nm = "bp" + std::to_string(problemNumber);
  if (affine) {
    nm += "AxAffine" + mesh.elementName() + ".okl";
  } else {
    nm += "Ax" + mesh.elementName() + ".okl";
  }

  properties_t keys;
  keys["dfloat"] = (sizeof(dfloat)==4) ? "float" : "double";
  keys["N"] = mesh.N;
  keys["mode"] = platform.device.mode();

  std::string arch = platform.device.arch();
  if (platform.device.mode()=="HIP") {
    arch = arch.substr(0,arch.find(":")); //For HIP mode, remove the stuff after the :
  }
  keys["arch"] = arch;

  properties_t kprops;
  kprops["KERNEL_NUMBER"] = bestKernel;

  int maxElementsPerBlock = maxElementsPerBlockFn(mesh, affine, problemNumber, bestKernel);
  int maxElementsPerThread = maxElementsPerThreadFn(mesh, affine, problemNumber, bestKernel);

  if (maxElementsPerBlock>1)  kprops["p_NelementsPerBlk"] = bestKernelElementsPerBlock;
  if (maxElementsPerThread>1) kprops["p_NelementsPerThd"] = bestKernelElementsPerThread;

  properties_t params;
  params["Name"] = nm;
  params["keys"] = keys;
  params["props"] = kprops;

  std::cout << parameters_t::toString(params) << std::endl;
}

int NkernelsFn(mesh_t& mesh, bool affine, int P) {
  if (mesh.elementType==mesh_t::TRIANGLES) {
    if (P==1 && !affine) return 4;
    if (P==1 &&  affine) return 4;
  } else if (mesh.elementType==mesh_t::QUADRILATERALS) {
    if (P==1 && !affine) return 2;
    if (P==1 &&  affine) return 2;
  } else if (mesh.elementType==mesh_t::TETRAHEDRA) {
    if (P==1 && !affine) return 4;
    if (P==1 &&  affine) return 4;
  } else {
    if (P==1 && !affine) return 3;
    if (P==1 &&  affine) return 3;
  }
  return 0;
}

int maxElementsPerBlockFn(mesh_t& mesh, bool affine, int P, int k) {
  if (mesh.elementType==mesh_t::TRIANGLES) {
    if (P==1 && !affine) {
      if (k==0) return 1;
      if (k==1) return 1024;
      if (k==2) return 1024;
      if (k==3) return 1;
    }
    if (P==1 &&  affine) {
      if (k==0) return 1;
      if (k==1) return 1024;
      if (k==2) return 1024;
      if (k==3) return 1;
    }
  } else if (mesh.elementType==mesh_t::QUADRILATERALS) {
    if (P==1 && !affine) {
      if (k==0) return 1024;
      if (k==1) return 1;
    }
    if (P==1 &&  affine) {
      if (k==0) return 1024;
      if (k==1) return 1;
    }
  } else if (mesh.elementType==mesh_t::TETRAHEDRA) {
    if (P==1 && !affine) {
      if (k==0) return 1;
      if (k==1) return 1024;
      if (k==2) return 1024;
      if (k==3) return 1;
    }
    if (P==1 &&  affine) {
      if (k==0) return 1;
      if (k==1) return 1024;
      if (k==2) return 1024;
      if (k==3) return 1;
    }
  } else {
    if (P==1 && !affine) {
      if (k==0) return 256;
      if (k==1) return 256;
      if (k==2) return 1;
    }
    if (P==1 &&  affine) {
      if (k==0) return 256;
      if (k==1) return 256;
      if (k==2) return 1;
    }
  }
  return 0;
}

int maxElementsPerThreadFn(mesh_t& mesh, bool affine, int P, int k) {
  if (mesh.elementType==mesh_t::TRIANGLES) {
    if (P==1 && !affine) {
      if (k==0) return 1;
      if (k==1) return 16;
      if (k==2) return 16;
      if (k==3) return 1;
    }
    if (P==1 &&  affine) {
      if (k==0) return 1;
      if (k==1) return 16;
      if (k==2) return 16;
      if (k==3) return 1;
    }
  } else if (mesh.elementType==mesh_t::QUADRILATERALS) {
    if (P==1 && !affine) {
      if (k==0) return 4;
      if (k==1) return 4;
    }
    if (P==1 &&  affine) {
      if (k==0) return 4;
      if (k==1) return 4;
    }
  } else if (mesh.elementType==mesh_t::TETRAHEDRA) {
    if (P==1 && !affine) {
      if (k==0) return 1;
      if (k==1) return 16;
      if (k==2) return 16;
      if (k==3) return 1;
    }
    if (P==1 &&  affine) {
      if (k==0) return 1;
      if (k==1) return 16;
      if (k==2) return 16;
      if (k==3) return 1;
    }
  } else {
    if (P==1 && !affine) {
      if (k==0) return 1;
      if (k==1) return 1;
      if (k==2) return 1;
    }
    if (P==1 &&  affine) {
      if (k==0) return 1;
      if (k==1) return 1;
      if (k==2) return 1;
    }
  }
  return 0;
}

int blockSizeFn(mesh_t& mesh, bool affine, int P, int k, int elementsPerBlock) {
  int Nq = mesh.Nq;
  int Np = mesh.Np;
  int cubNq = mesh.cubNq;
  int cubNp = mesh.cubNp;

  if (mesh.elementType==mesh_t::TRIANGLES) {
    if (P==1 && !affine) {
      if (k==0) return cubNp;
      if (k==1) return elementsPerBlock*cubNp;
      if (k==2) return elementsPerBlock*cubNp;
      if (k==3) return 4*16*((cubNp-1)/16 + 1);
    }
    if (P==1 &&  affine) {
      if (k==0) return Np;
      if (k==1) return elementsPerBlock*Np;
      if (k==2) return elementsPerBlock*Np;
      if (k==3) return 4*16*((Np-1)/16 + 1);
    }
  } else if (mesh.elementType==mesh_t::QUADRILATERALS) {
    if (P==1 && !affine) {
      if (k==0) return elementsPerBlock*cubNq*cubNq;
      if (k==1) return 256;
    }
    if (P==1 &&  affine) {
      if (k==0) return elementsPerBlock*Nq*Nq;
      if (k==1) return 256;
    }
  } else if (mesh.elementType==mesh_t::TETRAHEDRA) {
    if (P==1 && !affine) {
      if (k==0) return cubNp;
      if (k==1) return elementsPerBlock*cubNp;
      if (k==2) return elementsPerBlock*cubNp;
      if (k==3) return 1024;
    }
    if (P==1 &&  affine) {
      if (k==0) return Np;
      if (k==1) return elementsPerBlock*Np;
      if (k==2) return elementsPerBlock*Np;
      if (k==3) return 4*16*((Np-1)/16 + 1);
    }
  } else {
    if (P==1 && !affine) {
      if (k==0) return elementsPerBlock*cubNq*cubNq;
      if (k==1) return elementsPerBlock*cubNq*cubNq;
      if (k==2) return 256;
    }
    if (P==1 &&  affine) {
      if (k==0) return elementsPerBlock*Nq*Nq;
      if (k==1) return elementsPerBlock*Nq*Nq;
      if (k==2) return 256;
    }
  }
  return 0;
}

size_t shmemUse(mesh_t& mesh, bool affine, int P, int k,
                int elementsPerBlock, int elementsPerThread) {
  int Nq = mesh.Nq;
  int Np = mesh.Np;
  int cubNq = mesh.cubNq;
  int cubNp = mesh.cubNp;

  if (mesh.elementType==mesh_t::TRIANGLES) {
    if (P==1 && !affine) {
      if (k==0) return sizeof(dfloat)*cubNp;
      if (k==1) return sizeof(dfloat)*(Np*cubNp + cubNp*elementsPerBlock*elementsPerThread);
      if (k==2) return sizeof(dfloat)*(cubNp*elementsPerBlock*elementsPerThread);
      if (k==3) return sizeof(dfloat)*(16*cubNp);
    }
    if (P==1 &&  affine) {
      if (k==0) return sizeof(dfloat)*Np;
      if (k==1) return sizeof(dfloat)*(Np*Np + Np*elementsPerBlock*elementsPerThread);
      if (k==2) return sizeof(dfloat)*(Np*elementsPerBlock*elementsPerThread);
      if (k==3) return sizeof(dfloat)*(16*Np);
    }
  } else if (mesh.elementType==mesh_t::QUADRILATERALS) {
    if (P==1 && !affine) {
      if (k==0) return sizeof(dfloat)*(Nq*cubNq + cubNq*cubNq*elementsPerBlock*elementsPerThread);
      if (k==1) return sizeof(dfloat)*((16+1)*16 + (16+1)*16*elementsPerThread);
    }
    if (P==1 &&  affine) {
      if (k==0) return sizeof(dfloat)*(Nq*Nq + Nq*Nq*elementsPerBlock*elementsPerThread);
      if (k==1) return sizeof(dfloat)*((16+1)*16 + (16+1)*Nq*elementsPerThread);
    }
  } else if (mesh.elementType==mesh_t::TETRAHEDRA) {
    if (P==1 && !affine) {
      if (k==0) return sizeof(dfloat)*cubNp;
      if (k==1) return sizeof(dfloat)*(Np*cubNp + cubNp*elementsPerBlock*elementsPerThread);
      if (k==2) return sizeof(dfloat)*(cubNp*elementsPerBlock*elementsPerThread);
      if (k==3) return sizeof(dfloat)*(16*cubNp);
    }
    if (P==1 &&  affine) {
      if (k==0) return sizeof(dfloat)*Np;
      if (k==1) return sizeof(dfloat)*(Np*Np + Np*elementsPerBlock*elementsPerThread);
      if (k==2) return sizeof(dfloat)*(Np*elementsPerBlock*elementsPerThread);
      if (k==3) return sizeof(dfloat)*(16*Np);
    }
  } else {
    if (P==1 && !affine) {
      if (k==0) return sizeof(dfloat)*(cubNq*Nq + cubNq*cubNq*cubNq*elementsPerBlock);
      if (k==1) return sizeof(dfloat)*(cubNq*Nq + cubNq*cubNq*elementsPerBlock);
      if (k==2) return sizeof(dfloat)*((16+1)*16 + (16+1)*16);
    }
    if (P==1 &&  affine) {
      if (k==0) return sizeof(dfloat)*(Nq*(Nq+1) + (Nq+1)*Nq*Nq*elementsPerBlock);
      if (k==1) return sizeof(dfloat)*(Nq*(Nq+1) + (Nq+1)*Nq*elementsPerBlock);
      if (k==2) return sizeof(dfloat)*((16+1)*16 + (16+1)*16);
    }
  }
  return 0;
}
