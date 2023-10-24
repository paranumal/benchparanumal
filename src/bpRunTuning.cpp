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

  double maxBW = 0;
  int bestElementsPerBlock = 0;
  int bestElementsPerThread = 0;

  int wavesize = 1;
  if (platform.device.mode() == "CUDA") wavesize = 32;
  if (platform.device.mode() == "HIP") wavesize = 64;

  size_t shmemLimit = 64*1024; //64 KB
  if (platform.device.mode() == "CUDA") shmemLimit = 48*1024; //48 KB

  int blockStep = ((mesh.Np+wavesize)/wavesize)*wavesize;

  int kernelNumber=0;
  kernelInfo["defines/KERNEL_NUMBER"] = kernelNumber;

  for (int targetBlocksize=wavesize; targetBlocksize<=1024;targetBlocksize+=blockStep)
  // int targetBlocksize=wavesize;
  {

    //try to get close to target blocksize
    int elementsPerBlock = targetBlocksize/mesh.Np;

    if (elementsPerBlock<1) continue;

    for (int elementsPerThread=1; elementsPerThread<16;elementsPerThread++)
    // int elementsPerThread=1;
    {
      //Check shmem use
      // if (sizeof(dfloat)*elementsPerThread*elementsPerBlock*mesh.Np > shmemLimit) break;

      if (sizeof(dfloat)*elementsPerThread*elementsPerBlock*mesh.Np +
          sizeof(dfloat)*mesh.Np*mesh.Np > shmemLimit) break;

      properties_t props = kernelInfo;

      props["defines/p_NelementsPerBlk"] = elementsPerBlock;
      props["defines/p_NelementsPerThd"] = elementsPerThread;

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

      hlong Ndofs = ogs.NgatherGlobal;

      std::string suffix = "Element=" + mesh.elementName();
      if (settings.compareSetting("AFFINE MESH", "TRUE")) suffix += ", Affine";

      if (mesh.rank==0){
        printf("%s: N=%2d, DOFs=" hlongFormat ", elapsed=%4.4f, avg BW (GB/s)=%6.1f, avg GFLOPs=%6.1f, %s blocksize=%d, ElementsPerBlock=%d, ElementsPerThread=%d \n",
               name.c_str(),
               mesh.N,
               Ndofs,
               elapsedTime,
               Nbytes/(1.0e9 * elapsedTime),
               Nflops/(1.0e9 * elapsedTime),
               suffix.c_str(),
               elementsPerBlock*mesh.Np,
               elementsPerBlock,
               elementsPerThread);
      }

      double BW = Nbytes/(1.0e9 * elapsedTime);
      if (BW > maxBW) {
        maxBW = BW;
        bestElementsPerBlock = elementsPerBlock;
        bestElementsPerThread = elementsPerThread;
      }
    }
  }
  printf("%s: N=%2d, BW = %6.1f, bestElementsPerBlock = %d, bestElementsPerThread = %d\n",
         name.c_str(), mesh.N, maxBW, bestElementsPerBlock, bestElementsPerThread);
}
