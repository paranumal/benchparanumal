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

void bp_t::RunBK(){

  // build OCCA kernels
  properties_t kernelInfo = mesh.props; //copy base occa properties
  kernelInfo["defines/" "p_Nfields"]= Nfields;

  //rhs kernel
  forcingKernel = platform.buildKernel(rhsFileName(),
                                       rhsKernelName(),
                                       kernelInfo);

  //Add tuning parameters
  AxTuningParams(kernelInfo);

  // Ax kernel
  operatorKernel = platform.buildKernel(AxFileName(),
                                        AxKernelName(),
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


  int Ntests = 50;

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

  std::string name = "BK" + std::to_string(problemNumber);

  std::string suffix = "Element=" + mesh.elementName();
  if (settings.compareSetting("AFFINE MESH", "TRUE")) suffix += ", Affine";

  if (mesh.rank==0){
    printf("%s: N=%2d, DOFs=" hlongFormat ", elapsed=%4.4f, time per DOF=%1.2e, avg BW (GB/s)=%6.1f, avg GFLOPs=%6.1f, DOFs/ranks*time=%1.2e, %s \n",
           name.c_str(),
           mesh.N,
           Ndofs,
           elapsedTime,
           elapsedTime/(Ndofs),
           Nbytes/(1.0e9 * elapsedTime),
           Nflops/(1.0e9 * elapsedTime),
           Ndofs/(mesh.size*elapsedTime),
           suffix.c_str());
  }
}

void bp_t::LocalOperator(deviceMemory<dfloat> &o_q, deviceMemory<dfloat> &o_Aq){
  switch (problemNumber) {
    case 1:
    case 2:
      if (mesh.elementType == mesh_t::TRIANGLES
          && problemNumber==1
          && platform.device.mode() == "HIP") {
        // Using native HIP kernel, set launch sizes
        operatorKernel.setRunDims((mesh.NlocalGatherElements+15)/16,
                                  occa::dim(16, 4));
      }
      operatorKernel(mesh.NlocalGatherElements,
                     mesh.o_localGatherElementList,
                     o_GlobalToLocal,
                     mesh.o_cubwJ,
                     mesh.o_cubInterp,
                     mesh.o_MM,
                     o_q,
                     o_Aq);
      break;
    case 3:
    case 4:
      operatorKernel(mesh.NlocalGatherElements,
                     mesh.o_localGatherElementList,
                     o_GlobalToLocal,
                     mesh.o_cubwJ,
                     mesh.o_cubggeo,
                     mesh.o_cubD,
                     mesh.o_cubInterp,
                     mesh.o_invV,
                     mesh.o_S,
                     mesh.o_MM,
                     lambda,
                     o_q,
                     o_Aq);
      break;
    case 5:
    case 6:
      operatorKernel(mesh.NlocalGatherElements,
                   mesh.o_localGatherElementList,
                   o_GlobalToLocal,
                   mesh.o_wJ,
                   mesh.o_vgeo,
                   mesh.o_ggeo,
                   mesh.o_gllw,
                   mesh.o_D,
                   mesh.o_S,
                   mesh.o_MM,
                   lambda,
                   o_q,
                   o_Aq);
      break;
  }
}
