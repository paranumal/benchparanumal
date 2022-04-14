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

#include "bk3.hpp"
#include "timer.hpp"

void bk3_t::Run(){

  //create occa buffers
  dlong Nall = mesh.Np*mesh.Nelements*Nfields;
  dlong Ngather = ogs.Ngather;
  deviceMemory<dfloat> o_q  = platform.malloc<dfloat>(Ngather);
  deviceMemory<dfloat> o_Aq = platform.malloc<dfloat>(Nall);

  //populate x with a typical rhs (use Aq as temp storage)
  forcingKernel(mesh.Nelements,
                mesh.o_wJ,
                mesh.o_MM,
                mesh.o_x,
                mesh.o_y,
                mesh.o_z,
                lambda,
                o_Aq);

  // gather rhs
  ogs.Gather(o_q, o_Aq, 1, ogs::Add, ogs::Trans);

  int Ntests = 50;

  for(int n=0;n<5;++n){ //warmup
    operatorKernel(mesh.NlocalGatherElements,
                   mesh.o_localGatherElementList,
                   o_GlobalToLocal,
                   mesh.o_cubwJ,
                   mesh.o_cubggeo,
                   mesh.o_cubD,
                   mesh.o_cubInterp,
                   mesh.o_S,
                   mesh.o_MM,
                   lambda,
                   o_q,
                   o_Aq);
  }

  timePoint_t start = GlobalPlatformTime(platform);
  for(int n=0;n<Ntests;++n){
    operatorKernel(mesh.NlocalGatherElements,
                   mesh.o_localGatherElementList,
                   o_GlobalToLocal,
                   mesh.o_cubwJ,
                   mesh.o_cubggeo,
                   mesh.o_cubD,
                   mesh.o_cubInterp,
                   mesh.o_S,
                   mesh.o_MM,
                   lambda,
                   o_q,
                   o_Aq);
  }
  timePoint_t end = GlobalPlatformTime(platform);
  double elapsedTime = ElapsedTime(start,end)/Ntests;

  int Np = mesh.Np, cubNp = mesh.cubNp, Nq = mesh.Nq, cubNq = mesh.cubNq;

  hlong Ndofs = ogs.NgatherGlobal;

  size_t Nbytes =   Ndofs*sizeof(dfloat) //q
                  + (cubNp*(mesh.dim==3 ? 7 : 4)*sizeof(dfloat) // ggeo
                  + sizeof(dlong) // localGatherElementList
                  + Np*Nfields*sizeof(dlong) // GlobalToLocal
                  + Np*Nfields*sizeof(dfloat) /*Aq*/ )*mesh.NelementsGlobal;

  double Nflops=0;
  if (mesh.dim==3)
    Nflops =((double)  4*cubNq*Nq*Nq*Nq
             + 4*cubNq*cubNq*Nq*Nq
             + 4*cubNq*cubNq*cubNq*Nq
             +12*cubNq*cubNq*cubNq*cubNq
             +17*cubNq*cubNq*cubNq)*Nfields*mesh.NelementsGlobal;
  else
    Nflops =(  4*cubNq*Nq*Nq
             + 4*cubNq*cubNq*Nq
             + 8*cubNq*cubNq*cubNq
             + 9*cubNq*cubNq)*Nfields*mesh.NelementsGlobal;

  if (mesh.rank==0){
    printf("BK3: N=%2d, DOFs=" hlongFormat ", elapsed=%4.4f, time per DOF=%1.2e, avg BW (GB/s)=%6.1f, avg GFLOPs=%6.1f, DOFs/ranks*time=%1.2e \n",
           mesh.N,
           Ndofs,
           elapsedTime,
           elapsedTime/(Ndofs),
           Nbytes/(1.0e9 * elapsedTime),
           Nflops/(1.0e9 * elapsedTime),
           Ndofs/(mesh.size*elapsedTime));
  }
}
