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

#include "bp1.hpp"
#include "timer.hpp"

void bp1_t::Run(){

  dlong N = ogs.Ngather;
  dlong Nhalo = gHalo.Nhalo;

  //setup linear solver
  cg linearSolver(platform, N, Nhalo);

  hlong NgatherGlobal = ogs.NgatherGlobal;
  hlong NLocal = mesh.Np*mesh.Nelements*Nfields;
  deviceMemory<dfloat> o_rL = platform.malloc<dfloat>(NLocal);

  //create occa buffers
  dlong Nall = N+Nhalo;
  deviceMemory<dfloat> o_r = platform.malloc<dfloat>(Nall);
  deviceMemory<dfloat> o_x = platform.malloc<dfloat>(Nall);

  int verbose = settings.compareSetting("VERBOSE", "TRUE") ? 1 : 0;

  //set x =0
  platform.linAlg().set(Nall, 0.0, o_x);

  //populate rhs forcing
  dfloat zero=0.0;
  forcingKernel(mesh.Nelements,
                mesh.o_wJ,
                mesh.o_MM,
                mesh.o_x,
                mesh.o_y,
                mesh.o_z,
                zero,
                o_rL);

  // gather rhs
  ogs.Gather(o_r, o_rL, 1, ogs::Add, ogs::Trans);

  // Do warmup solve
  dfloat tol = 0.0;
  int warmupIter = 1000;
  int Niter = linearSolver.Solve(*this, o_x, o_r, tol, warmupIter, /* verbose = */ 0);

  // Re-set o_x and o_r for the timed solve
  //set x =0
  platform.linAlg().set(Nall, 0.0, o_x);

  //populate rhs forcing
  forcingKernel(mesh.Nelements,
                mesh.o_wJ,
                mesh.o_MM,
                mesh.o_x,
                mesh.o_y,
                mesh.o_z,
                zero,
                o_rL);

  // gather rhs
  ogs.Gather(o_r, o_rL, 1, ogs::Add, ogs::Trans);

  int maxIter = 100;

  //call the solver
  timePoint_t start = GlobalPlatformTime(platform);
  Niter = linearSolver.Solve(*this, o_x, o_r, tol, maxIter, verbose);
  timePoint_t end = GlobalPlatformTime(platform);
  double elapsedTime = ElapsedTime(start,end);

  int Np = mesh.Np, cubNp = mesh.cubNp, Nq = mesh.Nq, cubNq = mesh.cubNq;

  hlong NGlobal = NLocal;
  mesh.comm.Allreduce(NGlobal);

  hlong Ndofs = NgatherGlobal;

  size_t NbytesAx =   NgatherGlobal*sizeof(dfloat) //q
                    + (cubNp*sizeof(dfloat) // JW
                    + sizeof(dlong) // localGatherElementList
                    + Np*Nfields*sizeof(dlong) // GlobalToLocal
                    + Np*Nfields*sizeof(dfloat) /*Aq*/ )*mesh.NelementsGlobal;

  size_t NbytesGather =  (NgatherGlobal+1)*sizeof(dlong) //row starts
                       + NGlobal*sizeof(dlong) //local Ids
                       + NGlobal*sizeof(dfloat) //AqL
                       + NgatherGlobal*sizeof(dfloat);

  size_t Nbytes = ( 4*Ndofs*sizeof(dfloat) + NbytesAx + NbytesGather) //first iteration
                + (11*Ndofs*sizeof(dfloat) + NbytesAx + NbytesGather)*Niter; //bytes per CG iteration

  size_t NflopsAx=0;
  switch (mesh.elementType) {
    case mesh_t::TRIANGLES:
    case mesh_t::TETRAHEDRA:
      NflopsAx =(  4*cubNp*Np
                 + 1*cubNp)*Nfields*mesh.NelementsGlobal;
      break;
    case mesh_t::QUADRILATERALS:
      NflopsAx =(  4*cubNq*Nq*Nq
                 + 4*cubNq*cubNq*Nq
                 + 1*cubNq*cubNq)*Nfields*mesh.NelementsGlobal;
      break;
    case mesh_t::HEXAHEDRA:
      NflopsAx =(  4*cubNq*Nq*Nq*Nq
                 + 4*cubNq*cubNq*Nq*Nq
                 + 4*cubNq*cubNq*cubNq*Nq
                 + 1*cubNq*cubNq*cubNq)*Nfields*mesh.NelementsGlobal;
      break;
  }

  size_t NflopsGather = NGlobal;

  size_t Nflops =   ( 5*Ndofs + NflopsAx + NflopsGather) //first iteration
                  + (11*Ndofs + NflopsAx + NflopsGather)*Niter; //flops per CG iteration

  if ((mesh.rank==0)){
    printf("BP1: N=%2d, DOFs=" hlongFormat ", elapsed=%4.4f, iterations=%d, time per DOF=%1.2e, avg BW (GB/s)=%6.1f, avg GFLOPs=%6.1f, DOFs*iterations/ranks*time=%1.2e \n",
           mesh.N,
           Ndofs,
           elapsedTime,
           Niter,
           elapsedTime/(Ndofs),
           Nbytes/(1.0e9 * elapsedTime),
           Nflops/(1.0e9 * elapsedTime),
           Ndofs*((dfloat)Niter/(mesh.size*elapsedTime)));
  }

  std::string name;
  settings.getSetting("OUTPUT FILE NAME", name);
  if (name.size()>0) {

    memory<dfloat> x(N);
    memory<dfloat> xL(NLocal);
    deviceMemory<dfloat> o_xL = platform.malloc<dfloat>(NLocal);

    ogs.Scatter(o_xL, o_x, 1, ogs::NoTrans);

    // copy data back to host
    o_xL.copyTo(xL);

    // output field files
    char fname[BUFSIZ];
    sprintf(fname, "%s_%04d.vtu", name.c_str(), mesh.rank);

    mesh.PlotFields(xL, Nfields, std::string(fname));
  }
}
