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

#include "bk5.hpp"
#include "timer.hpp"

void bk5_t::Run(){

  //create occa buffers
  dlong Nall = mesh.Np*mesh.Nelements*Nfields;
  dlong Ngather = ogs.Ngather;
  deviceMemory<dfloat> o_q  = platform.malloc<dfloat>(Ngather);
  deviceMemory<dfloat> o_Aq = platform.malloc<dfloat>(Nall);

  //populate x with a typical rhs (use Aq as temp storage)
  forcingKernel(mesh.Nelements,
                mesh.o_wJ,
                mesh.o_gllw,
                mesh.o_MM,
                mesh.o_x,
                mesh.o_y,
                mesh.o_z,
                lambda,
                o_Aq);

  // gather rhs
  ogs.Gather(o_q, o_Aq, 1, ogs::Add, ogs::Trans);

  // int verbose = settings.compareSetting("VERBOSE", "TRUE") ? 1 : 0;

  int Nrepeats = 1000000;

  int Ntests = 50;

  for(int n=0;n<5;++n){ //warmup
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
  }

  for(int rep=0;rep<Nrepeats;++rep){
  
  timePoint_t start = GlobalPlatformTime(platform);
  for(int n=0;n<Ntests;++n){
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
  }
  timePoint_t end = GlobalPlatformTime(platform);
  double elapsedTime = ElapsedTime(start,end)/Ntests;

  int Np = mesh.Np, Nq = mesh.Nq;

  hlong Ndofs = ogs.NgatherGlobal;

  bool affine = settings.compareSetting("AFFINE MESH", "TRUE");

  size_t Nbytes = 0;
  if (affine) {

    Nbytes = Ndofs*sizeof(dfloat) //q
             + ((mesh.dim==3 ? 7 : 4)*sizeof(dfloat) // ggeo
             +  sizeof(dlong) // localGatherElementList
             +  Np*Nfields*sizeof(dlong) // GlobalToLocal
             +  Np*Nfields*sizeof(dfloat) /*AqL*/ )*mesh.NelementsGlobal;

  } else {
    switch (mesh.elementType) {
      case mesh_t::TRIANGLES:
      case mesh_t::TETRAHEDRA:
        Nbytes = Ndofs*sizeof(dfloat) //q
                 + (Np*(mesh.dim==3 ? 10 : 5)*sizeof(dfloat) // vgeo
                 +  sizeof(dlong) // localGatherElementList
                 +  Np*Nfields*sizeof(dlong) // GlobalToLocal
                 +  Np*Nfields*sizeof(dfloat) /*AqL*/ )*mesh.NelementsGlobal;
        break;
      case mesh_t::QUADRILATERALS:
      case mesh_t::HEXAHEDRA:
        Nbytes = Ndofs*sizeof(dfloat) //q
                 + (Np*(mesh.dim==3 ? 7 : 4)*sizeof(dfloat) // ggeo
                 +  sizeof(dlong) // localGatherElementList
                 +  Np*Nfields*sizeof(dlong) // GlobalToLocal
                 +  Np*Nfields*sizeof(dfloat) /*AqL*/ )*mesh.NelementsGlobal;
        break;
    }
  }

  size_t Nflops=0;
  if (affine) {
    switch (mesh.elementType) {
      case mesh_t::TRIANGLES:
        Nflops =( 8*Np*Np
                   +8*Np)*Nfields*mesh.NelementsGlobal;
        break;
      case mesh_t::TETRAHEDRA:
        Nflops =( 14*Np*Np
                   +14*Np)*Nfields*mesh.NelementsGlobal;
        break;
      case mesh_t::QUADRILATERALS:
        Nflops =(  8*Nq*Nq*Nq
                   +12*Nq*Nq)*Nfields*mesh.NelementsGlobal;
        break;
      case mesh_t::HEXAHEDRA:
        Nflops =( 12*Nq*Nq*Nq*Nq
                   +21*Nq*Nq*Nq)*Nfields*mesh.NelementsGlobal;
        break;
    }
  } else {
    switch (mesh.elementType) {
      case mesh_t::TRIANGLES:
        Nflops =( 14*Np*Np
                   +14*Np)*Nfields*mesh.NelementsGlobal;
        break;
      case mesh_t::TETRAHEDRA:
        Nflops =( 20*Np*Np
                   +32*Np)*Nfields*mesh.NelementsGlobal;
        break;
      case mesh_t::QUADRILATERALS:
        Nflops =(  8*Nq*Nq*Nq
                   + 8*Nq*Nq)*Nfields*mesh.NelementsGlobal;
        break;
      case mesh_t::HEXAHEDRA:
        Nflops =( 12*Nq*Nq*Nq*Nq
                   +18*Nq*Nq*Nq)*Nfields*mesh.NelementsGlobal;
        break;
    }
  }

  if (mesh.rank==0){
    std::string suffix("Element=");
    switch (mesh.elementType) {
      case mesh_t::TRIANGLES:
        suffix += "Tri";
        break;
      case mesh_t::TETRAHEDRA:
        suffix += "Tet";
        break;
      case mesh_t::QUADRILATERALS:
        suffix += "Quad";
        break;
      case mesh_t::HEXAHEDRA:
        suffix += "Hex";
        break;
    }
    if (affine) suffix += ", Affine";
    
    printf("%06d, %02d, " hlongFormat ", %6.4e, %6.4e, %6.4e, %6.4e, %6.4e; %%%% test, N, Ndofs, elapsed, throughput (dofs/s), gbytes/s, gflops/s, %s\n",
	   rep,
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
}
