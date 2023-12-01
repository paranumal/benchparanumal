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
#include "parameters.hpp"

std::string bp_t::rhsFileName(){
  std::string suffix = mesh.elementName();
  return std::string(LIBP_DIR) + "/okl/rhs" + suffix +".okl";
}

std::string bp_t::rhsKernelName(){
  std::string suffix = mesh.elementName();
  if (settings.compareSetting("AFFINE MESH", "TRUE")) {
    return "rhsAffine" + suffix;
  } else {
    return "rhs" + suffix;
  }
}

std::string bp_t::AxFileName(){
  std::string suffix = mesh.elementName();
  std::string name = "bp" + std::to_string(problemNumber);
  if (settings.compareSetting("AFFINE MESH", "TRUE")) {
    return std::string(LIBP_DIR) + "/okl/" + name + "AxAffine" + suffix + ".okl";
  } else {
    return std::string(LIBP_DIR) + "/okl/" + name + "Ax" + suffix + ".okl";
  }
}

std::string bp_t::AxKernelName(){
  std::string suffix = mesh.elementName();
  std::string name = "bp" + std::to_string(problemNumber);
  if (settings.compareSetting("AFFINE MESH", "TRUE")) {
    return name + "AxAffine" + suffix;
  } else {
    return name + "Ax" + suffix;
  }
}

void bp_t::AxTuningParams(properties_t& kernelInfo) {

  parameters_t tuningParameters;

  std::string suffix = mesh.elementName();
  std::string filename = std::string(LIBP_DIR) + "/json/bp" + std::to_string(problemNumber) + "Ax";
  if (settings.compareSetting("AFFINE MESH", "TRUE")) filename += "Affine";
  filename += suffix + ".json";

  properties_t keys;
  keys["dfloat"] = (sizeof(dfloat)==4) ? "float" : "double";
  keys["N"] = mesh.N;
  keys["mode"] = platform.device.mode();

  std::string arch = platform.device.arch();
  if (platform.device.mode()=="HIP") {
    arch = arch.substr(0,arch.find(":")); //For HIP mode, remove the stuff after the :
  }
  keys["arch"] = arch;

  std::string name = "bp" + std::to_string(problemNumber);
  if (settings.compareSetting("AFFINE MESH", "TRUE")) {
    name += "AxAffine" + suffix + ".okl";
  } else {
    name += "Ax" + suffix + ".okl";
  }

  if (settings.compareSetting("VERBOSE", "TRUE") && mesh.rank==0) {
    std::cout << "Loading Tuning Parameters, looking for match for Name:'" << name << "', keys:" << tuningParameters.toString(keys) << std::endl;
  }

  tuningParameters.load(filename, mesh.comm);

  properties_t matchProps = tuningParameters.findProperties(name, keys);

  if (settings.compareSetting("VERBOSE", "TRUE") && mesh.rank==0) {
    std::cout << "Found best match = " << tuningParameters.toString(matchProps) << std::endl;
  }

  kernelInfo["defines"] += matchProps["props"];
}

size_t bp_t::AxBytesMoved() {

  hlong Ndofs = ogs.NgatherGlobal;
  int Np = mesh.Np, cubNp = mesh.cubNp;

  bool affine = settings.compareSetting("AFFINE MESH", "TRUE");

  size_t NbytesAx=0;
  switch (problemNumber) {
    case 1:
    case 2:
      if (affine) {
        NbytesAx = Ndofs*sizeof(dfloat) //q
                + (sizeof(dfloat) // J
                +  sizeof(dlong) // localGatherElementList
                +  Np*Nfields*sizeof(dlong) // GlobalToLocal
                +  Np*Nfields*sizeof(dfloat) /*Aq*/ )*mesh.NelementsGlobal;
      } else {
        NbytesAx = Ndofs*sizeof(dfloat) //q
                + (cubNp*sizeof(dfloat) // JW
                +  sizeof(dlong) // localGatherElementList
                +  Np*Nfields*sizeof(dlong) // GlobalToLocal
                +  Np*Nfields*sizeof(dfloat) /*Aq*/ )*mesh.NelementsGlobal;
      }
      break;
    case 3:
    case 4:
      if (affine) {
        NbytesAx = Ndofs*sizeof(dfloat) //q
                  + ((mesh.dim==3 ? 7 : 4)*sizeof(dfloat) // ggeo
                  +  sizeof(dlong) // localGatherElementList
                  +  Np*Nfields*sizeof(dlong) // GlobalToLocal
                  +  Np*Nfields*sizeof(dfloat) /*Aq*/ )*mesh.NelementsGlobal;
      } else {
        NbytesAx =   Ndofs*sizeof(dfloat) //q
                  + (cubNp*(mesh.dim==3 ? 7 : 4)*sizeof(dfloat) // ggeo
                  +  sizeof(dlong) // localGatherElementList
                  +  Np*Nfields*sizeof(dlong) // GlobalToLocal
                  +  Np*Nfields*sizeof(dfloat) /*Aq*/ )*mesh.NelementsGlobal;
      }
      break;
    case 5:
    case 6:
      if (affine) {
        NbytesAx = Ndofs*sizeof(dfloat) //q
                 + ((mesh.dim==3 ? 7 : 4)*sizeof(dfloat) // ggeo
                 +  sizeof(dlong) // localGatherElementList
                 +  Np*Nfields*sizeof(dlong) // GlobalToLocal
                 +  Np*Nfields*sizeof(dfloat) /*AqL*/ )*mesh.NelementsGlobal;

      } else {
        switch (mesh.elementType) {
          case mesh_t::TRIANGLES:
          case mesh_t::TETRAHEDRA:
            NbytesAx = Ndofs*sizeof(dfloat) //q
                     + (Np*(mesh.dim==3 ? 10 : 5)*sizeof(dfloat) // vgeo
                     +  sizeof(dlong) // localGatherElementList
                     +  Np*Nfields*sizeof(dlong) // GlobalToLocal
                     +  Np*Nfields*sizeof(dfloat) /*AqL*/ )*mesh.NelementsGlobal;
            break;
          case mesh_t::QUADRILATERALS:
          case mesh_t::HEXAHEDRA:
            NbytesAx = Ndofs*sizeof(dfloat) //q
                     + (Np*(mesh.dim==3 ? 7 : 4)*sizeof(dfloat) // ggeo
                     +  sizeof(dlong) // localGatherElementList
                     +  Np*Nfields*sizeof(dlong) // GlobalToLocal
                     +  Np*Nfields*sizeof(dfloat) /*AqL*/ )*mesh.NelementsGlobal;
            break;
        }
      }
      break;
  }
  return NbytesAx;
}

size_t bp_t::AxFLOPs() {

  int Np = mesh.Np, cubNp = mesh.cubNp, Nq = mesh.Nq, cubNq = mesh.cubNq;

  bool affine = settings.compareSetting("AFFINE MESH", "TRUE");

  size_t NflopsAx=0;
  switch (problemNumber) {
    case 1:
    case 2:
      if (affine) {
        switch (mesh.elementType) {
          case mesh_t::TRIANGLES:
          case mesh_t::TETRAHEDRA:
            NflopsAx =(  2*Np*Np
                       + 1*Np)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::QUADRILATERALS:
            NflopsAx =(  4*Nq*Nq*Nq
                       + 1*Nq*Nq)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::HEXAHEDRA:
            NflopsAx =(  6*Nq*Nq*Nq*Nq
                       + 1*Nq*Nq*Nq)*Nfields*mesh.NelementsGlobal;
            break;
        }
      } else {
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
      }
      break;
    case 3:
    case 4:
      if (affine) {
        switch (mesh.elementType) {
          case mesh_t::TRIANGLES:
            NflopsAx =( 8*Np*Np
                       + 8*Np)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::TETRAHEDRA:
            NflopsAx =( 14*Np*Np
                       +14*Np)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::QUADRILATERALS:
            NflopsAx =(  16*Nq*Nq*Nq
                       + 8*Nq*Nq)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::HEXAHEDRA:
            NflopsAx =(  24*Nq*Nq*Nq*Nq
                       +17*Nq*Nq*Nq)*Nfields*mesh.NelementsGlobal;
            break;
        }
      } else {
        switch (mesh.elementType) {
          case mesh_t::TRIANGLES:
            NflopsAx =( 12*cubNp*Np
                       + 8*cubNp)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::TETRAHEDRA:
            NflopsAx =( 16*cubNp*Np
                       +17*cubNp)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::QUADRILATERALS:
            NflopsAx =(  4*cubNq*Nq*Nq
                       + 4*cubNq*cubNq*Nq
                       + 8*cubNq*cubNq*cubNq
                       + 9*cubNq*cubNq)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::HEXAHEDRA:
            NflopsAx =(  4*cubNq*Nq*Nq*Nq
                       + 4*cubNq*cubNq*Nq*Nq
                       + 4*cubNq*cubNq*cubNq*Nq
                       +12*cubNq*cubNq*cubNq*cubNq
                       +17*cubNq*cubNq*cubNq)*Nfields*mesh.NelementsGlobal;
            break;
        }
      }
      break;
    case 5:
    case 6:
      if (affine) {
        switch (mesh.elementType) {
          case mesh_t::TRIANGLES:
            NflopsAx =( 8*Np*Np
                       +8*Np)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::TETRAHEDRA:
            NflopsAx =( 14*Np*Np
                       +14*Np)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::QUADRILATERALS:
            NflopsAx =(  8*Nq*Nq*Nq
                       +12*Nq*Nq)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::HEXAHEDRA:
            NflopsAx =( 12*Nq*Nq*Nq*Nq
                       +21*Nq*Nq*Nq)*Nfields*mesh.NelementsGlobal;
            break;
        }
      } else {
        switch (mesh.elementType) {
          case mesh_t::TRIANGLES:
            NflopsAx =( 14*Np*Np
                       +14*Np)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::TETRAHEDRA:
            NflopsAx =( 20*Np*Np
                       +32*Np)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::QUADRILATERALS:
            NflopsAx =(  8*Nq*Nq*Nq
                       + 8*Nq*Nq)*Nfields*mesh.NelementsGlobal;
            break;
          case mesh_t::HEXAHEDRA:
            NflopsAx =( 12*Nq*Nq*Nq*Nq
                       +18*Nq*Nq*Nq)*Nfields*mesh.NelementsGlobal;
            break;
        }
      }
      break;
  }
  return NflopsAx;
}

size_t bp_t::CGBytesMoved(int Niter) {

  hlong Ndofs = ogs.NgatherGlobal;
  hlong NLocal = mesh.Np*mesh.Nelements*Nfields;
  hlong NGlobal = NLocal;
  mesh.comm.Allreduce(NGlobal);

  size_t NbytesAx = AxBytesMoved();

  size_t NbytesGather =  (Ndofs+1)*sizeof(dlong) //row starts
                       + NGlobal*sizeof(dlong) //local Ids
                       + NGlobal*sizeof(dfloat) //AqL
                       + Ndofs*sizeof(dfloat);

  size_t Nbytes = ( 4*Ndofs*sizeof(dfloat) + NbytesAx + NbytesGather) //first iteration
                + (11*Ndofs*sizeof(dfloat) + NbytesAx + NbytesGather)*Niter; //bytes per CG iteration

  return Nbytes;
}

size_t bp_t::CGFLOPs(int Niter) {

  hlong Ndofs = ogs.NgatherGlobal;
  hlong NLocal = mesh.Np*mesh.Nelements*Nfields;
  hlong NGlobal = NLocal;
  mesh.comm.Allreduce(NGlobal);

  size_t NflopsAx = AxFLOPs();

  size_t NflopsGather = NGlobal;

  size_t Nflops =   ( 5*Ndofs + NflopsAx + NflopsGather) //first iteration
                  + (11*Ndofs + NflopsAx + NflopsGather)*Niter; //flops per CG iteration

  return Nflops;
}

