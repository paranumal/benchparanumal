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

#include "bk1.hpp"

void bk1_t::Setup(platform_t& _platform, settings_t& _settings,
                  mesh_t& _mesh){

  platform = _platform;
  settings = _settings;
  mesh = _mesh;

  Nfields = 1;

  //Trigger JIT kernel builds
  ogs::InitializeKernels(platform, ogs::Dfloat, ogs::Add);

  mesh.CubatureSetup();
  ogs = mesh.MaskedGatherScatterSetup(Nfields); //make masked ogs

  gHalo.SetupFromGather(ogs);

  GlobalToLocal.malloc(mesh.Nelements*mesh.Np*Nfields);
  ogs.SetupGlobalToLocalMapping(GlobalToLocal);

  o_GlobalToLocal = platform.malloc<dlong>(GlobalToLocal);

  // OCCA build stuff
  properties_t kernelInfo = mesh.props; //copy base occa properties

  kernelInfo["defines/" "p_Nfields"]= Nfields;

  // set kernel name suffix
  char *suffix;
  if(mesh.elementType==mesh_t::TRIANGLES)
    suffix = strdup("Tri2D");
  else if(mesh.elementType==mesh_t::QUADRILATERALS)
    suffix = strdup("Quad2D");
  else if(mesh.elementType==mesh_t::TETRAHEDRA)
    suffix = strdup("Tet3D");
  else //if(mesh.elementType==mesh_t::HEXAHEDRA)
    suffix = strdup("Hex3D");

  char fileName[BUFSIZ], kernelName[BUFSIZ];

  // Ax kernel
  sprintf(fileName,  LIBP_DIR "/okl/bp1Ax%s.okl", suffix);
  sprintf(kernelName, "bp1Ax%s", suffix);

  operatorKernel = platform.buildKernel(fileName, kernelName,
                                   kernelInfo);

  sprintf(fileName, LIBP_DIR "/okl/rhs%s.okl", suffix);
  sprintf(kernelName, "rhs%s", suffix);
  forcingKernel = platform.buildKernel(fileName, kernelName,
                                   kernelInfo);
}
