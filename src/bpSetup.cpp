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

void bp_t::Setup(platform_t& _platform,
                 settings_t& _settings,
                 mesh_t& _mesh){

  platform = _platform;
  settings = _settings;
  mesh = _mesh;

  settings.getSetting("BENCHMARK PROBLEM", problemNumber);

  if (problemNumber==1 || problemNumber==3 || problemNumber==5) {
    Nfields = 1;
  } else {
    Nfields = mesh.dim;
  }

  lambda = 1.0;

  //Trigger JIT kernel builds
  ogs::InitializeKernels(platform, ogs::Dfloat, ogs::Add);

  if (settings.compareSetting("AFFINE MESH", "FALSE")) {
    if (problemNumber==1 || problemNumber==2 ||
        problemNumber==3 || problemNumber==4 ) {
      mesh.CubatureSetup();
    }
  }

  ogs = mesh.MaskedGatherScatterSetup(Nfields); //make masked ogs

  gHalo.SetupFromGather(ogs);

  GlobalToLocal.malloc(mesh.Nelements*mesh.Np*Nfields);
  ogs.SetupGlobalToLocalMapping(GlobalToLocal);

  o_GlobalToLocal = platform.malloc<dlong>(GlobalToLocal);

  //tmp local storage buffer for Ax op
  o_AqL = platform.malloc<dfloat>(mesh.Np*mesh.Nelements*Nfields);
}
