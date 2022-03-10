/*

The MIT License (MIT)

Copyright (c) 2020 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#ifndef BP4_HPP
#define BP4_HPP 1

#include "core.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "linearSolver.hpp"

using namespace libp;

class bp4Settings_t: public settings_t {
public:
  bp4Settings_t(const int argc, char** argv, comm_t _comm);
  void report();
};

class bp4_t: public solver_t {
public:
  mesh_t mesh;

  ogs::ogs_t ogs;
  ogs::halo_t gHalo;

  memory<dlong> GlobalToLocal;
  deviceMemory<dlong> o_GlobalToLocal;

  int Nfields;
  dfloat lambda;

  deviceMemory<dfloat> o_AqL;

  kernel_t operatorKernel;
  kernel_t forcingKernel;

  bp4_t() = default;
  bp4_t(platform_t& _platform, settings_t& _settings,
        mesh_t& _mesh, dfloat _lambda) {
    Setup(_platform, _settings, _mesh, _lambda);
  }

  //setup
  void Setup(platform_t& _platform, settings_t& _settings,
             mesh_t& _mesh, dfloat _lambda);

  void Run();

  void Operator(deviceMemory<dfloat>& o_q, deviceMemory<dfloat>& o_Aq);
};


#endif

