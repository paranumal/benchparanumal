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

void bp_t::Operator(deviceMemory<dfloat> &o_q, deviceMemory<dfloat> &o_Aq){
  switch (problemNumber) {
    case 1:
    case 2:
      FEMMassOperator(o_q, o_Aq); break;
    case 3:
    case 4:
      FEMHelmholtzOperator(o_q, o_Aq); break;
    case 5:
    case 6:
      SEMHelmholtzOperator(o_q, o_Aq); break;
  }
}

void bp_t::FEMMassOperator(deviceMemory<dfloat> &o_q, deviceMemory<dfloat> &o_Aq){

  gHalo.ExchangeStart(o_q, 1);

  if(mesh.NlocalGatherElements/2){
    operatorKernel(mesh.NlocalGatherElements/2,
                   mesh.o_localGatherElementList,
                   o_GlobalToLocal,
                   mesh.o_cubwJ,
                   mesh.o_cubInterp,
                   mesh.o_MM,
                   o_q, o_AqL);
  }

  // finalize halo exchange
  gHalo.ExchangeFinish(o_q, 1);

  if(mesh.NglobalGatherElements) {
    operatorKernel(mesh.NglobalGatherElements,
                   mesh.o_globalGatherElementList,
                   o_GlobalToLocal,
                   mesh.o_cubwJ,
                   mesh.o_cubInterp,
                   mesh.o_MM,
                   o_q, o_AqL);
  }

  //gather result to Aq
  ogs.GatherStart(o_Aq, o_AqL, 1, ogs::Add, ogs::Trans);

  if((mesh.NlocalGatherElements+1)/2){
    operatorKernel((mesh.NlocalGatherElements+1)/2,
                   mesh.o_localGatherElementList+(mesh.NlocalGatherElements/2),
                   o_GlobalToLocal,
                   mesh.o_cubwJ,
                   mesh.o_cubInterp,
                   mesh.o_MM,
                   o_q, o_AqL);
  }

  ogs.GatherFinish(o_Aq, o_AqL, 1, ogs::Add, ogs::Trans);
}

void bp_t::FEMHelmholtzOperator(deviceMemory<dfloat> &o_q, deviceMemory<dfloat> &o_Aq){

  gHalo.ExchangeStart(o_q, 1);

  if(mesh.NlocalGatherElements/2){
    operatorKernel(mesh.NlocalGatherElements/2,
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
                   o_q, o_AqL);
  }

  // finalize halo exchange
  gHalo.ExchangeFinish(o_q, 1);

  if(mesh.NglobalGatherElements) {
    operatorKernel(mesh.NglobalGatherElements,
                   mesh.o_globalGatherElementList,
                   o_GlobalToLocal,
                   mesh.o_cubwJ,
                   mesh.o_cubggeo,
                   mesh.o_cubD,
                   mesh.o_cubInterp,
                   mesh.o_invV,
                   mesh.o_S,
                   mesh.o_MM,
                   lambda,
                   o_q, o_AqL);
  }

  //gather result to Aq
  ogs.GatherStart(o_Aq, o_AqL, 1, ogs::Add, ogs::Trans);

  if((mesh.NlocalGatherElements+1)/2){
    operatorKernel((mesh.NlocalGatherElements+1)/2,
                   mesh.o_localGatherElementList+(mesh.NlocalGatherElements/2),
                   o_GlobalToLocal,
                   mesh.o_cubwJ,
                   mesh.o_cubggeo,
                   mesh.o_cubD,
                   mesh.o_cubInterp,
                   mesh.o_invV,
                   mesh.o_S,
                   mesh.o_MM,
                   lambda,
                   o_q, o_AqL);
  }

  ogs.GatherFinish(o_Aq, o_AqL, 1, ogs::Add, ogs::Trans);
}

void bp_t::SEMHelmholtzOperator(deviceMemory<dfloat> &o_q, deviceMemory<dfloat> &o_Aq){

  gHalo.ExchangeStart(o_q, 1);

  if(mesh.NlocalGatherElements/2){
    operatorKernel(mesh.NlocalGatherElements/2,
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
                   o_q, o_AqL);
  }

  // finalize halo exchange
  gHalo.ExchangeFinish(o_q, 1);

  if(mesh.NglobalGatherElements) {
    operatorKernel(mesh.NglobalGatherElements,
                   mesh.o_globalGatherElementList,
                   o_GlobalToLocal,
                   mesh.o_wJ,
                   mesh.o_vgeo,
                   mesh.o_ggeo,
                   mesh.o_gllw,
                   mesh.o_D,
                   mesh.o_S,
                   mesh.o_MM,
                   lambda,
                   o_q, o_AqL);
  }

  //gather result to Aq
  ogs.GatherStart(o_Aq, o_AqL, 1, ogs::Add, ogs::Trans);

  if((mesh.NlocalGatherElements+1)/2){
    operatorKernel((mesh.NlocalGatherElements+1)/2,
                   mesh.o_localGatherElementList+(mesh.NlocalGatherElements/2),
                   o_GlobalToLocal,
                   mesh.o_wJ,
                   mesh.o_vgeo,
                   mesh.o_ggeo,
                   mesh.o_gllw,
                   mesh.o_D,
                   mesh.o_S,
                   mesh.o_MM,
                   lambda,
                   o_q, o_AqL);
  }

  ogs.GatherFinish(o_Aq, o_AqL, 1, ogs::Add, ogs::Trans);
}
