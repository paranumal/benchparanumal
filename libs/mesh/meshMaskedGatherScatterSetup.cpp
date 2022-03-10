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

#include "mesh.hpp"

namespace libp {

ogs::ogs_t mesh_t::MaskedGatherScatterSetup(int Nfields){

  //make a masked version of the global id numbering
  memory<hlong> maskedGlobalIds(Nelements*Np*Nfields);
  for (dlong e=0;e<Nelements;e++) {
    for (int n=0;n<Np;n++) {
      hlong id = globalIds[e*Np+n];
      for (int f=0;f<Nfields;f++) {
        maskedGlobalIds[e*Np*Nfields+f*Np+n] = id*Nfields+f;
      }
    }
  }

  //The mesh is just a structured brick, so we don't have to worry about
  // singleton corners or edges that are on the boundary. Just
  // check the faces through EToE
  for (dlong e=0;e<Nelements;e++) {
    for (int f=0;f<Nfaces;f++) {
      if (EToE[f+e*Nfaces]<0) { //unconnected faces are boundary
        for (int n=0;n<Nfp;n++) {
          const int fid = faceNodes[n+f*Nfp];
          for (int nf=0;nf<Nfields;nf++) {
            maskedGlobalIds[e*Np*Nfields+nf*Np+fid] = 0; //mask
          }
        }
      }
    }
  }

  //use the masked ids to make another gs handle (signed so the gather is defined)
  bool verbose = settings.compareSetting("VERBOSE", "TRUE");
  bool unique = true; //flag a unique node in every gather node
  ogs::ogs_t ogsMasked;
  ogsMasked.Setup(Nelements*Np*Nfields, maskedGlobalIds,
                  comm, ogs::Signed, ogs::Auto,
                  unique, verbose, platform);

  // gHalo = new ogs::halo_t(platform);
  // gHalo->SetupFromGather(*ogsMasked);

  // GlobalToLocal = (dlong *) malloc(Nelements*Np*Nfields*sizeof(dlong));
  // ogsMasked->SetupGlobalToLocalMapping(GlobalToLocal);

  // o_GlobalToLocal = platform.malloc(Nelements*Np*Nfields*sizeof(dlong), GlobalToLocal);

  return ogsMasked;
}

} //namespace libp
