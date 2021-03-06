/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#include "BP.hpp"

dfloat BPNorm2(BP_t *BP, dlong Ntotal, dlong offset, occa::memory &o_a){

  mesh_t *mesh = BP->mesh;
  
  setupAide &options = BP->options;

  dfloat *tmp = BP->tmp;
  dlong Nblock = BP->Nblock;
  dlong Nblock2 = BP->Nblock2;

  occa::memory &o_tmp = BP->o_tmp;
  occa::memory &o_tmp2 = BP->o_tmp2;

  if(BP->Nfields==1)
    BP->norm2Kernel(Ntotal, o_a, o_tmp);
  else
    BP->multipleNorm2Kernel(Ntotal, offset, o_a, o_tmp);

  /* add a second sweep if Nblock>Ncutoff */
  dlong Ncutoff = 100;
  dlong Nfinal;
  if(Nblock>Ncutoff){

    mesh->sumKernel(Nblock, o_tmp, o_tmp2);

    o_tmp2.copyTo(tmp);

    Nfinal = Nblock2;
	
  }
  else{
    o_tmp.copyTo(tmp);
    
    Nfinal = Nblock;

  }    

  dfloat wa2 = 0;
  for(dlong n=0;n<Nfinal;++n){
    wa2 += tmp[n];
  }

  dfloat globalwa2 = 0;
  MPI_Allreduce(&wa2, &globalwa2, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

  return globalwa2;
}


dfloat BPWeightedNorm2(BP_t *BP, occa::memory &o_w, occa::memory &o_a){

  setupAide &options = BP->options;

  mesh_t *mesh = BP->mesh;
  dfloat *tmp = BP->tmp;
  dlong Nblock = BP->Nblock;
  dlong Nblock2 = BP->Nblock2;
  dlong Ntotal = mesh->Nelements*mesh->Np;

  occa::memory &o_tmp = BP->o_tmp;
  occa::memory &o_tmp2 = BP->o_tmp2;

  if(BP->Nfields==1)
    BP->weightedNorm2Kernel(Ntotal, o_w, o_a, o_tmp);
  else
    BP->weightedMultipleNorm2Kernel(Ntotal, Ntotal, o_w, o_a, o_tmp);

  /* add a second sweep if Nblock>Ncutoff */
  dlong Ncutoff = 100;
  dlong Nfinal;
  if(Nblock>Ncutoff){

    mesh->sumKernel(Nblock, o_tmp, o_tmp2);

    o_tmp2.copyTo(tmp);

    Nfinal = Nblock2;
	
  }
  else{
    o_tmp.copyTo(tmp);
    
    Nfinal = Nblock;

  }    

  dfloat wa2 = 0;
  for(dlong n=0;n<Nfinal;++n){
    wa2 += tmp[n];
  }

  dfloat globalwa2 = 0;
  MPI_Allreduce(&wa2, &globalwa2, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

  return globalwa2;
}

dfloat BPWeightedInnerProduct(BP_t *BP, occa::memory &o_w, occa::memory &o_a, occa::memory &o_b){

  setupAide &options = BP->options;

  mesh_t *mesh = BP->mesh;
  dfloat *tmp = BP->tmp;
  dlong Nblock = BP->Nblock;
  dlong Nblock2 = BP->Nblock2;
  dlong Ntotal = mesh->Nelements*mesh->Np;

  occa::memory &o_tmp = BP->o_tmp;
  occa::memory &o_tmp2 = BP->o_tmp2;

  if(BP->Nfields==1)
    BP->weightedInnerProduct2Kernel(Ntotal, o_w, o_a, o_b, o_tmp);
  else
    BP->weightedMultipleInnerProduct2Kernel(Ntotal, Ntotal, o_w, o_a, o_b, o_tmp);

  /* add a second sweep if Nblock>Ncutoff */
  dlong Ncutoff = 100;
  dlong Nfinal;
  if(Nblock>Ncutoff){

    mesh->sumKernel(Nblock, o_tmp, o_tmp2);

    o_tmp2.copyTo(tmp);

    Nfinal = Nblock2;
	
  }
  else{
    o_tmp.copyTo(tmp);
    
    Nfinal = Nblock;

  }    

  dfloat wab = 0;
  for(dlong n=0;n<Nfinal;++n){
    wab += tmp[n];
  }

  dfloat globalwab = 0;
  MPI_Allreduce(&wab, &globalwab, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

  return globalwab;
}

dfloat BPInnerProduct(BP_t *BP, dlong Ntotal, dlong offset, occa::memory &o_a, occa::memory &o_b){

  setupAide &options = BP->options;

  mesh_t *mesh = BP->mesh;
  dfloat *tmp = BP->tmp;
  dlong Nblock = BP->Nblock;
  dlong Nblock2 = BP->Nblock2;

  occa::memory &o_tmp = BP->o_tmp;
  occa::memory &o_tmp2 = BP->o_tmp2;

  if(BP->Nfields==1)
    BP->innerProduct2Kernel(Ntotal, o_a, o_b, o_tmp);
  else
    BP->multipleInnerProduct2Kernel(Ntotal, offset, o_a, o_b, o_tmp);

  /* add a second sweep if Nblock>Ncutoff */
  dlong Ncutoff = 100;
  dlong Nfinal;
  if(Nblock>Ncutoff){

    mesh->sumKernel(Nblock, o_tmp, o_tmp2);

    o_tmp2.copyTo(tmp);

    Nfinal = Nblock2;
	
  }
  else{
    o_tmp.copyTo(tmp);
    
    Nfinal = Nblock;

  }    

  dfloat wab = 0;
  for(dlong n=0;n<Nfinal;++n){
    wab += tmp[n];
  }

  dfloat globalwab = 0;
  MPI_Allreduce(&wab, &globalwab, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

  return globalwab;
}




// b[n] = alpha*a[n] + beta*b[n] n\in [0,Ntotal)
void BPScaledAdd(BP_t *BP, dfloat alpha, occa::memory &o_a, dfloat beta, occa::memory &o_b){

  mesh_t *mesh = BP->mesh;
  
  dlong Ntotal = mesh->Nelements*mesh->Np*BP->Nfields;

  BP->scaledAddKernel(Ntotal, alpha, o_a, beta, o_b);
}

#if 0
dfloat BPAtomicInnerProduct(BP_t *BP, dlong N, occa::memory &o_a, occa::memory &o_b){

  BP->o_zeroAtomic.copyTo(BP->o_tmpAtomic);
  
  BP->vecAtomicInnerProductKernel(N, o_a, o_b, BP->o_tmpAtomic);

  BP->o_tmpAtomic.copyTo(BP->tmpAtomic);
  
  dfloat globaladotb = 0;

  MPI_Allreduce(BP->tmpAtomic, &globaladotb, 1, MPI_DFLOAT, MPI_SUM, BP->mesh->comm);

  return globaladotb;
}
#endif
