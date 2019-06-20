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

typedef union intorfloat {
  int ier;
  float w;
} ierw_t;

dfloat BPCascadingWeightedInnerProduct(BP_t *BP, occa::memory &o_w, occa::memory &o_a, occa::memory &o_b){

  // use bin sorting by exponent to make the end reduction more robust
  // [ assumes that the partial reduction is ok in FP32 ]
  int Naccumulators = 256;
  int Nmantissa = 23;

  double *accumulators   = (double*) calloc(Naccumulators, sizeof(double));
  double *g_accumulators = (double*) calloc(Naccumulators, sizeof(double));

  mesh_t *mesh = BP->mesh;
  dfloat *tmp = BP->tmp;

  dlong Nblock = BP->Nblock;
  dlong Ntotal = mesh->Nelements*mesh->Np;
  
  occa::memory &o_tmp = BP->o_tmp;
  
  if(BP->options.compareArgs("DISCRETIZATION","CONTINUOUS"))
    BP->weightedInnerProduct2Kernel(Ntotal, o_w, o_a, o_b, o_tmp);
  else
    BP->innerProductKernel(Ntotal, o_a, o_b, o_tmp);
  
  o_tmp.copyTo(tmp);
  
  for(int n=0;n<Nblock;++n){
    const dfloat ftmpn = tmp[n];

    ierw_t ierw;
    ierw.w = fabs(ftmpn);
    
    int iexp = ierw.ier>>Nmantissa; // strip mantissa
    accumulators[iexp] += (double)ftmpn;
  }
  
  MPI_Allreduce(accumulators, g_accumulators, Naccumulators, MPI_DOUBLE, MPI_SUM, mesh->comm);
  
  double wab = 0.0;
  for(int n=0;n<Naccumulators;++n){ 
    wab += g_accumulators[Naccumulators-1-n]; // reverse order is important here (dominant first)
  }
  
  free(accumulators);
  free(g_accumulators);
  
  return wab;
}


dfloat BPInnerProduct(BP_t *BP, occa::memory &o_a, occa::memory &o_b){

  mesh_t *mesh = BP->mesh;
  dfloat *tmp = BP->tmp;
  dlong Nblock = BP->Nblock;
  dlong Ntotal = mesh->Nelements*mesh->Np;

  occa::memory &o_tmp = BP->o_tmp;

  BP->innerProductKernel(Ntotal, o_a, o_b, o_tmp);

  o_tmp.copyTo(tmp);

  dfloat ab = 0;
  for(dlong n=0;n<Nblock;++n){
    ab += tmp[n];
  }

  dfloat globalab = 0;
  MPI_Allreduce(&ab, &globalab, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

  return globalab;
}


template < int p_Nq > 
dfloat BPSerialWeightedNorm2Kernel(const hlong Nelements,
					 const dfloat * __restrict__ cpu_w,
					 const dfloat * __restrict__ cpu_a){
  
  
  cpu_a = (dfloat*)__builtin_assume_aligned(cpu_a, USE_OCCA_MEM_BYTE_ALIGN) ;
  cpu_w = (dfloat*)__builtin_assume_aligned(cpu_w, USE_OCCA_MEM_BYTE_ALIGN) ;

#define p_Np (p_Nq*p_Nq*p_Nq)

  dfloat wa2 = 0;

  for(hlong e=0;e<Nelements;++e){
    for(int i=0;i<p_Np;++i){
      const hlong id = e*p_Np+i;
      const dfloat ai = cpu_a[id];
      const dfloat wi = cpu_w[id];
      wa2 += ai*ai*wi;
    }
  }

  return wa2;

#undef p_Np
}

dfloat BPSerialWeightedNorm2(const int Nq, const hlong Nelements, occa::memory &o_w, occa::memory &o_a){

  const dfloat * __restrict__ cpu_a = (dfloat*)__builtin_assume_aligned(o_a.ptr(), USE_OCCA_MEM_BYTE_ALIGN) ;
  const dfloat * __restrict__ cpu_w = (dfloat*)__builtin_assume_aligned(o_w.ptr(), USE_OCCA_MEM_BYTE_ALIGN) ;

  switch(Nq){
  case  2: return BPSerialWeightedNorm2Kernel <  2 > (Nelements, cpu_w, cpu_a);
  case  3: return BPSerialWeightedNorm2Kernel <  3 > (Nelements, cpu_w, cpu_a);
  case  4: return BPSerialWeightedNorm2Kernel <  4 > (Nelements, cpu_w, cpu_a);
  case  5: return BPSerialWeightedNorm2Kernel <  5 > (Nelements, cpu_w, cpu_a);
  case  6: return BPSerialWeightedNorm2Kernel <  6 > (Nelements, cpu_w, cpu_a);
  case  7: return BPSerialWeightedNorm2Kernel <  7 > (Nelements, cpu_w, cpu_a);
  case  8: return BPSerialWeightedNorm2Kernel <  8 > (Nelements, cpu_w, cpu_a);
  case  9: return BPSerialWeightedNorm2Kernel <  9 > (Nelements, cpu_w, cpu_a);
  case 10: return BPSerialWeightedNorm2Kernel < 10 > (Nelements, cpu_w, cpu_a);
  case 11: return BPSerialWeightedNorm2Kernel < 11 > (Nelements, cpu_w, cpu_a);
  case 12: return BPSerialWeightedNorm2Kernel < 12 > (Nelements, cpu_w, cpu_a);
  }

  return -99;
}

dfloat BPWeightedNorm2(BP_t *BP, occa::memory &o_w, occa::memory &o_a){

  setupAide &options = BP->options;

  int continuous = options.compareArgs("DISCRETIZATION", "CONTINUOUS");
  int serial = options.compareArgs("THREAD MODEL", "Serial");
  int enableReductions = 1;
  options.getArgs("DEBUG ENABLE REDUCTIONS", enableReductions);

  mesh_t *mesh = BP->mesh;
  dfloat *tmp = BP->tmp;
  dlong Nblock = BP->Nblock;
  dlong Nblock2 = BP->Nblock2;
  dlong Ntotal = mesh->Nelements*mesh->Np;

  if(serial==1 && continuous==1){
    
    dfloat wa2 = BPSerialWeightedNorm2(mesh->Nq, mesh->Nelements, o_w, o_a);
    
    dfloat globalwa2 = 0;
    
    MPI_Allreduce(&wa2, &globalwa2, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
    
    return globalwa2;
  }
  
  occa::memory &o_tmp = BP->o_tmp;
  occa::memory &o_tmp2 = BP->o_tmp2;
  
  if(continuous==1)
    BP->weightedNorm2Kernel(Ntotal, o_w, o_a, o_tmp);
  else
    BP->innerProductKernel(Ntotal, o_a, o_a, o_tmp);

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


template < int p_Nq > 
dfloat BPSerialWeightedInnerProductKernel(const hlong Nelements,
						const dfloat * __restrict__ cpu_w,
						const dfloat * __restrict__ cpu_a,
						const dfloat * __restrict__ cpu_b){


  cpu_a = (dfloat*)__builtin_assume_aligned(cpu_a, USE_OCCA_MEM_BYTE_ALIGN);
  cpu_b = (dfloat*)__builtin_assume_aligned(cpu_b, USE_OCCA_MEM_BYTE_ALIGN);
  cpu_w = (dfloat*)__builtin_assume_aligned(cpu_w, USE_OCCA_MEM_BYTE_ALIGN);

#define p_Np (p_Nq*p_Nq*p_Nq)

  dfloat wab = 0;

  for(hlong e=0;e<Nelements;++e){
    for(int i=0;i<p_Np;++i){
      const hlong id = e*p_Np+i;
      const dfloat ai = cpu_a[id];
      const dfloat bi = cpu_b[id];
      const dfloat wi = cpu_w[id];
      wab += ai*bi*wi;
    }
  }

  return wab;

#undef p_Np
}

dfloat BPSerialWeightedInnerProduct(const int Nq, const hlong Nelements, occa::memory &o_w, occa::memory &o_a, occa::memory &o_b){

  const dfloat * __restrict__ cpu_a = (dfloat*)__builtin_assume_aligned(o_a.ptr(), USE_OCCA_MEM_BYTE_ALIGN) ;
  const dfloat * __restrict__ cpu_b = (dfloat*)__builtin_assume_aligned(o_b.ptr(), USE_OCCA_MEM_BYTE_ALIGN) ;
  const dfloat * __restrict__ cpu_w = (dfloat*)__builtin_assume_aligned(o_w.ptr(), USE_OCCA_MEM_BYTE_ALIGN) ;

  switch(Nq){
  case  2: return BPSerialWeightedInnerProductKernel <  2 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  case  3: return BPSerialWeightedInnerProductKernel <  3 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  case  4: return BPSerialWeightedInnerProductKernel <  4 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  case  5: return BPSerialWeightedInnerProductKernel <  5 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  case  6: return BPSerialWeightedInnerProductKernel <  6 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  case  7: return BPSerialWeightedInnerProductKernel <  7 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  case  8: return BPSerialWeightedInnerProductKernel <  8 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  case  9: return BPSerialWeightedInnerProductKernel <  9 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  case 10: return BPSerialWeightedInnerProductKernel < 10 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  case 11: return BPSerialWeightedInnerProductKernel < 11 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  case 12: return BPSerialWeightedInnerProductKernel < 12 > (Nelements, cpu_w, cpu_a, cpu_b); break;
  }

  return -99;
}

dfloat BPWeightedInnerProduct(BP_t *BP, occa::memory &o_w, occa::memory &o_a, occa::memory &o_b){

  setupAide &options = BP->options;

  int continuous = options.compareArgs("DISCRETIZATION", "CONTINUOUS");
  int serial = options.compareArgs("THREAD MODEL", "Serial");
  int enableReductions = 1;
  options.getArgs("DEBUG ENABLE REDUCTIONS", enableReductions);

  
  mesh_t *mesh = BP->mesh;
  dfloat *tmp = BP->tmp;
  dlong Nblock = BP->Nblock;
  dlong Nblock2 = BP->Nblock2;
  dlong Ntotal = mesh->Nelements*mesh->Np;

  if(serial==1 && continuous==1){
    
    dfloat wab = BPSerialWeightedInnerProduct(mesh->Nq, mesh->Nelements, o_w, o_a, o_b);
    dfloat globalwab = 0;
    
    MPI_Allreduce(&wab, &globalwab, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
    
    return globalwab;
  }
  
  occa::memory &o_tmp = BP->o_tmp;
  occa::memory &o_tmp2 = BP->o_tmp2;

  if(continuous==1)
    BP->weightedInnerProduct2Kernel(Ntotal, o_w, o_a, o_b, o_tmp);
  else
    BP->innerProductKernel(Ntotal, o_a, o_b, o_tmp);

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



void BPScaledAdd(BP_t *BP, dfloat alpha, occa::memory &o_a, dfloat beta, occa::memory &o_b){

  mesh_t *mesh = BP->mesh;

  setupAide &options = BP->options;
  
  int continuous = options.compareArgs("DISCRETIZATION", "CONTINUOUS");
  
  dlong Ntotal = mesh->Nelements*mesh->Np;

  // b[n] = alpha*a[n] + beta*b[n] n\in [0,Ntotal)
  
  // if not Serial
  BP->scaledAddKernel(Ntotal, alpha, o_a, beta, o_b);
}
