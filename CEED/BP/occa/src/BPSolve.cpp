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

int BPSolve(BP_t *BP, dfloat lambda, dfloat tol, occa::memory &o_r, occa::memory &o_x){
  
  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;

  int Niter = 0;
  int maxIter = 1000; 

  if(BP->allNeumann) // zero mean of RHS
    BPZeroMean(BP, o_r);
  
  options.getArgs("MAXIMUM ITERATIONS", maxIter);

  options.getArgs("SOLVER TOLERANCE", tol);
  
  Niter = BPPCG(BP, lambda, o_r, o_x, tol, maxIter);

  if(BP->allNeumann) // zero mean of RHS
    BPZeroMean(BP, o_x);
  
  return Niter;

}

// FROM NEKBONE: not appropriate since it assumes zero initial data
int BPPCG(BP_t* BP, dfloat lambda, 
	  occa::memory &o_r, occa::memory &o_x, 
	  const dfloat tol, const int MAXIT){
  
  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;

  int fixedIterationCountFlag = 0;
  int flexible = options.compareArgs("KRYLOV SOLVER", "FLEXIBLE");
  int verbose = options.compareArgs("VERBOSE", "TRUE");
  
  if(options.compareArgs("FIXED ITERATION COUNT", "TRUE"))
    fixedIterationCountFlag = 1;
  
  // register scalars
  dfloat rdotz1 = 0;
  dfloat rdotz2 = 0;


  // now initialized
  dfloat alpha = 0, beta = 0, pAp = 0;
  
  /*aux variables */
  occa::memory &o_p  = BP->o_p;
  occa::memory &o_z  = BP->o_z;
  occa::memory &o_Ap = BP->o_Ap;
  occa::memory &o_Ax = BP->o_Ax;

  pAp = 0;
  rdotz1 = 1;

  dfloat rdotr0;

  // compute A*x
  BPOperator(BP, lambda, o_x, BP->o_Ax, dfloatString);
  
  // subtract r = b - A*x
  BPScaledAdd(BP, -1.f, o_Ax, 1.f, o_r);

  rdotr0 = BPWeightedNorm2(BP, BP->o_invDegree, o_r);

  dfloat TOL =  mymax(tol*tol*rdotr0,tol*tol);
  
  int iter;
  for(iter=1;iter<=MAXIT;++iter){

    // z = Precon^{-1} r
    // need to copy r to z
    o_r.copyTo(o_z);

    rdotz2 = rdotz1;

    // r.z
    rdotz1 = BPWeightedInnerProduct(BP, BP->o_invDegree, o_r, o_z); 
    
    if(flexible){
      dfloat zdotAp = BPWeightedInnerProduct(BP, BP->o_invDegree, o_z, o_Ap);  
      
      beta = -alpha*zdotAp/rdotz2;
    }
    else{
      beta = (iter==1) ? 0:rdotz1/rdotz2;
    }
    
    // p = z + beta*p
    BPScaledAdd(BP, 1.f, o_z, beta, o_p);

#if 0
    // Ap
    BPOperator(BP, lambda, o_p, o_Ap, dfloatString); 
   
    // dot(p,A*p)
    pAp =  BPWeightedInnerProduct(BP, BP->o_invDegree, o_p, o_Ap);
#else
    // Ap and p.Ap
    pAp = BPOperatorDot(BP, lambda, o_p, o_Ap, dfloatString); 
#endif
    
    alpha = rdotz1/pAp;

    //  x <= x + alpha*p
    //  r <= r - alpha*A*p
    //  dot(r,r)
    
    dfloat rdotr = BPUpdatePCG(BP, o_p, o_Ap, alpha, o_x, o_r);
	
    if (verbose&&(mesh->rank==0)) {

      if(rdotr<0)
	printf("WARNING CG: rdotr = %17.15lf\n", rdotr);
      
      printf("CG: it %d r norm %12.12le alpha = %le \n", iter, sqrt(rdotr), alpha);    
    }
    
    if(rdotr<=TOL && !fixedIterationCountFlag) break;
    
  }

  return iter;
}


void BPZeroMean(BP_t *BP, occa::memory &o_q){

  dfloat qmeanLocal;
  dfloat qmeanGlobal;
  
  dlong Nblock = BP->Nblock;
  dfloat *tmp = BP->tmp;
  mesh_t *mesh = BP->mesh;

  occa::memory &o_tmp = BP->o_tmp;

  // this is a C0 thing [ assume GS previously applied to o_q ]
#define USE_WEIGHTED 1
  
#if USE_WEIGHTED==1
  BP->innerProductKernel(mesh->Nelements*mesh->Np, BP->o_invDegree, o_q, o_tmp);
#else
  mesh->sumKernel(mesh->Nelements*mesh->Np, o_q, o_tmp);
#endif
  
  o_tmp.copyTo(tmp);

  // finish reduction
  qmeanLocal = 0;
  for(dlong n=0;n<Nblock;++n)
    qmeanLocal += tmp[n];

  // globalize reduction
  MPI_Allreduce(&qmeanLocal, &qmeanGlobal, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

  // normalize
#if USE_WEIGHTED==1
  qmeanGlobal *= BP->nullProjectWeightGlobal;
#else
  qmeanGlobal /= ((dfloat) BP->NelementsGlobal*(dfloat)mesh->Np);
#endif
  
  // q[n] = q[n] - qmeanGlobal
  mesh->addScalarKernel(mesh->Nelements*mesh->Np, -qmeanGlobal, o_q);
}


template < int p_Nq >
dfloat BPSerialUpdatePCGKernel(const hlong Nelements,
				     const dfloat * __restrict__ cpu_invDegree,
				     const dfloat * __restrict__ cpu_p,
				     const dfloat * __restrict__ cpu_Ap,
				     const dfloat alpha,
				     dfloat * __restrict__ cpu_x,
				     dfloat * __restrict__ cpu_r){

#define p_Np (p_Nq*p_Nq*p_Nq)

  cpu_p  = (dfloat*)__builtin_assume_aligned(cpu_p,  USE_OCCA_MEM_BYTE_ALIGN) ;
  cpu_Ap = (dfloat*)__builtin_assume_aligned(cpu_Ap, USE_OCCA_MEM_BYTE_ALIGN) ;
  cpu_x  = (dfloat*)__builtin_assume_aligned(cpu_x,  USE_OCCA_MEM_BYTE_ALIGN) ;
  cpu_r  = (dfloat*)__builtin_assume_aligned(cpu_r,  USE_OCCA_MEM_BYTE_ALIGN) ;

  cpu_invDegree = (dfloat*)__builtin_assume_aligned(cpu_invDegree,  USE_OCCA_MEM_BYTE_ALIGN) ;
  
  dfloat rdotr = 0;
  
  cpu_p = (dfloat*)__builtin_assume_aligned(cpu_p, USE_OCCA_MEM_BYTE_ALIGN) ;

  for(hlong e=0;e<Nelements;++e){
    for(int i=0;i<p_Np;++i){
      const hlong n = e*p_Np+i;
      cpu_x[n] += alpha*cpu_p[n];

      const dfloat rn = cpu_r[n] - alpha*cpu_Ap[n];
      rdotr += rn*rn*cpu_invDegree[n];
      cpu_r[n] = rn;
    }
  }

#undef p_Np
  
  return rdotr;
}
				     
dfloat BPSerialUpdatePCG(const int Nq, const hlong Nelements,
			       occa::memory &o_invDegree, occa::memory &o_p, occa::memory &o_Ap, const dfloat alpha,
			       occa::memory &o_x, occa::memory &o_r){

  const dfloat * __restrict__ cpu_p  = (dfloat*)__builtin_assume_aligned(o_p.ptr(), USE_OCCA_MEM_BYTE_ALIGN) ;
  const dfloat * __restrict__ cpu_Ap = (dfloat*)__builtin_assume_aligned(o_Ap.ptr(), USE_OCCA_MEM_BYTE_ALIGN) ;
  const dfloat * __restrict__ cpu_invDegree = (dfloat*)__builtin_assume_aligned(o_invDegree.ptr(), USE_OCCA_MEM_BYTE_ALIGN) ;

  dfloat * __restrict__ cpu_x  = (dfloat*)__builtin_assume_aligned(o_x.ptr(), USE_OCCA_MEM_BYTE_ALIGN) ;
  dfloat * __restrict__ cpu_r  = (dfloat*)__builtin_assume_aligned(o_r.ptr(), USE_OCCA_MEM_BYTE_ALIGN) ;

  dfloat rdotr = 0;
  
  switch(Nq){
  case  2: rdotr = BPSerialUpdatePCGKernel <  2 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break; 
  case  3: rdotr = BPSerialUpdatePCGKernel <  3 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break;
  case  4: rdotr = BPSerialUpdatePCGKernel <  4 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break;
  case  5: rdotr = BPSerialUpdatePCGKernel <  5 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break;
  case  6: rdotr = BPSerialUpdatePCGKernel <  6 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break;
  case  7: rdotr = BPSerialUpdatePCGKernel <  7 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break;
  case  8: rdotr = BPSerialUpdatePCGKernel <  8 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break;
  case  9: rdotr = BPSerialUpdatePCGKernel <  9 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break;
  case 10: rdotr = BPSerialUpdatePCGKernel < 10 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break;
  case 11: rdotr = BPSerialUpdatePCGKernel < 11 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break;
  case 12: rdotr = BPSerialUpdatePCGKernel < 12 > (Nelements, cpu_invDegree, cpu_p, cpu_Ap, alpha, cpu_x, cpu_r); break;
  }

  return rdotr;
}

dfloat BPUpdatePCG(BP_t *BP,
			 occa::memory &o_p, occa::memory &o_Ap, const dfloat alpha,
			 occa::memory &o_x, occa::memory &o_r){

  setupAide &options = BP->options;
  
  int fixedIterationCountFlag = 0;
  int flexible = options.compareArgs("KRYLOV SOLVER", "FLEXIBLE");
  int verbose = options.compareArgs("VERBOSE", "TRUE");
  int serial = options.compareArgs("THREAD MODEL", "Serial");
  
  mesh_t *mesh = BP->mesh;

  if(serial==1){
    
    dfloat rdotr1 = BPSerialUpdatePCG(mesh->Nq, mesh->Nelements, 
					    BP->o_invDegree,
					    o_p, o_Ap, alpha, o_x, o_r);

    dfloat globalrdotr1 = 0;
    MPI_Allreduce(&rdotr1, &globalrdotr1, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
    
    return globalrdotr1;
  }
  
  dfloat rdotr1 = 0;
  
  // x <= x + alpha*p
  // r <= r - alpha*A*p
  // dot(r,r)

  // zero accumulator
  BP->o_zeroAtomic.copyTo(BP->o_tmpAtomic, BP->Nfields*sizeof(dfloat), 0);
  
  BP->updatePCGKernel(mesh->Nelements*mesh->Np, BP->NblocksUpdatePCG,
		      BP->o_invDegree, o_p, o_Ap, alpha, o_x, o_r, BP->o_tmpAtomic);
  
  BP->o_tmpAtomic.copyTo(BP->tmpAtomic, BP->Nfields*sizeof(dfloat), 0);
  
  rdotr1 = 0;
  for(int n=0;n<BP->Nfields;++n){
    rdotr1 += BP->tmpAtomic[n];
  }
  
  dfloat globalrdotr1 = 0;
  MPI_Allreduce(&rdotr1, &globalrdotr1, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
  
  rdotr1 = globalrdotr1;

  return rdotr1;
}

