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

#include <sys/time.h>
double getTod(){
  struct timeval time;
  if(gettimeofday( &time, 0 )) return -1;

  long cur_time = 1000000 * time.tv_sec + time.tv_usec;
  double sec = cur_time / 1000000.0;
  return sec;
}


int BPSolve(BP_t *BP, dfloat lambda, dfloat tol, occa::memory &o_r, occa::memory &o_x){
  
  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;

  int Niter = 0;
  int maxIter = 1000; 

  options.getArgs("MAXIMUM ITERATIONS", maxIter);
  options.getArgs("SOLVER TOLERANCE", tol);

  // zero mean of RHS
  if(BP->allNeumann) 
    BPZeroMean(BP, o_r);
  
  // solve with preconditioned conjugate gradient (no precon)
  Niter = BPPCG(BP, lambda, o_r, o_x, tol, maxIter);

  // zero mean of RHS
  if(BP->allNeumann) 
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
  dfloat alpha = 0, beta = 0;
  
  /*aux variables */
  occa::memory &o_p  = BP->o_p;
  occa::memory &o_z  = BP->o_z;
  occa::memory &o_Ap = BP->o_Ap;
  occa::memory &o_Ax = BP->o_Ax;

  rdotz1 = 1;

  dfloat rdotr0;

  // compute A*x
  dfloat pAp = BPOperator(BP, lambda, o_x, BP->o_Ax, dfloatString);
  
  // subtract r = b - A*x
  BPScaledAdd(BP, -1.f, o_Ax, 1.f, o_r);

  rdotr0 = BPWeightedNorm2(BP, BP->o_invDegree, o_r);

  dfloat TOL =  mymax(tol*tol*rdotr0,tol*tol);

  double elapsedAx = 0;
  double elapsedDot = 0;
  
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

    // Ap and p.Ap
    pAp = BPOperator(BP, lambda, o_p, o_Ap, dfloatString); 

    // alpha = r.z/p.Ap
    alpha = rdotz1/pAp;

    //  x <= x + alpha*p
    //  r <= r - alpha*A*p
    //  dot(r,r)
    
    dfloat rdotr = BPUpdatePCG(BP, o_p, o_Ap, alpha, o_x, o_r);
    // 2 + 2 + 3 + 7 + 3 + 8 = 25

    if (verbose&&(mesh->rank==0)) {

      if(rdotr<0)
	printf("WARNIxNG CG: rdotr = %17.15lf\n", rdotr);
      
      printf("CG: it %d r norm %12.12le alpha = %le \n", iter, sqrt(rdotr), alpha);    
    }
    
    if(rdotr<=TOL && !fixedIterationCountFlag) break;
    
  }

  //  printf("Elapsed: Ax = %e, Dot = %e\n", elapsedAx, elapsedDot);
  
  return iter;
}

// this will break for BP2,4,
void BPZeroMean(BP_t *BP, occa::memory &o_q){

  dfloat qmeanLocal;
  dfloat qmeanGlobal;
  
  dlong Nblock = BP->Nblock;
  dfloat *tmp = BP->tmp;
  mesh_t *mesh = BP->mesh;

  occa::memory &o_tmp = BP->o_tmp;

  // this is a C0 thing [ assume GS previously applied to o_q ]
  BP->innerProductKernel(mesh->Nelements*mesh->Np, BP->o_invDegree, o_q, o_tmp);
  
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

dfloat BPUpdatePCG(BP_t *BP,
			 occa::memory &o_p, occa::memory &o_Ap, const dfloat alpha,
			 occa::memory &o_x, occa::memory &o_r){

  setupAide &options = BP->options;
  
  int fixedIterationCountFlag = 0;
  int flexible = options.compareArgs("KRYLOV SOLVER", "FLEXIBLE");
  int verbose = options.compareArgs("VERBOSE", "TRUE");
  
  mesh_t *mesh = BP->mesh;
  
  // x <= x + alpha*p
  // r <= r - alpha*A*p
  // dot(r,r)

  // zero accumulator
  BP->o_zeroAtomic.copyTo(BP->o_tmpAtomic);

  dlong offset = mesh->Np*(mesh->Nelements+mesh->totalHaloPairs);
  if(BP->Nfields==1)
    BP->updatePCGKernel(mesh->Nelements*mesh->Np, BP->NblocksUpdatePCG,
			BP->o_invDegree, o_p, o_Ap, alpha, o_x, o_r, BP->o_tmpAtomic);
  else
    BP->updateMultiplePCGKernel(mesh->Nelements*mesh->Np, offset, BP->NblocksUpdatePCG,
				BP->o_invDegree, o_p, o_Ap, alpha, o_x, o_r, BP->o_tmpAtomic);
  
  
  BP->o_tmpAtomic.copyTo(BP->tmpAtomic);
  
  dfloat rdotr1 = BP->tmpAtomic[0];
  
  dfloat globalrdotr1 = 0;
  MPI_Allreduce(&rdotr1, &globalrdotr1, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
  
  rdotr1 = globalrdotr1;

  return rdotr1;
}
