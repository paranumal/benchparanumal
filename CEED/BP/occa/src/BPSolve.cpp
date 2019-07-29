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


int BPSolve(BP_t *BP, dfloat lambda, dfloat tol, occa::memory &o_r, occa::memory &o_x, double *opElapsed){
  
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
  Niter = BPPCG(BP, lambda, o_r, o_x, tol, maxIter, opElapsed);

  // zero mean of RHS
  if(BP->allNeumann) 
    BPZeroMean(BP, o_x);
  
  return Niter;

}

// FROM NEKBONE: not appropriate since it assumes zero initial data
int BPPCG(BP_t* BP, dfloat lambda, 
	  occa::memory &o_r, occa::memory &o_x, 
	  const dfloat tol, const int MAXIT,
	  double *opElapsed){
  
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

  occa::streamTag starts[MAXIT+1];
  occa::streamTag ends[MAXIT+1];
  
  rdotz1 = 1;

  dfloat rdotr0;

  // compute A*x
  dfloat pAp = BPOperator(BP, lambda, o_x, BP->o_Ax, dfloatString, starts, ends); 
  
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
    pAp = BPOperator(BP, lambda, o_p, o_Ap, dfloatString, starts+iter, ends+iter); 
    
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

  BP->mesh->device.finish();

  double elapsed = 0;
  for(int it=0;it<iter;++it){
    elapsed += BP->mesh->device.timeBetween(starts[it], ends[it]);
  }

  *opElapsed += elapsed;

  elapsed /= iter;

  printf("%e, %e ; \%\% (OP(x): elapsed, GNodes/s)\n", elapsed, BP->mesh->Nelements*BP->mesh->Np/(1.e9*elapsed));
  
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


static int BPSMINRES(BP_t *BP, dfloat lambda, dfloat mu, occa::memory &f, occa::memory &u, int zeroInitialGuess)
{
  dfloat      a0, a1, a2, a3, del, gam, gamp, c, cp, s, sp, eta, Au;
  dfloat      tol;
  int         verbose, maxiter;

  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;
  
  options.getArgs("KRYLOV SOLVER ITERATION LIMIT", maxiter);
  options.getArgs("KRYLOV SOLVER TOLERANCE",       tol);

  verbose = 0;
  if (options.compareArgs("VERBOSE", "TRUE"))
    verbose = 1;

  int Nq  = mesh->Nq;
  int NqP = Nq-1;
  int NpP = NqP*NqP*NqP;
  
  dlong Ndof = (mesh->Np*BP->dim + NpP)*mesh->Nelements;
  
  occa::memory p    = BP->o_solveWorkspace + 0*Ndof*sizeof(dfloat);
  occa::memory z    = BP->o_solveWorkspace + 1*Ndof*sizeof(dfloat);
  occa::memory r    = BP->o_solveWorkspace + 2*Ndof*sizeof(dfloat);
  occa::memory w    = BP->o_solveWorkspace + 3*Ndof*sizeof(dfloat);
  occa::memory rold = BP->o_solveWorkspace + 4*Ndof*sizeof(dfloat);
  occa::memory wold = BP->o_solveWorkspace + 5*Ndof*sizeof(dfloat);
  occa::memory res  = BP->o_solveWorkspace + 6*Ndof*sizeof(dfloat);

  BP->vecZeroKernel(Ndof, p);
  BP->vecZeroKernel(Ndof, z);
  BP->vecZeroKernel(Ndof, r);
  BP->vecZeroKernel(Ndof, rold);
  BP->vecZeroKernel(Ndof, wold);

  // TW: THIS NEEDS TO BE ZEROED
  BP->vecZeroKernel(Ndof, w);
  if(zeroInitialGuess)
    BP->vecZeroKernel(Ndof, u);

  BPOperator(BP, lambda, mu, u, r);                        /* r = f - Au               */

  dfloat zeta = 0;
  if(!zeroInitialGuess){
    // Hegedus trick
    dfloat normAx2;
    BPVecNorm2(BP, r, &normAx2);
    if (normAx2 != 0.0) {
      dfloat bdotAx;
      BPVecInnerProduct(BP, f, r, &bdotAx);

      zeta = bdotAx/normAx2;
      BPVecScale(BP, u, zeta);
      BPVecScale(BP, r, zeta);
    }
  }

  BPVecScaledAdd(BP, 1.0, f, -1.0, r);

  dfloat normResidual;
  BPVecNorm2(BP, r, &normResidual);
  normResidual = sqrt(normResidual);

  // no precon
  z.copyFrom(r);

  BPVecInnerProduct(BP, z, r, &gam);                   /* gam = sqrt(r'*z)         */
  if (gam < 0) {
    printf("BAD:  gam < 0.\n");
    exit(-1);
  }

  gamp = 0.0;
  gam  = sqrt(gam);
  eta  = gam;
  sp   = 0.0;
  s    = 0.0;
  cp   = 1.0;
  c    = 1.0;

  /* Adjust the tolerance to account for small initial residual norms. */
#if 1
  dfloat normf = 0.0;
  BPVecNorm2(BP, f, &normf);
  normf = sqrt(normf);
  tol = tol*normf;
#else
  tol = mymax(tol*fabs(eta), tol);
#endif
  if (verbose && (BP->meshV->rank == 0))
    printf("MINRES:  initial eta = % .15e, gamma = %.15e target %.15e\n", eta, gam, tol);

  /* MINRES iteration loop. */
  BP->meshV->device.finish();
  dfloat tic = MPI_Wtime();

  int i = 0;

  for (i = 0; i < maxiter; i++) {

    if (verbose && (BP->meshV->rank == 0))
      printf("MINRES:  it % 3d  eta = % .15e, gamma = %.15e, normResidual = %.15e\n", i, eta, gam, fabs(normResidual));

    //    if (fabs(eta) < tol && i>0) {
    if (fabs(normResidual) < tol && i>0) {
      if (verbose && (BP->meshV->rank == 0)) {
        BP->meshV->device.finish();
        dfloat toc = MPI_Wtime();
        printf("MINRES [%d] converged in %d iterations (eta = % .15e, normResidual = %.15e\n) took %g seconds (zeta=%17.15lf for Hegedus trick)\n",
            fsrPreconSwitch, i, eta, normResidual, toc-tic, zeta);
      }

      break;
    }

    BPVecScale(BP, z, (1./gam)); /* z = z/gam                */

    BPOperator(BP, lambda, mu, z, p);                      /* p = Az                   */
    BPVecInnerProduct(BP, p, z, &del);                 /* del = z'*p               */
    a0 = c*del - cp*s*gam;
    a2 = s*del + cp*c*gam;
    a3 = sp*gam;

    /* z = z - a2*w - a3*wold  */
    /* wold = w                */
    /* w = z                    */
    /* z = r                    */
    /* r = p - (del/gam)*r      */
    /* r = r - (gam/gamp)*rold */
    /* rold = z                */
    dfloat alpha =  -(del/gam);
    dfloat beta  = (i==0) ? 0: -(gam/gamp);

    BP->updateMinresKernel(Ndof, -a2, -a3, alpha, beta,
			       z, wold, w, rold, r, p);

    // no precon
    z.copyFrom(r);

    gamp = gam;
    BPVecInnerProduct(BP, z, r, &gam);                 /* gam = sqrt(r'*z)         */
    gam = sqrt(gam);
    a1 = sqrt(a0*a0 + gam*gam);
    cp = c;
    c  = a0/a1;
    sp = s;
    s  = gam/a1;
    BPVecScale(BP, w, 1.0/a1);                         /* w = w/a1                 */
    BPVecScaledAdd(BP, c*eta, w, 1.0, u);              /* u = u + c*eta*w          */

    eta = -s*eta;

#if 1
    /* tmp = f - Au               */
    BPOperator(BP, lambda, mu, u, res);
    BPVecScaledAdd(BP, 1.0, f, -1.0, res);
    BPVecNorm2(BP, res, &normResidual);
    normResidual = sqrt(normResidual);
#endif
  }

  return i;
}

