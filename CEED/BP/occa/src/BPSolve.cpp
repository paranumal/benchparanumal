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


int BPSolve(BP_t *BP, dfloat lambda, dfloat mu, dfloat tol, occa::memory &o_r, occa::memory &o_x, double *opElapsed){
  
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
  if(options.compareArgs("KRYLOV SOLVER", "PCG"))
    Niter = BPPCG(BP, lambda, mu, o_r, o_x, tol, maxIter, opElapsed);
  else   if(options.compareArgs("KRYLOV SOLVER", "MINRES"))
    Niter = BPMINRES(BP, lambda, mu, o_r, o_x, tol, maxIter, opElapsed);

  // zero mean of RHS
  if(BP->allNeumann) 
    BPZeroMean(BP, o_x);
  
  return Niter;

}

// FROM NEKBONE: not appropriate since it assumes zero initial data
int BPPCG(BP_t* BP, dfloat lambda, dfloat mu,
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

  dlong Ndof = mesh->Nelements*mesh->Np*BP->Nfields;
  dlong Nbytes = Ndof*sizeof(dfloat);
  
  /*aux variables */
  occa::memory o_p  = BP->o_solveWorkspace + 0*Ndof*sizeof(dfloat);
  occa::memory o_z  = BP->o_solveWorkspace + 1*Ndof*sizeof(dfloat);
  occa::memory o_Ap = BP->o_solveWorkspace + 2*Ndof*sizeof(dfloat);
  occa::memory o_Ax = BP->o_solveWorkspace + 3*Ndof*sizeof(dfloat);
  occa::memory o_res = BP->o_solveWorkspace + 4*Ndof*sizeof(dfloat);

  occa::streamTag starts[MAXIT+1];
  occa::streamTag ends[MAXIT+1];

  rdotz1 = 1;

  dfloat rdotr0;

  // compute A*x
  dfloat pAp = BPOperator(BP, lambda, mu, o_x, o_Ax, dfloatString, starts, ends); 
  
  // subtract r = b - A*x
  BPScaledAdd(BP, -1.f, o_Ax, 1.f, o_r);

  rdotr0 = BPWeightedNorm2(BP, BP->o_invDegree, o_r);

  dfloat TOL =  mymax(tol*tol*rdotr0,tol*tol);

  double elapsedCopy = 0;
  double elapsedPupdate = 0;
  double elapsedAx = 0;
  double elapsedDot = 0;
  double elapsedUpdate = 0;
  double elapsedOp = 0;
  double elapsedOverall = 0;


  occa::streamTag startCopy;
  occa::streamTag endCopy;

  occa::streamTag startPupdate;
  occa::streamTag endPupdate;

  occa::streamTag startUpdate;
  occa::streamTag endUpdate;

  occa::streamTag startDot;
  occa::streamTag endDot;

  occa::streamTag startOp;
  occa::streamTag endOp;

  occa::streamTag startOverall;
  occa::streamTag endOverall;
  
  int iter;

  startOverall = BP->mesh->device.tagStream();
  
  for(iter=1;iter<=MAXIT;++iter){

    // z = Precon^{-1} r [ just a copy for this example ]
    startCopy = BP->mesh->device.tagStream();
    //    o_r.copyTo(o_z, Nbytes);
    BP->vecCopyKernel(Ndof, o_r, o_z);
    endCopy = BP->mesh->device.tagStream();
    
    rdotz2 = rdotz1;

    // r.z
    startDot = BP->mesh->device.tagStream();
    rdotz1 = BPWeightedInnerProduct(BP, BP->o_invDegree, o_r, o_z); 
    
    if(flexible){
      dfloat zdotAp = BPWeightedInnerProduct(BP, BP->o_invDegree, o_z, o_Ap);  
      
      beta = -alpha*zdotAp/rdotz2;
    }
    else{
      beta = (iter==1) ? 0:rdotz1/rdotz2;
    }  
    endDot = BP->mesh->device.tagStream();
  
    // p = z + beta*p
    startPupdate = BP->mesh->device.tagStream();
    BPScaledAdd(BP, 1.f, o_z, beta, o_p);
    endPupdate = BP->mesh->device.tagStream();
	
    // Ap and p.Ap
    startOp = BP->mesh->device.tagStream();
    pAp = BPOperator(BP, lambda, mu, o_p, o_Ap, dfloatString, starts+iter, ends+iter);
    endOp = BP->mesh->device.tagStream();
    
    // alpha = r.z/p.Ap
    alpha = rdotz1/pAp;

    //  x <= x + alpha*p
    //  r <= r - alpha*A*p
    //  dot(r,r)

    startUpdate = BP->mesh->device.tagStream();
    dfloat rdotr = BPUpdatePCG(BP, o_p, o_Ap, alpha, o_x, o_r);
    endUpdate = BP->mesh->device.tagStream();

    BP->mesh->device.finish();
    
    elapsedUpdate  += BP->mesh->device.timeBetween(startUpdate,  endUpdate);
    elapsedCopy    += BP->mesh->device.timeBetween(startCopy,    endCopy);
    elapsedPupdate += BP->mesh->device.timeBetween(startPupdate, endPupdate);    
    elapsedDot     += BP->mesh->device.timeBetween(startDot,     endDot);
    elapsedOp      += BP->mesh->device.timeBetween(startOp,      endOp);

    if (verbose&&(mesh->rank==0)) {

      if(rdotr<0)
	printf("WARNIxNG CG: rdotr = %17.15lf\n", rdotr);
      
      printf("CG: it %d r norm %12.12le alpha = %le \n", iter, sqrt(rdotr), alpha);    
    }
    
    if(rdotr<=TOL && !fixedIterationCountFlag) break;
  }

  endOverall = BP->mesh->device.tagStream();
  
  BP->mesh->device.finish();
  
  elapsedOverall += BP->mesh->device.timeBetween(startOverall,endOverall);
  
  printf("Elapsed: overall: %g, PCG Update %g, Pupdate: %g, Copy: %g, dot: %g, op: %g\n",
	 elapsedOverall, elapsedUpdate, elapsedPupdate, elapsedCopy, elapsedDot, elapsedOp);

  double gbytesPCG = 7.*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);
  double gbytesCopy = Nbytes/1.e9;
  double gbytesOp = (7+2*BP->Nfields)*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);

  if(BP->BPid==1)
    gbytesOp = (mesh->cubNp  + 2*BP->Nfields*mesh->Np)*mesh->Nelements*(sizeof(dfloat)/1.e9);
  if(BP->BPid==3)
    gbytesOp = (mesh->Nggeo*mesh->cubNp  + 2*BP->Nfields*mesh->Np)*mesh->Nelements*(sizeof(dfloat)/1.e9);
  if(BP->BPid==5)
    gbytesOp = (mesh->Nggeo*mesh->Np  + 2*BP->Nfields*mesh->Np)*mesh->Nelements*(sizeof(dfloat)/1.e9);
  
  double gbytesDot = (2*BP->Nfields+1)*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);
  double gbytesPupdate =  3*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);

  int combineDot = 0;
  combineDot = options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

  if(!combineDot)
    gbytesOp += 3*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);

  printf("Bandwidth (GB/s): PCG update: %g, Copy: %g, Op: %g, Dot: %g, Pupdate: %g\n",
	 gbytesPCG*iter/elapsedUpdate,
	 gbytesCopy*iter/elapsedCopy,
	 gbytesOp*iter/elapsedOp,
	 gbytesDot*iter/elapsedDot,
	 gbytesPupdate*iter/elapsedPupdate);
  
  
  double elapsed = 0;

  for(int it=0;it<iter;++it){
    elapsed += BP->mesh->device.timeBetween(starts[it], ends[it]);
  }

  *opElapsed += elapsed;

  elapsed /= iter;

  printf("CG: %e, %e ; \%\% (OP(x): elapsed, GNodes/s)\n", elapsed, BP->mesh->Nelements*BP->mesh->Np/(1.e9*elapsed));
  
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


int BPMINRES(BP_t *BP, dfloat lambda, dfloat mu,
	     occa::memory &f, occa::memory &u,
	     const dfloat tol, const int MAXIT,
	     double *opElapsed)
{
  dfloat      a0, a1, a2, a3, del, gam, gamp, c, cp, s, sp, eta, Au;
  int         verbose, maxiter;

  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;
  
  verbose = 0;
  if (options.compareArgs("VERBOSE", "TRUE"))
    verbose = 1;

  // HACK at the moment, using C0 pressure filter 
  dlong Ndof = (mesh->Np*BP->Nfields)*mesh->Nelements;
  dlong Nbytes = Ndof*sizeof(dfloat);
  
  occa::memory p    = BP->o_solveWorkspace + 0*Ndof*sizeof(dfloat);
  occa::memory z    = BP->o_solveWorkspace + 1*Ndof*sizeof(dfloat);
  occa::memory r    = BP->o_solveWorkspace + 2*Ndof*sizeof(dfloat);
  occa::memory w    = BP->o_solveWorkspace + 3*Ndof*sizeof(dfloat);
  occa::memory rold = BP->o_solveWorkspace + 4*Ndof*sizeof(dfloat);
  occa::memory wold = BP->o_solveWorkspace + 5*Ndof*sizeof(dfloat);
  occa::memory res  = BP->o_solveWorkspace + 6*Ndof*sizeof(dfloat);

  occa::streamTag starts[MAXIT+1];
  occa::streamTag ends[MAXIT+1];
  
  BP->vecZeroKernel(Ndof, p);
  BP->vecZeroKernel(Ndof, z);
  BP->vecZeroKernel(Ndof, r);
  BP->vecZeroKernel(Ndof, rold);
  BP->vecZeroKernel(Ndof, wold);
  BP->vecZeroKernel(Ndof, w);
  BP->vecZeroKernel(Ndof, u); // zero initial guess

  //  BP->filterKernel(mesh->Nelements, mesh->Np*mesh->Nelements, mesh->o_filterMatrix, u);
  
  BPOperator(BP, lambda, mu, u, r, dfloatString, starts, ends);                        /* r = f - Au               */
  BP->scaledAddKernel(Ndof, (dfloat)1.0, f, (dfloat)-1.0, r);

  //  BP->filterKernel(mesh->Nelements, mesh->Np*mesh->Nelements, mesh->o_filterMatrix, r);
  
  dfloat normResidual;
  normResidual = BPWeightedNorm2(BP, BP->o_invDegree, r);
  normResidual = sqrt(normResidual);

  // no precon
  z.copyFrom(r, Nbytes, 0);

  gam = BPWeightedInnerProduct(BP, BP->o_invDegree, z, r);                   /* gam = sqrt(r'*z)         */
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
  if (verbose && (mesh->rank == 0))
    printf("MINRES:  initial eta = % .15e, gamma = %.15e target %.15e\n", eta, gam, tol);

  /* MINRES iteration loop. */
  mesh->device.finish();
  dfloat tic = MPI_Wtime();

  int iter = 0;

  for (iter = 1; iter <= MAXIT; ++iter) {
    
    if (verbose && (mesh->rank == 0))
      printf("MINRES:  it % 3d  eta = % .15e, gamma = %.15e, normResidual = %.15e\n", iter, eta, gam, fabs(normResidual));
    
    BP->vecScaleKernel(Ndof, (dfloat) (1./gam), z); /* z = z/gam                */

    del = BPOperator(BP, lambda, mu, z, p, dfloatString, starts+iter, ends+iter);   /* p = Az, del = z'*p  */
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
    dfloat beta  = (iter==1) ? 0: -(gam/gamp);

    BP->updateMINRESKernel(Ndof, -a2, -a3, alpha, beta, z, wold, w, rold, r, p);

    // filter out top modes of pressure residual
    //    BP->filterKernel(mesh->Nelements, mesh->Np*mesh->Nelements, mesh->o_filterMatrix, r);
    //    BP->filterKernel(mesh->Nelements, mesh->Np*mesh->Nelements, mesh->o_filterMatrix, r);
    
    // no precon
    z.copyFrom(r, Nbytes, 0);

    gamp = gam;
    gam = BPWeightedInnerProduct(BP, BP->o_invDegree, z, r);                 /* gam = sqrt(r'*z)         */
    gam = sqrt(gam);
    a1 = sqrt(a0*a0 + gam*gam);
    cp = c;
    c  = a0/a1;
    sp = s;
    s  = gam/a1;
    BP->vecScaleKernel (Ndof, (dfloat)1.0/a1, w);         /* w = w/a1                 */
    BP->scaledAddKernel(Ndof, c*eta, w, (dfloat)1.0, u);  /* u = u + c*eta*w          */

    eta = -s*eta;

#if 1
    BPOperator(BP, lambda, mu, u, res, dfloatString, starts, ends);                        /* r = f - Au               */
    BP->scaledAddKernel(Ndof, (dfloat)1.0, f, (dfloat)-1.0, res);
    
    normResidual = BPWeightedNorm2(BP, BP->o_invDegree, res);
    normResidual = sqrt(normResidual);
#endif
  }

  BP->mesh->device.finish();

  double elapsed = 0;
  for(int it=0;it<iter;++it){
    elapsed += BP->mesh->device.timeBetween(starts[it], ends[it]);
  }

  *opElapsed += elapsed;

  elapsed /= iter;

  printf("MINRES: %e, %e ; \%\% (OP(x): elapsed, GNodes/s)\n", elapsed, BP->mesh->Nelements*BP->mesh->Np/(1.e9*elapsed));
  
  return iter;
}

