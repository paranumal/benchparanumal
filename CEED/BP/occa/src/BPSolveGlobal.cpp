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

double getTod();


int BPSolveGlobal(BP_t *BP, dfloat lambda, dfloat mu, dfloat tol, occa::memory &o_r, occa::memory &o_x, double *opElapsed){
  
  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;

  int Niter = 0;
  int maxIter = 1000; 

  options.getArgs("MAXIMUM ITERATIONS", maxIter);
  options.getArgs("SOLVER TOLERANCE", tol);

  // solve with preconditioned conjugate gradient (no precon)
  if(options.compareArgs("KRYLOV SOLVER", "PCG"))
    Niter = BPPCGGlobal(BP, lambda, mu, o_r, o_x, tol, maxIter, opElapsed);
  
  return Niter;

}

// FROM NEKBONE: not appropriate since it assumes zero initial data
int BPPCGGlobal(BP_t* BP, dfloat lambda, dfloat mu,
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

  dlong Ndof = mesh->Nlocalized*BP->Nfields;
  dlong Nbytes = Ndof*sizeof(dfloat);
  
  /*aux variables */
  occa::memory o_p   = BP->o_solveWorkspace + 0*Ndof*sizeof(dfloat);
  occa::memory o_z   = BP->o_solveWorkspace + 1*Ndof*sizeof(dfloat);
  occa::memory o_res = BP->o_solveWorkspace + 2*Ndof*sizeof(dfloat);
  occa::memory o_Ap  = BP->o_solveWorkspace + 3*Ndof*sizeof(dfloat);

  occa::streamTag starts[MAXIT+1];
  occa::streamTag ends[MAXIT+1];

  rdotz1 = 1;

  dfloat rdotr0;

  // compute A*x
  dfloat pAp = BPOperatorGlobal(BP, lambda, mu, o_x, o_Ap, dfloatString, starts, ends); 

  // subtract r = b - A*x
  //  BPScaledAdd(BP, -1.f, o_Ax, 1.f, o_r);
  BP->scaledAddKernel(Ndof, (dfloat)-1.f, o_Ap, (dfloat)1.0f, o_r);

  rdotr0 = BPAtomicInnerProduct(BP, Ndof, o_r, o_r);

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
    rdotz1 = BPAtomicInnerProduct(BP, Ndof, o_r, o_z);

    if(flexible){
      dfloat zdotAp = BPAtomicInnerProduct(BP, Ndof, o_z, o_Ap);
      
      beta = -alpha*zdotAp/rdotz2;
    }
    else{
      beta = (iter==1) ? 0:rdotz1/rdotz2;
    }  
    endDot = BP->mesh->device.tagStream();
  
    // p = z + beta*p
    startPupdate = BP->mesh->device.tagStream();
    BP->scaledAddKernel(Ndof, (dfloat)1.f, o_z, beta, o_p);
    endPupdate = BP->mesh->device.tagStream();
	
    // Ap and p.Ap
    startOp = BP->mesh->device.tagStream();
    pAp = BPOperatorGlobal(BP, lambda, mu, o_p, o_Ap, dfloatString, starts+iter, ends+iter);
    endOp = BP->mesh->device.tagStream();
    
    // alpha = r.z/p.Ap
    alpha = rdotz1/pAp;

    //  x <= x + alpha*p
    //  r <= r - alpha*A*p
    //  dot(r,r)

    startUpdate = BP->mesh->device.tagStream();
    dfloat rdotr = BPUpdatePCGGlobal(BP, o_p, o_Ap, alpha, o_x, o_r);
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


dfloat BPUpdatePCGGlobal(BP_t *BP,
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

  // do this for all  (no invDegree)
  BP->updatePCGGlobalKernel(mesh->Nlocalized*BP->Nfields, o_p, o_Ap, alpha, o_x, o_r, BP->o_tmpAtomic);
  
  BP->o_tmpAtomic.copyTo(BP->tmpAtomic);
  
  dfloat rdotr1 = BP->tmpAtomic[0];
  
  dfloat globalrdotr1 = 0;
  MPI_Allreduce(&rdotr1, &globalrdotr1, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
  
  rdotr1 = globalrdotr1;

  return rdotr1;
}


