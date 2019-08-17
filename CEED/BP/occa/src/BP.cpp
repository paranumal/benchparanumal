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

int main(int argc, char **argv){

  // start up MPI
  MPI_Init(&argc, &argv);

  if(argc!=2){
    printf("usage: ./BP setupfile\n");

    MPI_Finalize();
    exit(-1);
  }

  // if argv > 2 then should load input data from argv
  setupAide options(argv[1]);

  // set up mesh stuff
  string fileName;
  int N, dim, elementType, kernelId;

  options.getArgs("POLYNOMIAL DEGREE", N);

  int cubN = 0;
  options.getArgs("CUBATURE DEGREE", cubN);

  int cubNinc = 0;
  if(!cubN)
    options.getArgs("CUBATURE DEGREE INCREMENT", cubNinc);

  if(cubNinc)
    cubN = N+cubNinc;

  // default to same degree nodes and cubature (check this)
  if(!cubN && !cubNinc)
    cubN = N;
  
  options.getArgs("ELEMENT TYPE", elementType);
  options.getArgs("MESH DIMENSION", dim);
  options.getArgs("KERNEL ID", kernelId);

  int combineDot = 0;
  combineDot = options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

  mesh_t *mesh;
  
  // set up mesh
  if(elementType==HEXAHEDRA)
    mesh = meshSetupBoxHex3D(N, cubN, options);
  else
    mesh = meshSetupBoxTet3D(N, cubN, options);

  mesh->elementType = elementType;
  
  dfloat lambda = 1, mu = 1;
  options.getArgs("LAMBDA", lambda);
  options.getArgs("VISCOSITY",  mu);
  
  // set up
  occa::properties kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();

  meshOccaSetup3D(mesh, options, kernelInfo);
  
  BP_t *BP = BPSetup(mesh, lambda, mu, kernelInfo, options);

  occa::memory o_r, o_x;

  int useGlobal  = options.compareArgs("USE GLOBAL STORAGE", "TRUE");

  dlong Ndofs = (useGlobal) ? mesh->Nlocalized: mesh->Np*mesh->Nelements;
  Ndofs *= BP->Nfields;
  if(!useGlobal)
    o_r = mesh->device.malloc(Ndofs*sizeof(dfloat), BP->o_r);
  else
    o_r = mesh->device.malloc(Ndofs*sizeof(dfloat), BP->o_gr);

  o_x = mesh->device.malloc(Ndofs*sizeof(dfloat), BP->o_x);    
  
  // convergence tolerance
  dfloat tol = 1e-8;
  
  int it;
  int bpstart = BP->BPid;
  int bpid = BP->BPid;
  //  for(int bpid=bpstart;bpid<=6;bpid+=2){
  {
    BP->BPid = bpid;
    
    MPI_Barrier(mesh->comm);

    // warm up
    double opElapsed = 0;

    if(!useGlobal)
      BPSolve(BP, lambda, mu, tol, BP->o_r, BP->o_x, &opElapsed);
    else
      BPSolveGlobal(BP, lambda, mu, tol, BP->o_gr, BP->o_x, &opElapsed);
    
    opElapsed = 0;
    
    int Ntests = 1;
    occa::streamTag *startTags = new occa::streamTag[Ntests];
    occa::streamTag *stopTags  = new occa::streamTag[Ntests];

    it = 0;
    for(int test=0;test<Ntests;++test){

      if(!useGlobal)
	o_r.copyTo(BP->o_r);
      else
	o_r.copyTo(BP->o_gr);

      o_x.copyTo(BP->o_x);
      
      startTags[test] = mesh->device.tagStream();

      if(!useGlobal)
	it += BPSolve(BP, lambda, mu, tol, BP->o_r, BP->o_x, &opElapsed);
      else
	it += BPSolveGlobal(BP, lambda, mu, tol, BP->o_gr, BP->o_x, &opElapsed);

      stopTags[test] = mesh->device.tagStream();
    }
    mesh->device.finish();  
    MPI_Barrier(mesh->comm);
    
    double elapsed = 0;
    for(int test=0;test<Ntests;++test){
      elapsed += mesh->device.timeBetween(startTags[test], stopTags[test]);
    }

    double globalElapsed;
    hlong globalNelements, localNelements=mesh->Nelements;
    
    MPI_Reduce(&elapsed, &globalElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, mesh->comm);
    MPI_Reduce(&localNelements, &globalNelements, 1, MPI_HLONG, MPI_SUM, 0, mesh->comm);
    
    if(mesh->rank==0){
      
      double NGbytes;
      int combineDot = options.compareArgs("COMBINE DOT PRODUCT", "TRUE");
      
      int knlId = 0;
      options.getArgs("KERNEL ID", knlId);

      int useInvDeg = !useGlobal;
      
      // PCG base
      if(options.compareArgs("KRYLOV SOLVER", "PCG"))
	NGbytes = mesh->Nlocalized*((BP->Nfields*(2+2+3+2+3+3+1)+2*useInvDeg)/1.e9);    // z=r, z.r/deg, p=z+beta*p, A*p (p in/Ap out), [x=x+alpha*p, r=r-alpha*Ap, r.r./deg]
      else
	NGbytes = mesh->Nlocalized*((BP->Nfields*(2+2+11+2+2+2+3)+2*useInvDeg)/1.e9); // z = z/gam, p = Az (z in, Az out), z.p/deg, [ z=z-a2*w-a3*wold, wold=w, w=z, z=r, r=p-(del/gam)*r-(gam/gamp)*rold, rold = z], z=r, gam=sqrt(r.z/invDegree), w=w/a1, u=u+c*eta*w 
      
      if(!combineDot) NGbytes += (BP->Nfields*2+useInvDeg)*(mesh->Nlocalized/1.e9);  // z.Az/deg
      
      if(BP->BPid==1 || BP->BPid==2) NGbytes += mesh->Nelements*(mesh->cubNp/1.e9);
      if(BP->BPid==3 || BP->BPid==4) NGbytes += mesh->Nelements*(mesh->Nggeo*mesh->cubNp/1.e9);
      if(BP->BPid==5 || BP->BPid==6) NGbytes += mesh->Nelements*(mesh->Nggeo*mesh->Np/1.e9);
      if(BP->BPid==9)                NGbytes += mesh->Nelements*((mesh->dim*mesh->dim+1)*mesh->Np/1.e9);
      if(BP->BPid==10)               NGbytes += mesh->Nelements*(mesh->cubNp/1.e9);
      
      NGbytes *= sizeof(dfloat);
      
      double bw = (it*(NGbytes/(globalElapsed)));
      
      printf("elapsed = %lf, globalElapsed = %lf, globalNelements = %lld\n",
	     elapsed, globalElapsed, globalNelements);
      
      printf("%d, %d, %d, %d, %g, %d, %g, %g, %g, %d, %d, %g, %d; "
	     "\%\% global: N, Nelements, dofs, globalDofs, elapsed, iterations, time per global node, fields*global nodes*iterations/time, BW GFLOPS/s, kernel Id, combineDot, fields*nodes*iterations/opElapsed, BPid\n",
	     mesh->N,
	     mesh->Nelements,
	     globalNelements*mesh->Np,
	     mesh->Nlocalized,
	     globalElapsed,
	     it,
	     globalElapsed/((double)it*mesh->Nlocalized),
	     BP->Nfields*(it*(mesh->Nlocalized/globalElapsed)),
	     bw,
	     knlId,
	     combineDot,
	     (it*BP->Nfields)*(mesh->Nlocalized/opElapsed),
	     BP->BPid);
    }
    
    if (options.compareArgs("VERBOSE", "TRUE")){
      fflush(stdout);
      MPI_Barrier(mesh->comm);
      printf("rank %d has %d internal elements and %d non-internal elements\n",
	     mesh->rank,
	     mesh->NinternalElements,
	     mesh->NnotInternalElements);
      MPI_Barrier(mesh->comm);
    }
  
    // copy solution from DEVICE to HOST
    if(!useGlobal)
      BP->o_x.copyTo(BP->q);
    else{
      occa::memory o_x = mesh->device.malloc(mesh->Np*mesh->Nelements*BP->Nfields*sizeof(dfloat));

      //      BP->vecScatterKernel(mesh->Np*mesh->Nelements, mesh->o_localizedIds, BP->o_x, o_x);
      BP->vecMultipleScatterKernel(mesh->Np*mesh->Nelements, mesh->Nlocalized, mesh->o_localizedIds, BP->o_x, o_x);
      o_x.copyTo(BP->q);
    }
      
      
    //    BPPlotVTU(BP, "foo", 0);
    
    dfloat maxError = 0;
    for(dlong e=0;e<mesh->Nelements;++e){
      for(int n=0;n<mesh->Np;++n){
	dlong   id = e*mesh->Np+n;
	dfloat xn = mesh->x[id];
	dfloat yn = mesh->y[id];
	dfloat zn = mesh->z[id];
      
	dfloat exact;
	double mode = 1.0;
	// hard coded to match the RHS used in BPSetup
	exact = (3.*M_PI*M_PI*mode*mode+lambda)*cos(mode*M_PI*xn)*cos(mode*M_PI*yn)*cos(mode*M_PI*zn);
	
	if(BP->BPid>2 && BP->BPid!=10)
	  exact /= (3.*mode*mode*M_PI*M_PI+lambda);
      
	dfloat error = fabs(exact-BP->q[id]);
      
	maxError = mymax(maxError, error);
      }
    }
  
    dfloat globalMaxError = 0;
    MPI_Allreduce(&maxError, &globalMaxError, 1, MPI_DFLOAT, MPI_MAX, mesh->comm);
    if(mesh->rank==0)
      printf("globalMaxError = %g\n", globalMaxError);
  }  
  // close down MPI
  MPI_Finalize();

  return 0;
}
