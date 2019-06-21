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
  int N, cubN, dim, elementType;

  options.getArgs("POLYNOMIAL DEGREE", N);
  options.getArgs("CUBATURE DEGREE", cubN);
  options.getArgs("ELEMENT TYPE", elementType);
  options.getArgs("MESH DIMENSION", dim);

  mesh_t *mesh;

  // set up mesh
  mesh = meshSetupBoxHex3D(N, cubN, options);

  dfloat lambda;
  options.getArgs("LAMBDA", lambda);

  // set up
  occa::properties kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();

  meshOccaSetup3D(mesh, options, kernelInfo);
  
  BP_t *BP = BPSetup(mesh, lambda, kernelInfo, options);

  occa::memory o_r = mesh->device.malloc(mesh->Np*mesh->Nelements*sizeof(dfloat), BP->o_r);
  occa::memory o_x = mesh->device.malloc(mesh->Np*mesh->Nelements*sizeof(dfloat), BP->o_x);    
  
  // convergence tolerance
  dfloat tol = 1e-8;
  
  int it;
  
  MPI_Barrier(mesh->comm);
  
  occa::streamTag startTag = mesh->device.tagStream();
  int Ntests = 1;
  it = 0;
  for(int test=0;test<Ntests;++test){
    o_r.copyTo(BP->o_r);
    o_x.copyTo(BP->o_x);
    it += BPSolve(BP, lambda, tol, BP->o_r, BP->o_x);
  }
  
  MPI_Barrier(mesh->comm);
  
  occa::streamTag stopTag = mesh->device.tagStream();
  mesh->device.finish();
  
  double elapsed = mesh->device.timeBetween(startTag, stopTag);
  
  double globalElapsed;
  hlong globalNelements, localNelements=mesh->Nelements;
  
  MPI_Reduce(&elapsed, &globalElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, mesh->comm);
  MPI_Reduce(&localNelements, &globalNelements, 1, MPI_HLONG, MPI_SUM, 0, mesh->comm);

  if(mesh->rank==0){
    printf("elapsed = %lf, globalElapsed = %lf, globalNelements = %lld\n",
	   elapsed, globalElapsed, globalNelements);

    printf("%d, %d, %g, %d, %g, %g; \%\%global: N, dofs, elapsed, iterations, time per node, nodes*iterations/time\n",
	   mesh->N,
	   globalNelements*mesh->Np,
	   globalElapsed,
	   it,
	   globalElapsed/(mesh->Np*globalNelements),
	   globalNelements*(it*mesh->Np/globalElapsed));
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
  BP->o_x.copyTo(BP->q);

  BPPlotVTU(BP, "foo", 0);
  
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
      exact = cos(mode*M_PI*xn)*cos(mode*M_PI*yn)*cos(mode*M_PI*zn);
      
      dfloat error = fabs(exact-BP->q[id]);
      
      maxError = mymax(maxError, error);
    }
  }
  
  dfloat globalMaxError = 0;
  MPI_Allreduce(&maxError, &globalMaxError, 1, MPI_DFLOAT, MPI_MAX, mesh->comm);
  if(mesh->rank==0)
    printf("globalMaxError = %g\n", globalMaxError);
  
  // close down MPI
  MPI_Finalize();

  return 0;
}
