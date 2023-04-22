#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <timer.h>

__global__ void bk1weighting(int N, double *GWJ, double *res){

  int n = threadIdx.x + blockIdx.x*blockDim.x;

  if(n<N){
    res[n] *= GWJ[n];
  }

}

// to compile: 
//   nvcc -I.  -o cublasDgemm cublasDgemm.cu -lcublas
//./cublasDgemm  168 56  162000
int main(int argc, char **argv) {

  if(argc<4){
    printf("usage: ./cublasDgemm NrowsA NcolsA NcolsB\n");
    exit(-1);
  }

  // Allocate 3 arrays on CPU
  unsigned long long int NrowsA = atoi(argv[1]);
  unsigned long long int NcolsA = atoi(argv[2]);
  unsigned long long int NcolsB = atoi(argv[3]);
  unsigned long long int NrowsB = NcolsA;
  unsigned long long int NrowsC = NrowsA;
  unsigned long long int NcolsC = NcolsB;
  unsigned long long int NrowsD = NrowsA;
  unsigned long long int NcolsD = NcolsB;
    
  
  double *h_A = (double *)malloc(NrowsA * NcolsA * sizeof(double));
  double *h_B = (double *)malloc(NrowsB * NcolsB * sizeof(double));
  double *h_C = (double *)malloc(NrowsC * NcolsC * sizeof(double));
  double *h_D = (double *)malloc(NrowsD * NcolsD * sizeof(double));
  
  for(int row=0;row<NrowsA;++row){
    for(int col=0;col<NcolsA;++col){
      h_A[row + col*NrowsA] = drand48();
    }
  }
  
  for(int row=0;row<NrowsB;++row){
    for(int col=0;col<NcolsB;++col){
      h_B[row + col*NrowsB] = drand48();
    }
  }
  
  // Allocate 3 arrays on GPU
  double *c_A, *c_B, *c_C, *c_D, *c_WJ;
  cudaMalloc(&c_A,NrowsA * NcolsA * sizeof(double));
  cudaMalloc(&c_B,NrowsB * NcolsB * sizeof(double));
  cudaMalloc(&c_C,NrowsC * NcolsC * sizeof(double));
  cudaMalloc(&c_WJ,NrowsC * NcolsC * sizeof(double));
  cudaMalloc(&c_D,NrowsD * NcolsD * sizeof(double));
  
  // ....
  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSetMatrix(NrowsA,NcolsA, sizeof(double), h_A, NrowsA, c_A, NrowsA);
  cublasSetMatrix(NrowsB,NcolsB, sizeof(double), h_B, NrowsB, c_B, NrowsB);
  cublasSetMatrix(NrowsC,NcolsC, sizeof(double), h_C, NrowsC, c_C, NrowsC);
  cublasSetMatrix(NrowsD,NcolsD, sizeof(double), h_D, NrowsD, c_D, NrowsD);
  
  const double alpha = 1.0, beta = 0.0;
  
  // C = beta*C + alpha*A*B
  const unsigned long long int lda = NrowsA, ldb = NrowsB, ldc = NrowsC, ldd = NrowsD;

  int Nmult = 5;
  int Ntests = 10;
  for(int m=0;m<Nmult;++m){

    cudaDeviceSynchronize();

    static uint64_t tic = ns();

    cudaEvent_t etic, etoc;
    cudaEventCreate(&etic);
    cudaEventCreate(&etoc);

    cudaEventRecord(etic);

    cublasStatus_t err;
    for(int t=0;t<Ntests;++t){
      
      err = cublasDgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        NrowsA, NcolsB, NcolsA,
                        &alpha,
                        c_A, lda,
                        c_B, ldb,
                        &beta,
                        c_C, ldc);

      int N = NrowsC*NcolsC;
      int T = 256;
      int B = (N+T-1)/T;

      bk1weighting <<< B, T >>> (N, c_WJ, c_C);

      err = cublasDgemm(handle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        NrowsA, NcolsB, NcolsA,
                        &alpha,
                        c_A, lda,
                        c_C, ldc,
                        &beta,
                        c_D, ldd);

      
    }
    
    cudaEventRecord(etoc);
    
    if(err != CUBLAS_STATUS_SUCCESS){
      printf("cublasDgemm failed, exiting\n");
      exit(-1);
    }

    cudaDeviceSynchronize();
    
    static uint64_t toc = ns();
    if(m==Nmult-1){
      
      double elapsed = ((toc-tic)/1.e9)/Ntests;

      float eventElapsed = 0;
      cudaEventElapsedTime(&eventElapsed, etic, etoc);
      eventElapsed /= (Ntests*1000.);
      
      double gflop = 2.*NrowsA*NcolsA*NcolsB*2./1.e9;
      
      
      printf("%lld, %lld, %lld, %g, %g, %g, %g %%%% N, elapsed, eventElapsed, gflops, event gflops\n",
             NrowsA, NcolsA, NcolsB, elapsed, eventElapsed, gflop/elapsed, gflop/eventElapsed);
    }
    
  }
  
  cudaMemcpy(h_C, c_C, NrowsC*NcolsC*sizeof(double), cudaMemcpyDeviceToHost);

#if 0
  for(int n=0;n<100;++n){
    printf("%f ", h_C[n]);
  }
#endif
  
  // TIDY UP
  // Destroy the handle
  cublasDestroy(handle);
  
  //Free GPU memory
  cudaFree(c_A);
  cudaFree(c_B);
  cudaFree(c_C);
  
  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C);
  
  return 0;
}
