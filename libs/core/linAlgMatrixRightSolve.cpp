/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim WarburtonTim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#include "linAlg.hpp"

extern "C" {
  void dgesv_ ( int     *N, int     *NRHS, double  *A,
                int     *LDA,
                int     *IPIV,
                double  *B,
                int     *LDB,
                int     *INFO );

  void sgesv_ ( int     *N, int     *NRHS, float  *A,
                int     *LDA,
                int     *IPIV,
                float  *B,
                int     *LDB,
                int     *INFO );

  void dgels_ ( char   *TRANS,
                int    *M,
                int    *N,
                int    *NRHS,
                double *A,
                int    *LDA,
                double *B,
                int    *LDB,
                double *WORK,
                int    *LWORK,
                int    *INFO);

  void sgels_ ( char   *TRANS,
                int    *M,
                int    *N,
                int    *NRHS,
                float  *A,
                int    *LDA,
                float  *B,
                int    *LDB,
                float  *WORK,
                int    *LWORK,
                int    *INFO);

  void dgeqp3_( int    *M,
                int    *N,
                double *A,
                int    *LDA,
                int    *JPVT,
                double *TAU,
                double *WORK,
                int    *LWORK,
                int    *INFO);

  void sgeqp3_( int    *M,
                int    *N,
                float  *A,
                int    *LDA,
                int    *JPVT,
                float  *TAU,
                float  *WORK,
                int    *LWORK,
                int    *INFO);

  void dormqr_( char   *SIDE,
                char   *TRANS,
                int    *M,
                int    *N,
                int    *K,
                double *A,
                int    *LDA,
                double *TAU,
                double *C,
                int    *LDC,
                double *WORK,
                int    *LWORK,
                int    *INFO);

  void sormqr_( char   *SIDE,
                char   *TRANS,
                int    *M,
                int    *N,
                int    *K,
                float  *A,
                int    *LDA,
                float  *TAU,
                float  *C,
                int    *LDC,
                float  *WORK,
                int    *LWORK,
                int    *INFO);

  void dtrsm_ ( char   *SIDE,
                char   *UPLO,
                char   *TRANSA,
                char   *DIAG,
                int    *M,
                int    *N,
                double *ALPHA,
                double *A,
                int    *LDA,
                double *B,
                int    *LDB);

  void strsm_ ( char   *SIDE,
                char   *UPLO,
                char   *TRANSA,
                char   *DIAG,
                int    *M,
                int    *N,
                float  *ALPHA,
                float  *A,
                int    *LDA,
                float  *B,
                int    *LDB);
}

namespace libp {

// C = A/B  = trans(trans(B)\trans(A))
// assume row major
void linAlg_t::matrixRightSolve(const int NrowsA, const int NcolsA, const memory<double> A,
                                const int NrowsB, const int NcolsB, const memory<double> B,
                                memory<double> &C){

  int info;

  int NrowsX = NcolsB;
  int NcolsX = NrowsB;

  int NrowsY = NcolsA;
  int NcolsY = NrowsA;

  int lwork = NrowsX*NcolsX;

  // compute inverse mass matrix
  memory<double> tmpX(NrowsX*NcolsX);
  memory<int>    ipiv(NrowsX);
  memory<double> work(lwork);

  tmpX.copyFrom(B, NrowsX*NcolsX);
  C.copyFrom(A, NrowsY*NcolsY);

  dgesv_(&NrowsX, &NcolsY, tmpX.ptr(), &NrowsX, ipiv.ptr(), C.ptr(), &NrowsY, &info);

  LIBP_ABORT("dgesv_ reports info = " << info, info);
}

// C = A/B  = trans(trans(B)\trans(A))
// assume row major
void linAlg_t::matrixRightSolve(const int NrowsA, const int NcolsA, const memory<float> A,
                                const int NrowsB, const int NcolsB, const memory<float> B,
                                memory<float> &C){

  int info;

  int NrowsX = NcolsB;
  int NcolsX = NrowsB;

  int NrowsY = NcolsA;
  int NcolsY = NrowsA;

  int lwork = NrowsX*NcolsX;

  // compute inverse mass matrix
  memory<float> tmpX(NrowsX*NcolsX);
  memory<int>   ipiv(NrowsX);
  memory<float> work(lwork);

  tmpX.copyFrom(B, NrowsX*NcolsX);
  C.copyFrom(A, NrowsY*NcolsY);

  sgesv_(&NrowsX, &NcolsY, tmpX.ptr(), &NrowsX, ipiv.ptr(), C.ptr(), &NrowsY, &info); // ?

  LIBP_ABORT("sgesv_ reports info = " << info, info);
}


} //namespace libp
