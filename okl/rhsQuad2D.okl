/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#define PI 3.14159265358979323846

/* forcing function   */
/* forcing function   */
#if p_Nfields==1
#define Forcing2D(x, y, lambda, f)            \
{                                             \
  f  = (2*PI*PI+lambda)*sin(PI*x)*sin(PI*y);  \
}
#else
#define Forcing2D(x, y, lambda, f, g)              \
{                                                  \
  f  = (3*PI*PI+lambda)*sin(2.0*PI*x)*sin(PI*y);   \
  g  = (3*PI*PI+lambda)*sin(PI*x)*sin(2.0*PI*y);   \
}
#endif

@kernel void rhsQuad2D(const dlong Nelements,
                       @restrict const dfloat* wJ,
                       @restrict const dfloat* gllw,
                       @restrict const dfloat* MM,
                       @restrict const dfloat* x,
                       @restrict const dfloat* y,
                       @restrict const dfloat* z,
                                 const dfloat  lambda,
                       @restrict       dfloat* rhs){

  for(dlong e=0;e<Nelements;e++;@outer(0)){

    for(int n=0;n<p_Np;++n;@inner(0)){

      // assumes w*J built into G entries
      const dlong id = e*p_Np + n;
      const dfloat r_GwJ = wJ[id];

    #if p_Nfields==1
      dfloat f = 0;
      Forcing2D(x[id], y[id], lambda, f);

      rhs[id] = r_GwJ*f;
    #else
      dfloat f = 0, g = 0;
      Forcing2D(x[id], y[id], lambda, f, g);

      const dlong base = e*p_Np*p_Nfields + n;
      rhs[base+0*p_Np] = r_GwJ*f;
      rhs[base+1*p_Np] = r_GwJ*g;
    #endif
    }
  }
}

@kernel void rhsAffineQuad2D(const dlong Nelements,
                       @restrict const dfloat* wJ,
                       @restrict const dfloat* gllw,
                       @restrict const dfloat* MM,
                       @restrict const dfloat* x,
                       @restrict const dfloat* y,
                       @restrict const dfloat* z,
                                 const dfloat  lambda,
                       @restrict       dfloat* rhs){

  for(dlong e=0;e<Nelements;e++;@outer(0)){

    for(int j=0;j<p_Nq;++j;@inner(1)){
      for(int i=0;i<p_Nq;++i;@inner(0)){

        // assumes w*J built into G entries
        const dlong id = e*p_Np + j*p_Nq + i;
        const dfloat J = wJ[e];

      #if p_Nfields==1
        dfloat f = 0;
        Forcing2D(x[id], y[id], lambda, f);

        rhs[id] = J*gllw[i]*gllw[j]*f;
      #else
        dfloat f = 0, g = 0;
        Forcing2D(x[id], y[id], lambda, f, g);

        const dlong base = e*p_Np*p_Nfields + j*p_Nq + i;
        rhs[base+0*p_Np] = J*gllw[i]*gllw[j]*f;
        rhs[base+1*p_Np] = J*gllw[i]*gllw[j]*g;
      #endif
      }
    }
  }
}
