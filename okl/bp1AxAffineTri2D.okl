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


@kernel void bp1AxAffineTri2D(const dlong Nelements,
                        @restrict const  dlong  *  elementList,
                        @restrict const  dlong  *  GlobalToLocal,
                        @restrict const  dfloat *  wJ,
                        @restrict const  dfloat *  I,
                        @restrict const  dfloat *  MM,
                        @restrict const  dfloat *  q,
                              @restrict dfloat *  Aq){

  for(int e=0;e<Nelements;e++;@outer(0)){

    @shared dfloat  s_q[p_Np];

    @exclusive dlong element;

    for(int n=0;n<p_Np;++n;@inner(0)){

      element = elementList[e];

      const dlong id = GlobalToLocal[n + element*p_Np];
      s_q[n] = (id!=-1) ? q[id] : 0.0;
    }

    for(int n=0;n<p_Np;++n;@inner(0)){

      const dfloat J = wJ[element];

      dfloat Mq = 0.0;

      for(int m=0;m<p_Np;++m){
        const dfloat Mm = MM[n + m*p_Np];
        Mq += Mm * s_q[m];
      }

      const dlong base = n + element*p_Np;
      Aq[base] = J*Mq;
    }
  }
}
