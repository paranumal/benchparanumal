
#define p_G00ID 0
#define p_G01ID 1
#define p_G02ID 2
#define p_G11ID 3
#define p_G12ID 4
#define p_G22ID 5


#define p_Nq (p_N+1)
#define p_cubNq (p_N+2)
#define p_Np ( p_Nq*p_Nq*p_Nq )
#define p_cubNp ( p_cubNq*p_cubNq*p_cubNq )

__global__ void benchpBP3Global_v0(const int Nelements,
				   void *context,
				   CudaFieldsInt * __restrict__ localizedIds,
				   const double * __restrict__ q,
				   const double * __restrict__ ggeo,
				   const double * __restrict__ I,
				   const double * __restrict__ D,
				   double * __restrict__ Aq){
  
  int e = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  __shared__ double s_Iq[p_cubNq][p_cubNq][p_cubNq];	
  
  __shared__ double s_D[p_cubNq][p_cubNq];
  __shared__ double s_I[p_cubNq][p_Nq];
  
  __shared__ double s_Gqr[p_cubNq][p_cubNq];
  __shared__ double s_Gqs[p_cubNq][p_cubNq];
  
  double r_qt, r_q[p_cubNq], r_Aq[p_cubNq];
  
  // array of threads
  s_D[ty][tx] = D[p_cubNq*ty+tx]; 
  
  if(tx<p_Nq){
    s_I[ty][tx] = I[p_Nq*ty+tx];
  }
  
  // load pencil of u into register
  if(tx<p_Nq && ty<p_Nq){
    for(int k = 0; k < p_Nq; k++) {
      const int id = e*p_Np + k*p_Nq*p_Nq+ ty*p_Nq + tx;
      int localId = localizedIds[id]-1;
      r_q[k] = q[localId];
    }
  }

  __syncthreads();
  
  {
    int b = ty, a = tx;
    if(a<p_Nq && b<p_Nq){				
      for(int k=0;k<p_cubNq;++k){			
	double res = 0;				
	for(int c=0;c<p_Nq;++c){			
	  res += s_I[k][c]*r_q[c];			
	}						
	s_Iq[k][b][a] = res;			
      }						
    }
  }

  __syncthreads();

  {
    int k = ty, a = tx;
    if(a<p_Nq){					
      for(int b=0;b<p_Nq;++b){			
	r_Aq[b] = s_Iq[k][b][a];			
      }						
      
      for(int j=0;j<p_cubNq;++j){			
	double res = 0;				
	for(int b=0;b<p_Nq;++b){			
	  res += s_I[j][b]*r_Aq[b];			
	}						
	s_Iq[k][j][a] = res;			
      }						
    }						
  }

  __syncthreads();
  
  {
    int k = ty, j = tx;
    for(int a=0;a<p_Nq;++a){			
      r_Aq[a] = s_Iq[k][j][a];			
    }						
    
    for(int i=0;i<p_cubNq;++i){			
      double res = 0;				
      for(int a=0;a<p_Nq;++a){			
	res += s_I[i][a]*r_Aq[a];			
      }						
      s_Iq[k][j][i] = res;				
    }						
  }							
  
  {
    for(int k = 0; k < p_cubNq; k++) {
      r_Aq[k] = 0.f; // zero the accumulator
    }
  }
  
  // Layer by layer
#pragma unroll p_cubNq
  for(int k = 0;k < p_cubNq; k++){
    
    __syncthreads();
    
    int j = ty, i = tx;
      
    // share u(:,:,k)
    double qr = 0, qs = 0;
      
    r_qt = 0;
      
#pragma unroll p_cubNq
    for(int m = 0; m < p_cubNq; m++) {
      double Dim = s_D[i][m];
      double Djm = s_D[j][m];
      double Dkm = s_D[k][m];
	
      qr += Dim*s_Iq[k][j][m];
      qs += Djm*s_Iq[k][m][i];
      r_qt += Dkm*s_Iq[m][j][i];	    
    }
      
    // prefetch geometric factors
    //    const int gbase = e*p_Nggeo*p_cubNp + k*p_cubNq*p_cubNq + j*p_cubNq + i;
    const int gbase = e*p_cubNp + k*p_cubNq*p_cubNq + j*p_cubNq + i;
    const int stride = p_cubNp*Nelements;

    const double G00 = ggeo[gbase+p_G00ID*stride];
    const double G01 = ggeo[gbase+p_G01ID*stride];
    const double G02 = ggeo[gbase+p_G02ID*stride];
    const double G11 = ggeo[gbase+p_G11ID*stride];
    const double G12 = ggeo[gbase+p_G12ID*stride];
    const double G22 = ggeo[gbase+p_G22ID*stride];
      
    s_Gqr[j][i] = (G00*qr + G01*qs + G02*r_qt);
    s_Gqs[j][i] = (G01*qr + G11*qs + G12*r_qt);
      
    r_qt = G02*qr + G12*qs + G22*r_qt;
      
    __syncthreads();
	  
    double Aqtmp = 0;
      
#pragma unroll p_cubNq
    for(int m = 0; m < p_cubNq; m++){
      double Dmi = s_D[m][i];
      double Dmj = s_D[m][j];
      double Dkm = s_D[k][m];
	
      Aqtmp += Dmi*s_Gqr[j][m];
      Aqtmp += Dmj*s_Gqs[m][i];
      r_Aq[m] += Dkm*r_qt;
    }
      
    r_Aq[k] += Aqtmp;
  }

  __syncthreads();

  {							
    /* lower 'k' */
    {
      int j = ty, i = tx;
      							
      for(int c=0;c<p_Nq;++c){			
	double res = 0;				
	for(int k=0;k<p_cubNq;++k){			
	  res += s_I[k][c]*r_q[k];			
	}						
	s_Iq[c][j][i] = res;				
      }						
    }

    __syncthreads();
      
    {
      int c = ty, i = tx;
							
      if(c<p_Nq){					
	for(int j=0;j<p_cubNq;++j){			
	  r_q[j] = s_Iq[c][j][i];			
	}						
	  
	for(int b=0;b<p_Nq;++b){			
	  double res = 0;				
	  for(int j=0;j<p_cubNq;++j){			
	    res += s_I[j][b]*r_q[j];			
	  }						
	    
	  s_Iq[c][b][i] = res;			
	}						
      }						
    }

    __syncthreads();

    {
      int c = ty, b = tx;
							
      if(b<p_Nq && c<p_Nq){				
	for(int i=0;i<p_cubNq;++i){			
	  r_q[i] = s_Iq[c][b][i];			
	}						
	  
	for(int a=0;a<p_Nq;++a){			
	  double res = 0;				
	  for(int i=0;i<p_cubNq;++i){			
	    res += s_I[i][a]*r_q[i];			
	  }						
	    
	  s_Iq[c][b][a] = res;			
	}						
      }						
    }							
  }
    
  // write out

  {
    int j = ty, i = tx;
    if(i<p_Nq && j<p_Nq){
#pragma unroll p_Nq
      for(int k = 0; k < p_Nq; k++){
	const int id = e*p_Np +k*p_Nq*p_Nq+ j*p_Nq + i;
	int localId = localizedIds[id]-1;
	double res = s_Iq[k][j][i];
	atomicAdd(Aq+localId, res); // atomic assumes Aq zerod
      }
    }
  }
}
}

#if p_cubNq==3
#define p_Nblk 3
#elif p_cubNq==4
#define p_Nblk 2
#else
#define p_Nblk 1
#endif

__global__ void BP3Global_v1(const int Nelements,
			     void *context,
			     CudaFieldsInt * __restrict__ localizedIds,
			     const double * __restrict__ q,
			     const double * __restrict__ ggeo,
			     const double * __restrict__ I,
			     const double * __restrict__ D,
			     double * __restrict__ Aq){
  
  int eo = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int es = threadIdx.z;
    
  __shared__ double s_Iq[p_Nblk][p_cubNq][p_cubNq][p_cubNq];	
    
  __shared__ double s_D[p_cubNq][p_cubNq];
  __shared__ double s_I[p_cubNq][p_Nq];
    
  __shared__ double s_Gqr[p_Nblk][p_cubNq][p_cubNq];
  __shared__ double s_Gqs[p_Nblk][p_cubNq][p_cubNq];

  double r_qt;
    
  // heavy on registers (FP64, 2*3*8 for N=7)
  double r_q[p_cubNq], r_Aq[p_cubNq];

  int r_e = eo + es;

  {
    int j = ty, i = tx;
      
    if(es==0){
      s_D[j][i] = D[p_cubNq*j+i]; 
	
      if(i<p_Nq){
	s_I[j][i] = I[p_Nq*j+i];
      }
    }
      
    if(r_e<Nelements){
      // load pencil of u into register
	
      if(i<p_Nq && j<p_Nq){
	for(int k = 0; k < p_Nq; k++) {
	  const int id = r_e*p_Np +k*p_Nq*p_Nq+ j*p_Nq + i;
	  int localId = localizedIds[id]-1;
	  r_q[k] = q[localId];
	}
      }
    }
  }

  __syncthreads();
    
  {
    int b = ty, a = tx;
    if(a<p_Nq && b<p_Nq){
#pragma unroll p_cubNq
      for(int k=0;k<p_cubNq;++k){			
	double res = 0;
#pragma unroll p_Nq
	for(int c=0;c<p_Nq;++c){			
	  res += s_I[k][c]*r_q[c];			
	}						
	s_Iq[es][k][b][a] = res;			
      }							
    }							
  }

  __syncthreads();

  {
    int k = ty, a = tx;
							
    if(a<p_Nq){					
      for(int b=0;b<p_Nq;++b){			
	r_Aq[b] = s_Iq[es][k][b][a];		
      }						
	
#pragma unroll p_cubNq
      for(int j=0;j<p_cubNq;++j){			
	double res = 0;
#pragma unroll p_Nq
	for(int b=0;b<p_Nq;++b){			
	  res += s_I[j][b]*r_Aq[b];			
	}						
	s_Iq[es][k][j][a] = res;			
      }						
    }						
  }

  __syncthreads();
    
  {
    int k = ty, j = tx;
    for(int a=0;a<p_Nq;++a){			
      r_Aq[a] = s_Iq[es][k][j][a];			
    }						
      
#pragma unroll p_cubNq
    for(int i=0;i<p_cubNq;++i){			
      double res = 0;
#pragma unroll p_Nq
      for(int a=0;a<p_Nq;++a){			
	res += s_I[i][a]*r_Aq[a];			
      }						
      s_Iq[es][k][j][i] = res;			
    }						
      
    for(int a = 0; a < p_cubNq; a++) {
      r_Aq[a] = 0.f; // zero the accumulator
    }
  }

  // Layer by layer
#pragma unroll p_cubNq
  for(int k = 0;k < p_cubNq; k++){
      
    __syncthreads();

    {
      int j = ty, i = tx;

      if(r_e<Nelements){
	double qr = 0, qs = 0;
	  
	r_qt = 0;
	  
#pragma unroll p_cubNq
	for(int m = 0; m < p_cubNq; m++) {
	  double Dim = s_D[i][m];
	  double Djm = s_D[j][m];
	  double Dkm = s_D[k][m];
	    
	  qr += Dim*s_Iq[es][k][j][m];
	  qs += Djm*s_Iq[es][k][m][i];
	  r_qt += Dkm*s_Iq[es][m][j][i];	    
	}
	  
	// prefetch geometric factors
	const int gbase = r_e*p_cubNp + k*p_cubNq*p_cubNq + j*p_cubNq + i;
	const int stride = p_cubNp*Nelements;
	
	const double G00 = ggeo[gbase+p_G00ID*stride];
	const double G01 = ggeo[gbase+p_G01ID*stride];
	const double G02 = ggeo[gbase+p_G02ID*stride];
	const double G11 = ggeo[gbase+p_G11ID*stride];
	const double G12 = ggeo[gbase+p_G12ID*stride];
	const double G22 = ggeo[gbase+p_G22ID*stride];
	  
	s_Gqr[es][j][i] = (G00*qr + G01*qs + G02*r_qt);
	s_Gqs[es][j][i] = (G01*qr + G11*qs + G12*r_qt);
	  
	r_qt = G02*qr + G12*qs + G22*r_qt;
	  
      }
    }

    __syncthreads();
      
    {
      int j = ty, i = tx;
	    
      double Aqtmp = 0;
	
#pragma unroll p_cubNq
      for(int m = 0; m < p_cubNq; m++){
	double Dmi = s_D[m][i];
	double Dmj = s_D[m][j];
	double Dkm = s_D[k][m];
	  
	Aqtmp += Dmi*s_Gqr[es][j][m];
	Aqtmp += Dmj*s_Gqs[es][m][i];
	r_Aq[m] += Dkm*r_qt;
      }
	
      r_Aq[k] += Aqtmp;
    }
      
    __syncthreads();
      
    {
      int j = ty, i = tx;

#pragma unroll p_Nq	    
      for(int c=0;c<p_Nq;++c){		
	double res = 0;
#pragma unroll p_cubNq
	for(int k=0;k<p_cubNq;++k){		
	  res += s_I[k][c]*r_Aq[k];		
	}					
	s_Iq[es][c][j][i] = res;		
      }						
    }

    __syncthreads();

    {
      int c = ty, i = tx;
						
      if(c<p_Nq){				
	for(int j=0;j<p_cubNq;++j){		
	  r_q[j] = s_Iq[es][c][j][i];	
	}					
	  
#pragma unroll p_Nq	    
	for(int b=0;b<p_Nq;++b){		
	  double res = 0;
#pragma unroll p_cubNq	      
	  for(int j=0;j<p_cubNq;++j){	
	    res += s_I[j][b]*r_q[j];	
	  }					
	    
	  s_Iq[es][c][b][i] = res;		
	}					
      }					
    }

    __syncthreads();

    {
      int c = ty, b = tx;
						
      if(b<p_Nq && c<p_Nq){			
	for(int i=0;i<p_cubNq;++i){		
	  r_q[i] = s_Iq[es][c][b][i];	
	}					
	  
#pragma unroll p_Nq	    
	for(int a=0;a<p_Nq;++a){		
	  double res = 0;
#pragma unroll p_cubNq
	  for(int i=0;i<p_cubNq;++i){	
	    res += s_I[i][a]*r_q[i];	
	  }					
	    
	  s_Iq[es][c][b][a] = res;		
	}					
      }
    }
  }						
    
  __syncthreads();
    
  {
    int j = ty, i = tx;
    if(r_e<Nelements){
      if(i<p_Nq && j<p_Nq){
#pragma unroll p_Nq
	for(int k = 0; k < p_Nq; k++){
	  const int id = r_e*p_Np +k*p_Nq*p_Nq+ j*p_Nq + i;
	  int localId = localizedIds[id]-1;
	  double res  = s_Iq[es][k][j][i];
	  atomicAdd(Aq+localId, res);
	}
      }
    }
  }
}

#if 0
__global__ void BP3Global_v2(const int Nelements,
			     @restrict const int  *elementList,
			     @restrict const int *localizedIds,
			     @restrict const double *ggeo,
			     @restrict const double *D,
			     @restrict const double *I,
			     const double lambda,
			     @restrict const double *q,
			     @restrict double *Aq){
  
  for(int e=0; e<Nelements; ++e; @outer(0)){
    
    __shared__ double s_Iq[p_cubNq][p_cubNq][p_cubNq];	
    
    __shared__ double s_D[p_cubNq][p_cubNq];
    __shared__ double s_I[p_cubNq][p_Nq];
    
    __shared__ double s_Gqr[p_cubNq][p_cubNq];
    __shared__ double s_Gqs[p_cubNq][p_cubNq];

    @exclusive double r_qt;
    
    // heavy on registers (FP64, 2*3*8 for N=7)
    @exclusive double r_q[p_cubNq];

    @exclusive int element;
    
    // array of threads
    for(int j=0;j<p_cubNq;++j;@inner(1)){
      for(int i=0;i<p_cubNq;++i;@inner(0)){
	
        s_D[j][i] = D[p_cubNq*j+i]; 
	
	if(i<p_Nq){
	  s_I[j][i] = I[p_Nq*j+i];
	}

	element = elementList[e];
	
        // load pencil of u into register
	if(i<p_Nq && j<p_Nq){
	  for(int k = 0; k < p_Nq; k++) {
	    const int id = element*p_Np +k*p_Nq*p_Nq+ j*p_Nq + i;
	    int localId = localizedIds[id]-1;
	    r_q[k] = q[localId];
	  }
	}
      }
    }
    
    // raise pressure degree
    //    interpolateHex3D(s_I, r_q, s_Iq);
    for(int b=0;b<p_cubNq;++b;@inner(1)){		
      for(int a=0;a<p_cubNq;++a;@inner(0)){		
	if(a<p_Nq && b<p_Nq){				
	  for(int k=0;k<p_cubNq;++k){			
	    double res = 0;				
	    for(int c=0;c<p_Nq;++c){			
	      res += s_I[k][c]*r_q[c];			
	    }						
	    s_Iq[k][b][a] = res;			
	  }						
	}						
      }							
    }							
    for(int k=0;k<p_cubNq;++k;@inner(1)){		
      for(int a=0;a<p_cubNq;++a;@inner(0)){		
							
	if(a<p_Nq){					
	  for(int b=0;b<p_Nq;++b){			
	    r_q[b] = s_Iq[k][b][a];			
	  }						
	  						
	  for(int j=0;j<p_cubNq;++j){			
	    double res = 0;				
	    for(int b=0;b<p_Nq;++b){			
	      res += s_I[j][b]*r_q[b];			
	    }						
	    s_Iq[k][j][a] = res;			
	  }						
	}						
      }							
    }							
    for(int k=0;k<p_cubNq;++k;@inner(1)){		
      for(int j=0;j<p_cubNq;++j;@inner(0)){		
	for(int a=0;a<p_Nq;++a){			
	  r_q[a] = s_Iq[k][j][a];			
	}						
							
	for(int i=0;i<p_cubNq;++i){			
	  double res = 0;				
	  for(int a=0;a<p_Nq;++a){			
	    res += s_I[i][a]*r_q[a];			
	  }						
	  s_Iq[k][j][i] = res;				
	}						

	for(int a = 0; a < p_cubNq; a++) {
	  r_q[a] = 0.f; // zero the accumulator
	}
      }
    }
    
    // Layer by layer
#pragma unroll p_cubNq
    for(int k = 0;k < p_cubNq; k++){
      
      @barrier("local");
      
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){
	  
          // share u(:,:,k)
          double qr = 0, qs = 0;

	  r_qt = 0;
	  
#pragma unroll p_cubNq
          for(int m = 0; m < p_cubNq; m++) {
            double Dim = s_D[i][m];
            double Djm = s_D[j][m];
            double Dkm = s_D[k][m];

            qr += Dim*s_Iq[k][j][m];
            qs += Djm*s_Iq[k][m][i];
	    r_qt += Dkm*s_Iq[m][j][i];	    
          }
	  
          // prefetch geometric factors
          const int gbase = element*p_cubNp + k*p_cubNq*p_cubNq + j*p_cubNq + i;
	  const int stride = p_cubNp*Nelements;
	  
          const double G00 = ggeo[gbase+p_G00ID*stride];
          const double G01 = ggeo[gbase+p_G01ID*stride];
          const double G02 = ggeo[gbase+p_G02ID*stride];
          const double G11 = ggeo[gbase+p_G11ID*stride];
          const double G12 = ggeo[gbase+p_G12ID*stride];
          const double G22 = ggeo[gbase+p_G22ID*stride];
	  
          s_Gqr[j][i] = (G00*qr + G01*qs + G02*r_qt);
          s_Gqs[j][i] = (G01*qr + G11*qs + G12*r_qt);

          r_qt = G02*qr + G12*qs + G22*r_qt;

        }
      }
      
      for(int j=0;j<p_cubNq;++j;@inner(1)){
        for(int i=0;i<p_cubNq;++i;@inner(0)){
	  
	  double Aqtmp = 0;
	  
#pragma unroll p_cubNq
          for(int m = 0; m < p_cubNq; m++){
            double Dmi = s_D[m][i];
            double Dmj = s_D[m][j];
            double Dkm = s_D[k][m];
	    
            Aqtmp += Dmi*s_Gqr[j][m];
            Aqtmp += Dmj*s_Gqs[m][i];
            r_q[m] += Dkm*r_qt;
          }

          r_q[k] += Aqtmp;
        }
      }
    }
    
    // lower pressure degree
    testHex3D(s_I, r_q, s_Iq);

    // write out
    
    for(int j=0;j<p_cubNq;++j;@inner(1)){
      for(int i=0;i<p_cubNq;++i;@inner(0)){

	if(i<p_Nq && j<p_Nq){
#pragma unroll p_Nq
	  for(int k = 0; k < p_Nq; k++){
	    const int id = element*p_Np +k*p_Nq*p_Nq+ j*p_Nq + i;
	    int localId = localizedIds[id]-1;
	    double res = s_Iq[k][j][i];
	    atomicAdd(Aq+localId, res); // atomic assumes Aq zerod
	  }
	}
      }
    }
  }
}







#if 0
// assume ggeo encodes built blocks
__global__ void BP3Global_v2(const int Nelements,
			     @restrict const int  *elementList,
			     @restrict const int *localizedIds,
			     @restrict const double *ggeo,
			     @restrict const double *D,
			     @restrict const double *I,
			     const double lambda,
			     @restrict const double *q,
			     @restrict double *Aq){
  
  for(int e=0; e<Nelements; ++e; @outer(0)){

    __shared__ double s_q[p_Np];
    
    for(int n=0;n<p_Np;++n;@inner(0)){
      s_q[n] = q[n + e*p_Np];
    }

    
    for(int n=0;n<p_Np;++n;@inner(0)){
      double res = 0;
      const double *base = ggeo + e*p_Np*p_Np; // only works for N=1
      for(int m=0;m<p_Np;++m){
	res += base[m*p_Np+n]*s_q[m];
      }
      
      Aq[e*p_Np+n] = res;
    }
  }
}
    

// assume ggeo encodes built blocks
__global__ void BP3Dot_v2(const int Nelements,
			  @restrict const int  *elementList,
			  @restrict const int *localizedIds,
			  @restrict const double *ggeo,
			  @restrict const double *D,
			  @restrict const double *I,
			  const double lambda,
			  @restrict const double *q,
			  @restrict double *Aq,
			  @restrict double *qAq){
  
  for(int e=0; e<Nelements; ++e; @outer(0)){

    __shared__ double s_q[p_Np];
    __shared__ volatile double s_qAq[p_Np];
    __shared__ volatile double s_warp[32];
    
    for(int n=0;n<p_Np;++n;@inner(0)){
      s_q[n] = q[n + e*p_Np];
    }

    
    for(int n=0;n<p_Np;++n;@inner(0)){
      double res = 0;
      const double *base = ggeo + e*p_Np*p_Np; // only works for N=1
      for(int m=0;m<p_Np;++m){
	res += base[m*p_Np+n]*s_q[m];
      }
      
      Aq[e*p_Np+n] = res;

      s_qAq[n] = s_q[n]*res;
    }
    
    // do partial reduction on p.Ap [ two phase vSIMD32 sync ]
    for(int t=0;t<p_Np;++t;@inner(0)){
      int n = t%32;
      int w = t/32;
      
      // totally hard wired for SIMD32
      if(n<16 && t+16<p_Np) s_qAq[t] += s_qAq[t+16];
      if(n< 8 && t+8<p_Np)  s_qAq[t] += s_qAq[t+8];
      if(n< 4 && t+4<p_Np)  s_qAq[t] += s_qAq[t+4];
      if(n< 2 && t+2<p_Np)  s_qAq[t] += s_qAq[t+2];
      if(n< 1 && t+1<p_Np)  s_qAq[t] += s_qAq[t+1];
      if(n==0) s_warp[w] = s_qAq[t];
    }
    
    for(int t=0;t<p_Np;++t;@inner(0)){
      int n = t%32;
      int w = t/32;
      
      if(w==0 && n*32<p_Np){ // is this the base warp, and was there an entry from above
	if( n<16 && ((n+16)*32)<p_Np) s_warp[n] += s_warp[n+16];
	if( n< 8 && ((n+ 8)*32)<p_Np) s_warp[n] += s_warp[n+ 8];
	if( n< 4 && ((n+ 4)*32)<p_Np) s_warp[n] += s_warp[n+ 4];
	if( n< 2 && ((n+ 2)*32)<p_Np) s_warp[n] += s_warp[n+ 2];
	if( n< 1 && ((n+ 1)*32)<p_Np) s_warp[n] += s_warp[n+ 1];
	
	if(n==0){
	  double res = s_warp[0];
	  atomicAdd(qAq, res);
	}
      }
    }       
  }
}

__global__ void BP3Global_v2(const int Nelements,
			     @restrict const int  *elementList,
			     @restrict const int *localizedIds,
			     @restrict const double *ggeo,
			     @restrict const double *D,
			     @restrict const double *I,
			     const double lambda,
			     @restrict const double *q,
			     @restrict double *Aq){
  
  for(int eo=0; eo<Nelements; eo+=p_Nblk; @outer(0)){

    __shared__ double s_q[p_Nblk][p_Np];
    @exclusive int r_e, element;
    
    for(int es=0;es<p_Nblk;++es;@inner(1)){
      for(int n=0;n<p_Np;++n;@inner(0)){
	r_e = es + eo;
	if(r_e<Nelements){
	  element = elementList[r_e];
	  s_q[es][n] = q[n + element*p_Np];
	}
      }
    }
    
    for(int es=0;es<p_Nblk;++es;@inner(1)){    
      for(int n=0;n<p_Np;++n;@inner(0)){
	if(r_e<Nelements){
	  double res = 0;
	  const double *base = ggeo + element*p_Np*p_Np; // only works for N=1 (ggeo needs to be hijacked to make this work)
	  for(int m=0;m<p_Np;++m){
	    res += base[m*p_Np+n]*s_q[es][m];
	  }
	  
	  Aq[element*p_Np+n] = res;
	}
      }
    }
  }
}

#endif
#endif
