// input : d, batch_dim
int mul = (d>1024)? (d / 1024) : 1;
 
// alocar na GPU um vetor M de tamanho  2 * batch_dim * d
// copiar cada vetor j para A[j][0....d]  (deixando vazio A[j][d+1, ...2d-1]
// A[batch_id][ 0, ... d//2] guarda vetor base e A[batch_id][d//2+1, ....d]  a solucao ou vice versa,  usar i%2.  

f(i, log d){

    // for each vector to sort, 2**( log d - i -1) merges to do, each merge take 2**(i+1) threads  => always d threads on total 
	kernel<<< batch_dim*mul, (d > 1024)? 1024: d) >>> (M, i);
	sync threads;
} 

__device__ void trifusion(int * A, int * B, int * M, int size){
	// algo almost already implemented

}

__global__ void kernel(int * M, int i){
	// which sort array?
	int k = blockIdx.x/mul; 

	// which sizes of A e B ? i
	int size = 2**i;

	// thread 2 from second block must represents thread 1025 of a virtual "superblock", where superblock is mul blocks together)
	int intermediate_threadIdx =  (blockIdx.x % mul) * blockDim.x + threadIdx.x ;
	
	// which merge? find offset of M corresponding to A and B
	int offset =   k*d +  intermediate_threadIdx;
	int idx_start_a = offset + (i%2)*d;
	int idx_start_b = idx_start_a + size;
	int m =  intermediate_threadIdx % (2**(i+1));

	// using appropriated offsets, call trifusion
	trifusion(M+ idx_start_a, M+idx_start_b, M+offset+ !(i%2), size);
}














