/******************************************
   Authors:  Pedro Macedo Flores and Hudson Braga Vieira 
   Projet: Batch merge path sort
   Sorbonne Université - Master 2
   Massive parallel programming on GPU devices for Big Data 
   Paris, mars 2021
*******************************************/

#include <iostream>
#include <iterator>
#include <string>
#include "utils.h"

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void kernel_batch_sort(int * M, int i, int mul, int d){
    // which sort array?
	int k = blockIdx.x/mul; 

	// which sizes of A e B ? i
	int size = std::pow(2,i);

	// thread 2 from second block must represents thread 1025 of a virtual "superblock", where superblock is mul blocks together)
	int intermediate_threadIdx =  (blockIdx.x % mul) * blockDim.x + threadIdx.x ;
	
	// which merge? find offset of M corresponding to A and B
	int offset =   k*d +  intermediate_threadIdx;
	int idx_start_a = offset + (i%2)*d;
	int idx_start_b = idx_start_a + size;
	int m =  intermediate_threadIdx % ((int) std::pow(2, (i+1)));

    // device function
    //trifusion(M+ idx_start_a, M+idx_start_b, M+offset+ !(i%2), size, size, m);
}

void batch_sort(int d, int batch_dim, int max_threads_per_block){

    // store on GPU a vectot M of size  2 * batch_dim * d
    // copy each vector j to A[j][0....d]  (setting 0 to  A[j][d+1, ...2d-1]
    // A[batch_id][ 0, ... d//2] keeps old values and A[batch_id][d//2+1, ....d]  new ones or vice versa,  using i%2 trick 
    int * mCPU = rand_int_array(2*d*batch_dim);
    int * mSOL = new int[2*d*batch_dim];
    
    int * mGPU;
    
    testCUDA(cudaMalloc(&mGPU,2*d*batch_dim*sizeof(int)));
    testCUDA(cudaMemcpy(mGPU,mCPU, 2*d*batch_dim*sizeof(int), cudaMemcpyHostToDevice));
    
    // inplace sort. mCPU will be used in the future to compare sol from GPU
    cpu_batch_sort(mCPU, d , batch_dim);
    int mul = (d>max_threads_per_block)? (d / max_threads_per_block) : 1;

    // timer block
    float TimeVar;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop)); 
    testCUDA(cudaEventRecord(start,0));
    // timer block
    
    // execution block
    f(i, log(d)){
        // for each vector to sort, 2**( log d - i -1) merges to do, each merge take 2**(i+1) threads  => always d threads on total 
	    kernel_batch_sort<<< batch_dim*mul, (d > max_threads_per_block)? max_threads_per_block: d >>> (mGPU, i, mul, d);
	    cudaDeviceSynchronize();
    } 
    // execution block

    //timer block
    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    // timer block

    testCUDA(cudaMemcpy(mSOL, mGPU,  2*d*batch_dim*sizeof(int), cudaMemcpyDeviceToHost));

    if(check_solution_batch(mCPU, mSOL, d, batch_dim)) std::cout << "Parallel solution OK" << std::endl; 
    else std::cout << "Parallel solution Wrong" << std::endl;
    std::cout << "Elapsed GPU time: " <<  TimeVar << " ms" << std::endl << std::endl;
    
    // memory free
    testCUDA(cudaFree(mGPU));
    delete [] mCPU;
    delete [] mSOL;
}


int main(int argc, char * argv[]){
    // cin and cout as fast as printf
	std::ios_base::sync_with_stdio(false);
    
    // function to test merge algorithm. Tested
    //trifusion_test();
    
    int d = 256;
    int batch_dim = 2;

    if(argc==3){
        d = std::stoi(argv[1]);
        batch_dim= std::stoi(argv[2]);
    }

    if(isPowerOfTwo(d)){
        // check the number of SM and the parameters given 
        cudaDeviceProp prop; 
	    testCUDA(cudaGetDeviceProperties(&prop,0));
        std::cout << "GPU informations " << std::endl;
        std::cout << "-----------------" << std::endl;
	    std::cout <<  "Max threads per block: " << prop.major << std::endl;
	    std::cout <<  "SM count: " << prop.multiProcessorCount << std::endl << std::endl;

        int mul = (d>prop.maxThreadsPerBlock)? (d / prop.maxThreadsPerBlock) : 1;
        if(mul*batch_dim > prop.multiProcessorCount){
            std::cout << "WARNING: number of blocks greater than GPU SM count" << std::endl << std::endl;
        }
    
        batch_sort(d,batch_dim, prop.maxThreadsPerBlock);
    }else{
        std::cout << "ABORTED: d is not power of 2" << std::endl;
    }

    return 0; 
}



