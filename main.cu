/******************************************
   Authors:  Pedro Macedo Flores and Hudson Braga Vieira 
   Projet: Batch merge path sort
   Sorbonne Université - Master 2
   Massive parallel programming on GPU devices for Big Data 
   Paris, mars 2021
*******************************************/

#include <cuda_device_runtime_api.h>
#include <iostream>
#include <ostream>
#include <string>
#include "utils.h"

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


__device__ void trifusion(int * a, int* b, int * sol, int modA, int modB, int idx){
    int * K = new int[2]; 
    int * P = new int[2]; 
    int * Q = new int[2];

    // initial setup
    bool aux1 = (idx > modA);
    K[0]= P[1] = aux1* (idx-modA);
    K[1]= P[0] = aux1*modA + (1-aux1)*idx;

    bool loop_bool = true;
    while(loop_bool){
        /*********************
          set Q position after K or P move following binary search. 
         ( P move 1 segment bellow Q or K move 1 segment above Q if break condition is not met yet)
        *********************/
        // mid distance between K and P 
        int offset = abs(K[1]-P[1])/2; 
        // midpoint in diagonal
        Q[0]= K[0]+offset;  
        Q[1]=  K[1]-offset;

        /********************
          P move one segment bellow Q in schema 1, 1  (bottom left = 1) 
          K move one segment above Q in schema 0, 0 (upper right = 0 )
          break condition: schema  0, 1 
        *********************/
        bool bottom_left =  (Q[1]>=0)*(Q[0] <= modB)*(!((a[Q[1]] <= b[Q[0]-1])*(Q[0] !=0)* (Q[1] != modA)));
        bool upper_right =   !((Q[0]!= modB)* (Q[1]!=0) * (a[Q[1]-1] > b[Q[0]]));
        // in break condition, tells if upper left is 0 or 1. 
        bool from_upper_or_left =  (Q[1] < modA)* (!((Q[0]!=modB)*(a[Q[1]] > b[Q[0]])));

        P[0] = (!bottom_left)*(Q[0]-1) + bottom_left*P[0];
        P[1] = (!bottom_left)*(Q[1]+1) + bottom_left*P[1];
        K[0] = bottom_left* (!upper_right) * (Q[0]+1) + (!(bottom_left* (!upper_right)))*K[0]; 
        K[1] = bottom_left* (!upper_right) *(Q[1]-1) + (!(bottom_left* (!upper_right)))*K[1];

        // only really updates in schema 0,1 
        sol[idx]= bottom_left * upper_right* (from_upper_or_left*a[Q[1]] + (!from_upper_or_left)*b[Q[0]]);
        loop_bool = !(upper_right* bottom_left);
    }
}

__global__ void trifusion_kernel_test(int * a, int* b, int * sol, int modA, int modB){
    trifusion(a, b, sol, modA, modB, threadIdx.x);
}

void trifusion_test(void){
    int M = 2;
    int modA= 1 , modB = M-modA;
    int maxAB = (modA > modB)? modA : modB;

    // random sorted vectors
    int * a = rand_int_array_sorted(modA);
    int * b = rand_int_array_sorted(modB);
    int * aGPU, *bGPU, * solGPU, *solCPU = new int[M];

    // memory alloc
    testCUDA(cudaMalloc(&aGPU, maxAB*sizeof(int)));
    testCUDA(cudaMalloc(&bGPU, maxAB*sizeof(int)));
    testCUDA(cudaMalloc(&solGPU, M*sizeof(int)));


    /***********************
         CPU run
    ************************/
    Timer timer;
    timer.start();
    int * sol = merge_sequential(a, b, modA, modB);
    timer.add();

    if(check_solution(sol, a, b, modA, modB)) std::cout << "Sequential solution OK" << std::endl; 
    else std::cout << "Sequential solution Wrong" << std::endl;
    std::cout << "Elapsed CPU time: " << timer.getsum()*1000 << " ms" << std::endl << std::endl;


    /***********************
         GPU run
    ************************/
    testCUDA(cudaMemcpy(aGPU,a, modA * sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(bGPU,b, modB * sizeof(int), cudaMemcpyHostToDevice));

    // timer block
    float TimeVar;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop)); 
    testCUDA(cudaEventRecord(start,0));
    // timer block
    
    // execution block
    trifusion_kernel_test<<<1, M>>>(aGPU, bGPU, solGPU, modA, modB);
    // execution block

    //timer block
    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    // timer block
    
    testCUDA(cudaMemcpy(solCPU, solGPU,  M * sizeof(int), cudaMemcpyDeviceToHost));

    // print results 
    
    f(i, modA){
        std::cout << a[i] << "\t" ;
    }
    std::cout<< std::endl;
    f(i, modB){
        std::cout << b[i] << "\t" ;
    }
    std::cout<< std::endl;

    f(i, M){
        std::cout << solCPU[i] << "\t" ;
    }
    std::cout<< std::endl;
    

    
    if(check_solution(solCPU, a, b, modA, modB)) std::cout << "Parallel solution OK" << std::endl; 
    else std::cout << "Parallel solution Wrong" << std::endl;
    std::cout << "Elapsed GPU time: " <<  TimeVar << " ms" << std::endl << std::endl;

    /***********************
     Memory Free
    ***********************/

    testCUDA(cudaFree(aGPU));
    testCUDA(cudaFree(bGPU));
    testCUDA(cudaFree(solGPU));

    // memory free
    delete [] a;
    delete [] b;
    delete [] sol;
    delete [] solCPU;
}


__global__ void kernel_batch_sort(int * M, int i, int mul, int d){
    // which sort array?
	int k = (int) blockIdx.x/mul; 
    //printf("%d\n", blockIdx.x % mul);
	// which sizes of A e B ? 
	int size = ((int) std::pow(2,i));

	// thread 2 from second block must represents thread 1025 of a virtual "superblock", where superblock is mul blocks together)
	int intermediate_threadIdx =  (blockIdx.x % mul) * blockDim.x + threadIdx.x ;
	
	// which merge? find offset of M corresponding to A and B
	int offset =   k*2*d +  (intermediate_threadIdx /((int) std::pow(2, (i+1)))) * std::pow(2, i+1);
	int idx_start_a = offset + (i%2)*d;
	int idx_start_b = idx_start_a + size;
	int m =  intermediate_threadIdx % ((int) std::pow(2, (i+1)));

    // device function
    trifusion(M+ idx_start_a, M+idx_start_b, M+offset + (!(i%2))*d, size, size, m);
}

void batch_sort(int d, int batch_dim, int max_threads_per_block){

    // store on GPU a vectot M of size  2 * batch_dim * d
    // copy each vector j to A[j][0....d]  (setting 0 to  A[j][d+1, ...2d-1]
    // A[batch_id][ 0, ... d//2] keeps old values and A[batch_id][d//2+1, ....d]  new ones or vice versa,  using i%2 trick 
    int * mCPU = rand_int_array(2*d*batch_dim);
    int * mSOL = new int[2*d*batch_dim];
    memset(mSOL, 0, 2*d*sizeof(int));

    f(i, batch_dim){
        memset(&mCPU[2*d*i+d], 0, d*sizeof(int));
    }

    // print result
    /*
    f(i, 2*d*batch_dim){
        std::cout << mCPU[i] << "\t";
    }
    std::cout<< std::endl;*/
    
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
    f(i, ((int) (log(d)/ log(2)))){
        // for each vector to sort, 2**( log d - i -1) merges to do, each merge take 2**(i+1) threads  => always d threads on total 
	    kernel_batch_sort<<< batch_dim*mul, (d > max_threads_per_block)? max_threads_per_block: d >>> (mGPU, i, mul, d);
    } 
    // execution block

    //timer block
    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    // timer block

    testCUDA(cudaMemcpy(mSOL, mGPU,  2*d*batch_dim*sizeof(int), cudaMemcpyDeviceToHost));

    // print result
    /*
    f(i, 2*d*batch_dim){
        std::cout << mSOL[i] << "\t";
    }
    std::cout<< std::endl;*/

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
    
    
    int d = 4;
    int batch_dim = 1;

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
	    std::cout <<  "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
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



