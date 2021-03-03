/******************************************
   Authors:  Pedro Macedo Flores and Hudson Braga Vieira 
   Projet: Batch merge path sort
   Sorbonne Université - Master 2
   Massive parallel programming on GPU devices for Big Data 
   Paris, mars 2021
*******************************************/

#include <cuda_device_runtime_api.h>
#include <iostream>
#include <iterator>
#include <ostream>
#include <stdio.h>
#include <random>
#include <algorithm>  
#include <vector>
#include "timer.h"
#include <random>
#include "utils.h"

#define cl(x, v) memset((x), (v), sizeof(x))
#define f(i, t) for(int (i) = 0; (i) < (t); (i)++)

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d, code %d \n", file, line, error);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


int * merge_sequential(int * a, int* b, int modA, int modB){
    int *sol= new int[modA+modB-1];

    int i=0, j = 0;
    while((i+j)< (modA+modB)){
        if(i>=modA){
            sol[i+j]= b[j];
            j++;

        }else if (j >= modB || a[i] < b[j]) {
            sol[i+j] = a[i];       // goes 
            i++;
        }else {
            sol[i+j]= b[j];        // goes right
            j++;
        }

    }

    return sol;
}

// using global memory (gpu) . Not yet optimized, not testes
__global__ void kernel_k(int * a, int* b, int * sol, int modA, int modB){
    int idx = threadIdx.x;
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
        bool upper_right =       (  )   * !(a[Q[1]-1] > b[Q[0]]);
        bool bottom_left = (a[Q[1]] < b[Q[0]-1]);
        // in break condition, tells if upper left is 0 or 1. 
        bool from_upper_or_left = (a[Q[1]] <= b[Q[0]]);

        P[1] = bottom_left*(Q[1]+1);
        P[0] = bottom_left*(Q[0]-1);
        K[1] = (!bottom_left)*(Q[1]-1);
        K[0] = (!bottom_left)*(Q[0]+1);

        // only really updates in schema 0,1 
        sol[idx]+= upper_right* (!bottom_left) * (from_upper_or_left*a[Q[1]] + (!from_upper_or_left)*b[Q[0]]);
        loop_bool =  upper_right* (!bottom_left);
    }
}


int main(void){
    // cin and cout as fast as printf
	std::ios_base::sync_with_stdio(false);

    int M = 20;
    int modA= 5 , modB = M-modA;
    int maxAB = (modA > modB)? modA : modB;

    // random sorted vectors
    int * a = rand_int_array(modA);
    int * b = rand_int_array(modB);
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
    kernel_k<<<1, M>>>(aGPU, bGPU,  solGPU, modA, modB);
    // execution block

    //timer block
    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    // timer block
    
    testCUDA(cudaMemcpy(solCPU, solGPU,  M * sizeof(int), cudaMemcpyDeviceToHost));
    
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

    return 0; 
}



