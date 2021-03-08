#include "utils.h"


std::default_random_engine generator;
std::uniform_int_distribution<int> distribution(MIN,MAX);

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


bool check_solution_batch(int *mCPU, int * mSOL,  int d, int batch_dim){
    f(i, batch_dim){
        f(j, d){
            if(mCPU[i*2*d +j] != mSOL[i*2*d+j + ((int) log(d) -1)%2]) return false;
        }
    }
    return true;
}

void cpu_batch_sort(int *mCPU, int d, int batch_dim){
    Timer timer;
    timer.start();
    f(i, batch_dim){
       std::sort(&mCPU[2*d*i], &mCPU[2*d*i+d]);
    }
    timer.add();
    std::cout << "Elapsed CPU time: " << timer.getsum()*1000 << " ms" << std::endl << std::endl;
}

int * merge_sequential(int * a, int* b, int modA, int modB){
    int *sol= new int[modA+modB-1];

    int i=0, j = 0;
    while((i+j)< (modA+modB)){
        if(i>=modA){
            sol[i+j]= b[j];
            j++;

        }else if (j >= modB || a[i] < b[j]) {
            sol[i+j] = a[i];   // goes down
            i++;
        }else {
            sol[i+j]= b[j];    // goes right
            j++;
        }
    }

    return sol;
}

int randint(){
    return distribution(generator);
}

int * rand_int_array_sorted(int size){ 
    int *a = new int[size];
    std::generate_n(a, size, randint);
    std::sort(a, a+size);
    return a;
}


int * rand_int_array(int size){ 
    int *a = new int[size];
    std::generate_n(a, size, randint);
    return a;
}

int isPowerOfTwo (unsigned int x)
{
 while (((x % 2) == 0) && x > 1) /* While x is even and > 1 */
   x /= 2;
 return (x == 1);
}

  
/****************************************************
Check solution using C++ libraries
****************************************************/
bool check_solution(int * sol, int *a, int * b, int modA, int modB){
    std::vector<int> a_v (a, a+modA);    
    std::vector<int> b_v (b, b+modB);
    // concatenate vectors
    a_v.insert(a_v.end(), b_v.begin(), b_v.end());   
    
    // sort with algorithms 
    std::sort(a_v.begin(), a_v.end());

    f(i, modA+modB){
        if(sol[i]!= a_v[i]) return false;
    }

    return true;
}

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
        sol[idx]+= bottom_left * upper_right* (from_upper_or_left*a[Q[1]] + (!from_upper_or_left)*b[Q[0]]);
        loop_bool = !(upper_right* bottom_left);
    }
}

__global__ void trifusion_kernel_test(int * a, int* b, int * sol, int modA, int modB){
    trifusion(a, b, sol, modA, modB, threadIdx.x);
}

void trifusion_test(void){
    int M = 256;
    int modA= 254 , modB = M-modA;
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
    /*
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
    */

    
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

