#include "utils.h"

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);
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
            if(mCPU[i*2*d +j] != mSOL[i*2*d+j + d*(((int) (log(d) /log(2)))%2)]) return false;
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


