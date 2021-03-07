#include "utils.h"
#include <iterator>

#define cl(x, v) memset((x), (v), sizeof(x))
#define f(i, t) for(int (i) = 0; (i) < (t); (i)++)

std::default_random_engine generator;
std::uniform_int_distribution<int> distribution(MIN,MAX);

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d, code %d \n", file, line, error);
       exit(EXIT_FAILURE);
	} 
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
