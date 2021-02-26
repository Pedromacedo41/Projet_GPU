#include "utils.h"
#include <iterator>

#define cl(x, v) memset((x), (v), sizeof(x))
#define f(i, t) for(int (i) = 0; (i) < (t); (i)++)

std::default_random_engine generator;
std::uniform_int_distribution<int> distribution(MIN,MAX);

int randint(){
    return distribution(generator);
}

int * rand_int_array(int size){ 
    int *a = new int[size];
    std::generate_n(a, size, randint);
    std::sort(a, a+size);
    return a;
}


  
/****************************************************
Check solution for small dimensions 
using C++ libraries
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
