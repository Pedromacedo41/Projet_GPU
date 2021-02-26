/******************************************
   Authors:  Pedro Macedo Flores and Hudson Braga Vieira 
   Projet: Batch merge path sort
   Sorbonne Université - Master 2
   Massive parallel programming on GPU devices for Big Data 
   Paris, mars 2021
*******************************************/

#include <iostream>
#include <iterator>
#include <stdio.h>
#include <random>
#include <algorithm>  
#include <vector>
#include "timer.h"

#define cl(x, v) memset((x), (v), sizeof(x))
#define f(i, t) for(int (i) = 0; (i) < (t); (i)++)


__global__ void kernel_k(int * a, int* b, int modA, int modB){

}



int * merge_path_cpu(int * a, int* b, int modA, int modB){
    int *sol= new int[modA+modB-1];



    return sol;
}


// non tested
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


int main(void){
    // cin and cout as fast as printf
	std::ios_base::sync_with_stdio(false);

    int * a, * b, * aGPU, * bGPU;
    int d = 256;
    int modA= 128 , modB = d-modA;




    /****************************************************
       Check solution for small dimensions 
       using C++ libraries
    ****************************************************/
    int * sol = merge_path_cpu(a, b, modA, modB);

    if(check_solution(sol, a, b, modA, modB)) std::cout << "Valid solution" << std::endl; 
    else std::cout << "Wrong solution" << std::endl;
    
    
    /*
    f(i, modA+modB-1){
        std::cout << sol  
    }*/

    delete [] a;
    delete [] b;

    return 0; 
}



