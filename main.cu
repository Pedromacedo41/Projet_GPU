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
#include <random>
#include "utils.h"

#define cl(x, v) memset((x), (v), sizeof(x))
#define f(i, t) for(int (i) = 0; (i) < (t); (i)++)


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


__global__ void kernel_k(int * a, int* b, int modA, int modB){

}



int main(void){
    // cin and cout as fast as printf
	std::ios_base::sync_with_stdio(false);

    int d = 20;
    int modA= 5 , modB = d-modA;

    // random sorted vectors
    int * a = rand_int_array(modA);
    int * b = rand_int_array(modB);

    f(i, modA){
        std::cout<< a[i] << std::endl;
    }
    
    /****************************************************
    Check solution for small dimensions 
    using C++ libraries
    ****************************************************/
    
    int * sol = merge_sequential(a, b, modA, modB);
    
    /*
    f(i, modA){
        std::cout<< sol[i] << std::endl;
    }*/

    if(check_solution(sol, a, b, modA, modB)) std::cout << "Valid solution" << std::endl; 
    else std::cout << "Wrong solution" << std::endl;

    // memory free
    delete [] a;
    delete [] b;
    delete [] sol;

    return 0; 
}



