/******************************************
   Authors:  Pedro Macedo Flores and Hudson Braga Vieira 
   Projet: Batch merge path sort
   Sorbonne Université - Master 2
   Massive parallel programming on GPU devices for Big Data 
   Paris, mars 2021
*******************************************/

#include <iterator>
#include <stdio.h>
#include <random>
#include <algorithm>  
#include <vector>
#include <random>
#include <iostream>
#define f(i, t) for(int (i) = 0; (i) < (t); (i)++)

#define MIN 0
#define MAX 100

int randint(void);
bool check_solution(int * sol, int *a, int * b, int modA, int modB);
int * rand_int_array_sorted(int size); 
int * rand_int_array(int size);
int isPowerOfTwo (unsigned int x);
int * merge_sequential(int * a, int* b, int modA, int modB);
void testCUDA(cudaError_t error, const char *file, int line);


