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

#define MIN 0
#define MAX 100


int randint(void);

bool check_solution(int * sol, int *a, int * b, int modA, int modB);

int * rand_int_array(int size);