/**
  * Copyright 2014, SINTEF ICT, Applied Mathematics.
  * Released under GPLv3
  */

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif
#include <vector>
#include <iostream>
#include <algorithm>

#include "../Reference_Solutions/SimpleRNG.h"
#include "../Reference_Solutions/reference_solution.h"

//Forward declaration of our local search functions
namespace CPU {
	unsigned int local_search(std::vector<unsigned int>& solution, const std::vector<float> &city_coordinates, const unsigned int max_iterations);
}

/** 
  * Return the current time as number of elapsed seconds of this day.
  */
inline double getCurrentTime() {
#ifdef _WIN32
    LARGE_INTEGER f;
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return t.QuadPart/(double) f.QuadPart;
#else
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec+tv.tv_usec*1e-6;
#endif
};

int main(int argc, char** argv) {

	//Number of nodes in our solution	
	unsigned int num_nodes = 1000;
	// Maximal number of iterations performed in local search
	unsigned int max_iterations = 5000;

	// choose number of nodes and maximal number of iterations according to command line argument
	if (argc > 1)
	{
		switch(atoi(argv[1])) 
		{
		// case 1 default
		case 2:
			num_nodes = 2000;
			max_iterations = 5000;
			break;       			
		case 3:
			num_nodes = 10000;
			max_iterations = 500;
			break;
		}
	}
	else {
		std::cout << "problem size not explicitly chosen, using default problem size" << std::endl;
	}

	// info output
	std::cout << "problem size:       " << num_nodes << " nodes" << std::endl;
	std::cout << "maximal iterations: " << max_iterations << std::endl;


	// generate random coordinates
	unsigned int seed = 12345;
	srand(seed);
	std::vector<float> city_coordinates(2*num_nodes);
	for(unsigned int i = 0; i < num_nodes; ++i)
	{
		city_coordinates[2*i] = ((float)rand())/((float)RAND_MAX);
		city_coordinates[2*i+1] = ((float)rand())/((float)RAND_MAX);
	}

	//Generate our circular solution vector (last node equal first)
	std::vector<unsigned int> cpu_solution(num_nodes+1);
	for(unsigned int i = 0; i < num_nodes; ++i) {
		cpu_solution[i] = i;
	}
	SimpleRNG rng(54321);
	std::random_shuffle(cpu_solution.begin()+1, cpu_solution.end()-1, rng);
	//Dummy node for more readable code
	cpu_solution[num_nodes] = cpu_solution[0];

	double cpu_start = getCurrentTime();
	unsigned int cpu_num_iterations = CPU::local_search(cpu_solution, city_coordinates, max_iterations);
	double cpu_end = getCurrentTime();
	float cpu_cost = solution_cost(cpu_solution, city_coordinates);
	std::cout << "CPU completed " << cpu_num_iterations << " iterations in " << cpu_end-cpu_start << " seconds" 
		<< std::endl;
	std::cout << " Solution has a cost of " << cpu_cost << std::endl;
	check_results(cpu_solution, max_iterations, true, false, true);
	
	return 0;
}

