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
namespace GPU {
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

	// check solution result towards reference
	bool check_solution = true;
	//Number of nodes in our solution	
	unsigned int num_nodes = 1000;
	// Maximal number of iterations performed in local search
	unsigned int max_iterations = 5000;

	// choose number of nodes and maximal number of iterations according to command line argument
	if (argc > 1)
	{
		switch(abs(atoi(argv[1]))) 
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
		case 4:
			num_nodes = 2000;
			max_iterations = 350;
			check_solution = false;
			break;
		case 5:
			num_nodes = 4000;
			max_iterations = 1000;
			check_solution = false;
			break;
		case 6:
			num_nodes = 10000;
			max_iterations = 500;
			check_solution = false;
			break;
		}

		// negative argument => profiling
		if (atoi(argv[1]) < 0) {
			max_iterations = PROFILING_ITER;
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
	std::vector<unsigned int> gpu_solution(num_nodes+1);
	for(unsigned int i = 0; i < num_nodes; ++i) {
		gpu_solution[i] = i;
	}
	SimpleRNG rng(54321);
	std::random_shuffle(gpu_solution.begin()+1, gpu_solution.end()-1, rng);
	//Dummy node for more readable code
	gpu_solution[num_nodes] = gpu_solution[0];
	
	double gpu_start = getCurrentTime();
	unsigned int gpu_num_iterations = GPU::local_search(gpu_solution, city_coordinates, max_iterations);
	double gpu_end = getCurrentTime();
	float gpu_cost = solution_cost(gpu_solution, city_coordinates);
	std::cout << "GPU completed " << gpu_num_iterations << " iterations in " << gpu_end-gpu_start << " seconds" 
		<< std::endl;
	std::cout << " Solution has a cost of " << gpu_cost << std::endl;
	if (check_solution)
		check_results(gpu_solution, max_iterations, false, false);

	return 0;
}

