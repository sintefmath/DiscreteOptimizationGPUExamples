/**
  * Copyright 2014, SINTEF ICT, Applied Mathematics.
  * Released under GPLv3
  */

/**
* Example for TSP & 2-exchange neighborhood (swaps 2 cities)
*/

#include <vector>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

namespace GPU {

/**
  * Class that represents a swap
  */
struct Move {
	__device__ __host__ explicit Move() 
		: i(0), j(0), solution(NULL)  {}

	/**
	  * Constructor that represents the swap of node i and j
	  */
	__device__ explicit Move(unsigned int i_, unsigned int j_, unsigned int* solution_)
		: i(i_), j(j_), solution(solution_) {}

	unsigned int* solution;
	unsigned int i,j;
};

/**
  * Cost function that calculates the cost of the edge 
  * from node i to j (e.g., in travelling salesman: the distance 
  * between the nodes)
  */
__device__ float cost_of_edge(const unsigned int& i, const unsigned int& j, const unsigned int* &solution, const float * &city_coordinates) {
	unsigned int a = solution[i];
	unsigned int b = solution[j];

	float x = city_coordinates[2*a] - city_coordinates[2*b];
	float y = city_coordinates[2*a+1] - city_coordinates[2*b+1];
	return sqrtf(x*x + y*y);
}

/**
  * The cost of swapping node i with node j. 
  */
__device__ float cost(const Move& move, const float * &city_coordinates) {
	const unsigned int i = move.i;
	const unsigned int j = move.j;
	const unsigned int* ptr = move.solution;

	// special case: switch neighboring cities
	if (j == i+1)
		return
		  cost_of_edge(i-1, j, ptr, city_coordinates) + cost_of_edge(j,i,ptr,city_coordinates)
		+ cost_of_edge(i, j+1, ptr, city_coordinates)
		- cost_of_edge(i-1, i, ptr, city_coordinates) - cost_of_edge(i,j,ptr,city_coordinates)
		- cost_of_edge(j, j+1, ptr, city_coordinates);

	return 
		  cost_of_edge(i-1, j, ptr, city_coordinates) + cost_of_edge(j, i+1, ptr, city_coordinates) // cost of new edges i-1,j,i+1
		- cost_of_edge(i-1, i, ptr, city_coordinates) - cost_of_edge(i, i+1, ptr, city_coordinates) // cost of existing edges i-1,i,i+1
		+ cost_of_edge(j-1, i, ptr, city_coordinates) + cost_of_edge(i, j+1, ptr, city_coordinates) // cost of new edges j-1,i,j+1
		- cost_of_edge(j-1, j, ptr, city_coordinates) - cost_of_edge(j, j+1, ptr, city_coordinates); // cost of existing edges j-1,j,j+1
}

/**
  * Apply the move
  */
__device__ void apply_move(Move& move) {
	unsigned int tmp = move.solution[move.i];
	move.solution[move.i] = move.solution[move.j];
	move.solution[move.j] = tmp;
}

/**
  * @param move_number Move number to generate
  * @param num_nodes_ number of nodes
  */
__device__ Move generate_move(const unsigned int& move_number_, const unsigned int& num_nodes_, unsigned int* solution_) {
	float n = static_cast<float>(num_nodes_);
	float i = static_cast<float>(move_number_);

	//Generates move number i in the following series:
	//(1, 2), (1, 3), ..., (1, n-2), (1, n-1)
	//(2, 3), (2, 4), ..., (2, n-1)
	// ...
	//(n-2, n-1)
	float dx = n-2.0f-floor((sqrtf(4.0f*(n-1.0f)*(n-2.0f) - 8.0f*i - 7.0f)-1.0f)/2.0f);
	float dy = 2.0f+i-(dx-1)*(n-2.0f)+(dx-1.0f)*dx/2.0f;
	
	unsigned int x = static_cast<unsigned int>(dx);
	unsigned int y = static_cast<unsigned int>(dy);

	return Move(x, y, solution_);
}


/**
  * CUDA kernel that executes local search on the GPU
  */
__global__ void evaluate_moves_kernel(unsigned int* solution_, const float* city_coordinates_, float* deltas_, unsigned int* moves_, unsigned int num_nodes_) {
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int grid_size = gridDim.x * blockDim.x;
	const unsigned int num_moves = (static_cast<int>(num_nodes_)-2)*(static_cast<int>(num_nodes_)-1)/2;

	float min_delta = 0.0;
	unsigned int best_move = tid;

	for (int i=tid; i < num_moves; i += grid_size) {
		Move move = generate_move(i, num_nodes_, solution_);
		float move_cost = cost(move, city_coordinates_);
		if (move_cost < min_delta) {
			min_delta = move_cost;
			best_move = i;
		}
	}

	deltas_[tid] = min_delta;
	moves_[tid] = best_move;
}

template<unsigned int threads>
__device__ void reduce_to_minimum(const unsigned int& tid, float (&deltas_shmem_)[threads], unsigned int (&moves_shmem_)[threads]) {
	volatile float (&deltas_shmem_volatile_)[threads] = deltas_shmem_;
	volatile unsigned int (&moves_shmem_volatile_)[threads] = moves_shmem_;

	for (unsigned int i=1; i<threads; i*=2) {
		const unsigned int k = 2*i*tid;
		if (k+i < threads) {
			if (deltas_shmem_volatile_[k+i] < deltas_shmem_volatile_[k]) {
				deltas_shmem_volatile_[k] = deltas_shmem_volatile_[k+i];
				moves_shmem_volatile_[k] = moves_shmem_volatile_[k+i];
			}
		}
		__syncthreads();
	}
}

/**
  * Kernel that reduces deltas_ into a single element, and then
  * applies this move
  */
template <unsigned int threads>
__global__ void apply_best_move_kernel(unsigned int* solution_, float* deltas_, unsigned int* moves_, unsigned int num_nodes_, unsigned int deltas_size_) {
	// Thread id
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

	//Shared memory available to all threads to find the minimum delta
	__shared__ float deltas_shmem[threads]; 

	//Shared memory to find the index corresponding to the minimum delta
	__shared__ unsigned int moves_shmem[threads]; 

	//Reduce from num_nodes_ elements to blockDim.x elements
	deltas_shmem[tid] = 0;
	for (unsigned int i=tid; i<deltas_size_; i += threads) {
		if (deltas_[i] < deltas_shmem[tid]) {
			deltas_shmem[tid] = deltas_[i];
			moves_shmem[tid] = moves_[i];
		}
	}
	__syncthreads();
	
	//Now, reduce all elements into a single element
	reduce_to_minimum<threads>(tid, deltas_shmem, moves_shmem);
	if (tid == 0) {
		if (deltas_shmem[0] < -1e-7f) {
			Move move = generate_move(moves_shmem[0], num_nodes_, solution_);
			apply_move(move);
		}
		deltas_[0] = deltas_shmem[0];
	}
}

/**
* Local Search with 'Best Accept' strategy (steepest descent)
* @param solution:     Current solution, will be changed during local search
* @param neighborhood: Collection of moves == potential neighbors
*/
unsigned int local_search(std::vector<unsigned int>& solution, const std::vector<float> &city_coordinates, const unsigned int max_iterations) {
	//Number of legal moves. 
	//We do not want to swap the first node, which means we have n-1 nodes we can swap
	//For the first node, we then have n-2 nodes to swap with, 
	//for the second node, we have n-3 nodes to swap with, etc.
	//and for the n-2'th node, we have 1 node to swap with.
	//which gives us sum_i=1^{n-2} i legal swaps,
	//or (n-2)(n-1)/2 legal swaps
	const unsigned int num_nodes = solution.size()-1;

	const unsigned int num_evaluate_threads = 20160;
	const unsigned int num_apply_threads = 512;
	const dim3 evaluate_block(192);
	const dim3 evaluate_grid((num_evaluate_threads + evaluate_block.x-1) / evaluate_block.x);
	const dim3 apply_block(num_apply_threads);
	const dim3 apply_grid(1);

	//Pointer to memory on the GPU 
	unsigned int* solution_gpu;
	float* coords_gpu;
	float* deltas_gpu;
	unsigned int* moves_gpu;
	cudaError err;

	unsigned int num_iterations = 0;
	
	//Ensure clean start
	err = cudaDeviceReset();
	if (err != cudaSuccess) {
		std::cout << "Could not reset device to ensure clean start" << std::endl;
		return num_iterations;
	}

	//Allocate GPU memory for solution
	err = cudaMalloc(&solution_gpu, solution.size()*sizeof(unsigned int));
	if (err != cudaSuccess) {
		std::cout << "Could not allocate GPU memory for solution" << std::endl;
		return num_iterations;
	}
	//Copy solution to GPU
	err = cudaMemcpy(solution_gpu, &solution[0], solution.size()*sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Could not copy solution to GPU memory" << std::endl;
		return num_iterations;
	}

	// Allocate GPU memory for coordinates
	err = cudaMalloc(&coords_gpu, num_nodes*2*sizeof(float));
	if (err != cudaSuccess) {
		std::cout << "Could not allocate GPU memory for coordinates" << std::endl;
		return num_iterations;
	}
	//Copy coordinates to GPU
	err = cudaMemcpy(coords_gpu, &city_coordinates[0], num_nodes*2*sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Could not copy coordinates to GPU memory" << std::endl;
		return num_iterations;
	}

	//Allocate memory for deltas
	err = cudaMalloc(&deltas_gpu, num_evaluate_threads*sizeof(float));
	if (err != cudaSuccess) {
		std::cout << "Could not allocate GPU memory for deltas" << std::endl;
		return num_iterations;
	}
	//Allocate memory for moves
	err = cudaMalloc(&moves_gpu, num_evaluate_threads*sizeof(unsigned int));
	if (err != cudaSuccess) {
		std::cout << "Could not allocate GPU memory for moves" << std::endl;
		return num_iterations;
	}

	//Use big L1 cache for move evaluation
	err = cudaFuncSetCacheConfig(evaluate_moves_kernel, cudaFuncCachePreferL1);
	if (err != cudaSuccess) {
		std::cout << "Could not set prefered L1 cache for move evaluation kernel" << std::endl;
		return num_iterations;
	}

	for (;;) {
		float min_delta = 0.0; 
		
		//Loop through all possible moves and find best (steepest descent)
		evaluate_moves_kernel<<<evaluate_grid, evaluate_block>>>(solution_gpu, coords_gpu, deltas_gpu, moves_gpu, num_nodes);
		apply_best_move_kernel<num_apply_threads><<<apply_grid, apply_block>>>(solution_gpu, deltas_gpu, moves_gpu, num_nodes, num_evaluate_threads);
		
		//Copy the smallest delta and best move to the CPU.
		err = cudaMemcpy(&min_delta, &deltas_gpu[0], sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cout << "Could not copy minimum delta to CPU" << std::endl;
			return 0;
		}

		// If no moves improve the solution, we are finished
		if (min_delta > -1e-7) {
			break;
		}

		++num_iterations;
		if (num_iterations >= max_iterations)
			break;
	}

	//Copy solution to CPU
	err = cudaMemcpy(&solution[0], solution_gpu, solution.size()*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "Could not copy solution to CPU memory" << std::endl;
		return num_iterations;
	}

	err = cudaDeviceReset();
	if (err != cudaSuccess) {
		std::cout << "Could not reset device after finishing" << std::endl;
		return num_iterations;
	}

	return num_iterations;
}

} //Namespace
