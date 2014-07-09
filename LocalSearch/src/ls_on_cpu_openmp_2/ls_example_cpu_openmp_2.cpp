/**
  * Copyright 2014, SINTEF ICT, Applied Mathematics.
  * Released under GPLv3
  */

/**
* Example for TSP & 2-exchange neighborhood (swaps 2 cities)
*/
#include <vector>
#include <cmath>
#include <cstddef>
#include <omp.h>

namespace CPU {

/**
  * Class that represents a swap
  */
struct Move {
	explicit Move() 
		: i(0), j(0), solution(NULL)  {}

	/**
	  * Constructor that represents the swap of node i and j
	  */
	explicit Move(unsigned int i_, unsigned int j_, unsigned int* solution_)
		: i(i_), j(j_), solution(solution_) {}

	unsigned int* solution;
	unsigned int i,j;
};

/**
  * Cost function that calculates the cost of the edge 
  * from node i to j (e.g., in travelling salesman: the distance 
  * between the nodes)
  */
inline float cost_of_edge(const unsigned int& i, const unsigned int& j, const unsigned int* &solution, const std::vector<float> &city_coordinates) {
	unsigned int a = solution[i];
	unsigned int b = solution[j];

	float x = city_coordinates[2*a] - city_coordinates[2*b];
	float y = city_coordinates[2*a+1] - city_coordinates[2*b+1];
	return sqrtf(x*x + y*y);
}

/**
  * The cost of swapping node i with node j. 
  */
inline float cost(const Move& move, const std::vector<float> &city_coordinates) {
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
inline void apply(Move& move) {
	unsigned int tmp = move.solution[move.i];
	move.solution[move.i] = move.solution[move.j];
	move.solution[move.j] = tmp;
}

/**
  * @param move_number Move number to generate
  * @param num_nodes_ number of nodes
  */
Move generate_move(const unsigned int& move_number_, const unsigned int& num_nodes_, unsigned int* solution_) {
	//Generates move number i in the following series:	
	//(1, 2),       (1, 3),    ..., (1, n-2), (1, n-1)
	//(n-2,n-1),    (2, 3), (2, 4),      ..., (2, n-1)
	//(n-3,n-1), (n-3,n-2), (3, 4),      ..., (3, n-1)
	// ...
	//(  2,n-1), (  2,n-2),    ...,    (2,3), (n-2, n-1)
	unsigned int x = move_number_ / (num_nodes_-2);
	unsigned int y = 2 + move_number_ - (x * (num_nodes_-2)); // == 2 + move_number_ % (num_nodes_-2); start with index 2
	x += 1; // start with index 1

	// on inverted triangle?
	if (x >= y) {
		x = num_nodes_ - x;
		y = num_nodes_+1 - y;
	}

	// return move
	return Move(x, y, solution_);
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
	const unsigned int num_moves = (num_nodes-2)*(num_nodes-1)/2;

	unsigned int num_iterations = 0;
	int omp_num_threads = omp_get_num_procs();
	std::vector<int> best_moves(omp_num_threads);
	std::vector<float> min_deltas(omp_num_threads);
	omp_set_num_threads(omp_num_threads);

	for (;;) {
		//Loop through all possible moves and find best (steepest descent) in parallel for each thread
        #pragma omp parallel
		{
			int best_move;
			float min_delta = 0.0; 

            #pragma omp for
			for (int i=0; i < static_cast<int>(num_moves);++i) {
				Move move = generate_move(i, num_nodes, &solution[0]);
				float delta = cost(move, city_coordinates); 
				if (delta < min_delta) {
					best_move = i;
					min_delta = delta;
				}
			}

			best_moves[omp_get_thread_num()] = best_move;
			min_deltas[omp_get_thread_num()] = min_delta;
		}

		// find best move of threads
		unsigned int best_move_id = best_moves[0];
		float min_delta = min_deltas[0]; 
		for(int i = 1; i < omp_num_threads; ++i)
		{
			if (min_deltas[i] < min_delta) {
				best_move_id = best_moves[i];
				min_delta = min_deltas[i];
			}
		}

		// If no moves improve the solution, we are finished
		if (min_delta > -1e-7) {
			break;
		}

		// Applies best move to current solution
		Move best_move = generate_move(best_move_id, num_nodes, &solution[0]);
		apply(best_move); 
		++num_iterations;

		if (num_iterations >= max_iterations)
			break;
	}

	return num_iterations;
}

} //Namespace CPU
