#include "body.cuh"

#define reduce_threads 1024

__global__ void calc_force_cuda(body* bodies, float g, float t, int N, int it)
{

		
	// global thread ID using the formula from the CUDA docs
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// check if this is actually a valid calcuation
	if (tid < N)
	{
		//runs Newton's formula
		float dx = bodies[it].x - bodies[tid].x;
		float dy = bodies[it].y - bodies[tid].y;

		float dist = dx * dx + dy * dy;

		float force, dist_inv;
		if ((bodies[it].r + bodies[tid].r) * (bodies[it].r + bodies[tid].r) >= dist)
		{
			force = 0;
			dist_inv = 0;
		}
		else
		{
			force = -(bodies[it].m * bodies[tid].m * g) / dist;
			dist_inv = rsqrtf(dist);  // ignore the squiggle lines, this is a CUDA inbuilt function
		}

		dx *= dist_inv;
		// no writing on the iterator output field
		// a) would not work anyway, because thousands of GPU threads will try to do it simultaneously and corrupt the result and
		// b) we get the correct result anyway, because when the iterator changes the previous iterator's values will be changed then
		//v_x[it] += dx * force * t / m[it];
		bodies[tid].v_x -= dx * force * t / bodies[tid].m;

		dy *= dist_inv;
		//v_y[it] += dy * force * t / m[it];
		bodies[tid].v_y -= dy * force * t / bodies[tid].m;
	}

}

__global__ void calc_force_cuda_full(const body* bodies, float* interactions_x, float* interactions_y, const float g, const float t, const int N)
{

	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (tid_x < N && tid_y < N && tid_y != tid_x)
	{
		//runs Newton's formula
		float dx = bodies[tid_x].x - bodies[tid_y].x;
		float dy = bodies[tid_x].y - bodies[tid_y].y;
		
		float dist = dx * dx + dy * dy;

		float force, dist_inv;
		if ((bodies[tid_x].r + bodies[tid_y].r) * (bodies[tid_x].r + bodies[tid_y].r) >= dist)
		{
			force = 0;
			dist_inv = 0;
		}
		else
		{
			force = -(bodies[tid_x].m * bodies[tid_y].m * g) / dist;
			dist_inv = rsqrtf(dist);  // ignore the squiggle lines, this is a CUDA inbuilt function
		}

		// write to output pointer
		dx *= dist_inv;
		interactions_x[tid_y * N + tid_x] = -dx * force * t / bodies[tid_y].m;

		dy *= dist_inv;
		interactions_y[tid_y * N + tid_x] = -dy * force * t / bodies[tid_y].m;
	}

}

__global__ void reduce_interactions(body* bodies, const float* interactions_x, const float* interactions_y, const int N)
{
	__shared__ float shared_x[reduce_threads];
	__shared__ float shared_y[reduce_threads];

	int row = blockIdx.x;
	int tid = threadIdx.x;

	// reduce reduce_threads many elements in one go in one row
	float res_x = 0;
	float res_y = 0;
	for (int i = 0; i < ceilf(static_cast<float>(N) / reduce_threads); i++)
	{
		int idx = i * reduce_threads + tid;
		// load into shared memory
		if (idx < N)
		{
			shared_x[tid] = interactions_x[row * N + idx];
			shared_y[tid] = interactions_y[row * N + idx];
		}
		__syncthreads();

		// go over the elements with a stride
		for (int s = blockDim.x / 2; s > 0; s >>= 1)
		{
			if (tid < s && tid + s < N)
			{
				shared_x[tid] += shared_x[tid + s];
				shared_y[tid] += shared_y[tid + s];
			}
			__syncthreads();
		}

		// write result of one pass into the accumulator
		if (tid == 0)
		{
			res_x += shared_x[0];
			res_y += shared_y[0];
		}
	}
	// write accumulator into bodies pointer
	if (tid == 0)
	{
		bodies[row].v_x += res_x;
		bodies[row].v_y += res_y;
	}

}

__global__ void calc_movement_cuda(body* bodies, const float t, const int N)
{

	// global thread ID using the formula from the CUDA docs
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N)
	{
		bodies[tid].x += 0.5 * bodies[tid].v_x * t;
		bodies[tid].y += 0.5 * bodies[tid].v_y * t;
	}	

}

void process_bodies_cuda(std::vector<body>& bodies,	body* d_bodies, float* d_interactions_x, float* d_interactions_y, const sim_settings& ss)
{

	size_t n_threads = 32;
	size_t n_blocks = (bodies.size() + n_threads - 1) / n_threads;

	dim3 threads_2d(n_threads, n_threads);
	dim3 blocks_2d(n_blocks, n_blocks);
	// writes all body-body interactions into the d_interactions float pointer
	calc_force_cuda_full<<<blocks_2d, threads_2d>>>(d_bodies, d_interactions_x, d_interactions_y, ss.g, ss.timestep, bodies.size());
	// wait for all calculations to finish
	cudaDeviceSynchronize();
	// reduces the interactions matrices into one float per body
	reduce_interactions<<<bodies.size(), reduce_threads>>>(d_bodies, d_interactions_x, d_interactions_y, bodies.size());
	cudaDeviceSynchronize();
	// run movement calc on the GPU
	calc_movement_cuda<<<n_blocks, n_threads>>>(d_bodies, ss.timestep, bodies.size());

	// copy data back onto the CPU
	const size_t bytes = sizeof(body) * bodies.size();
	cudaMemcpy(bodies.data(), d_bodies, bytes, cudaMemcpyDeviceToHost);

}

#ifdef USE_OCTREE

__device__ __forceinline__ bool in_bounds(cu_octree_node& node, body& test_body)
{

	float box = node.size / 2;
	if (test_body.x >= node.c_x + box || test_body.x < node.c_x - box) return false;
	if (test_body.y >= node.c_y + box || test_body.y < node.c_y - box) return false;
	#ifdef THREED
	if (test_body.x > node.c_x + box || test_body.x < node.c_x - box) return false;
	#endif
	return true;

}

__global__ void build_grid_bottom(body* bodies, cu_octree_node* nodes, size_t* bodies_nodes_assignment, size_t n_bodies, size_t n_nodes, size_t n_nodes_bottom_side, size_t bottom_nodes_idx, float size, float min_x, float min_y)
{

	// initializes the octree nodes that cover the grid at deepest level determined by the user
	// the next function then builds up from that until the root

	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	//int tid_node = bottom_nodes_idx + tid_y * n_nodes_bottom_side + tid_x;  // linear first x then y memory mapping
	int tid_node = bottom_nodes_idx + floorf(static_cast<float>(tid_x) / 2.0) * n_nodes_bottom_side * 2 + floorf(static_cast<float>(tid_y) / 2.0) * 4 + (tid_x % 2) * 2 + tid_y % 2;  // mapping each 2x2 block of quadtree nodes to adjacent spots in linear memory  // n_nodes_bottom_side / 2 * 4, where 4 is the amount of subdivisions (relevant for octree later on)

	if (tid_node < n_nodes)
	{
		// set up the bottom level of the grid
		nodes[tid_node].c_x = min_x + size * (tid_x + 0.5);
		nodes[tid_node].c_y = min_y + size * (tid_y + 0.5);
		nodes[tid_node].is_leaf = true;
		nodes[tid_node].node_id = tid_node;
		nodes[tid_node].m = 0;
		nodes[tid_node].com_x = 0;
		nodes[tid_node].com_y = 0;
		nodes[tid_node].num_bodies = 0;
		for (size_t i = 0; i < 4; i++)
		{
			nodes[tid_node].children[i] = -1;
		}

		// for all bodies check if they fall into this node's cell
		// and update the cell accordingly
		for (size_t it = 0; it < n_bodies; it++)
		{
			if (in_bounds(nodes[tid_node], bodies[it]))
			{
				bodies_nodes_assignment[it] = tid_node;  // the assignment array will only consider nodes at the smallest grid size, i.e. leafs of the octree
				nodes[tid_node].com_x += bodies[it].x * bodies[it].m;
				nodes[tid_node].com_x += bodies[it].y * bodies[it].m;
				nodes[tid_node].m += bodies[it].m;
				nodes[tid_node].num_bodies += 1;
			}
		}

		if (nodes[tid_node].m != 0)
		{
			nodes[tid_node].com_x /= nodes[tid_node].m;
			nodes[tid_node].com_y /= nodes[tid_node].m;
		}
	}

}

__global__ void build_grid_at_level(body* bodies, cu_octree_node* nodes, size_t n_nodes_at_level_side, size_t nodes_at_level_start_idx, size_t nodes_at_next_level_start_idx, float size, float min_x, float min_y)
{

	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	int tid_within_current_grid = floorf(static_cast<float>(tid_x) / 2.0) * n_nodes_at_level_side * 2 + floorf(static_cast<float>(tid_y) / 2.0) * 4 + (tid_x % 2) * 2 + tid_y % 2;
	int tid_node = nodes_at_level_start_idx + tid_within_current_grid;

	if (tid_node >= nodes_at_level_start_idx && tid_node < nodes_at_next_level_start_idx)
	{
		nodes[tid_node].c_x = min_x + size * (tid_x + 0.5);
		nodes[tid_node].c_y = min_y + size * (tid_y + 0.5);
		nodes[tid_node].is_leaf = false;
		nodes[tid_node].node_id = tid_node;
		nodes[tid_node].m = 0;
		nodes[tid_node].com_x = 0;
		nodes[tid_node].com_y = 0;
		nodes[tid_node].num_bodies = 0;
		for (size_t i = 0; i < 4; i++)
		{
			nodes[tid_node].children[i] = nodes_at_next_level_start_idx + 4 * (tid_node - nodes_at_level_start_idx);
			nodes[tid_node].com_x += nodes[nodes[tid_node].children[i]].m * nodes[nodes[tid_node].children[i]].com_x;
			nodes[tid_node].com_y += nodes[nodes[tid_node].children[i]].m * nodes[nodes[tid_node].children[i]].com_y;
			nodes[tid_node].m += nodes[nodes[tid_node].children[i]].m;
		}

		if (nodes[tid_node].m != 0)
		{
			nodes[tid_node].com_x /= nodes[tid_node].m;
			nodes[tid_node].com_y /= nodes[tid_node].m;
		}
	}

	__syncthreads();
	// deal with the root, if we're that far up
	if (tid_node == 1)
	{
		nodes[0].c_x = min_x + size;
		nodes[0].c_y = min_y + size;
		nodes[0].is_leaf = false;
		nodes[0].node_id = 0;
		nodes[0].m = 0;
		nodes[0].com_x = 0;
		nodes[0].com_y = 0;
		nodes[0].num_bodies = 0;

		for (size_t i = 0; i < 4; i++)
		{
			nodes[0].children[i] = nodes_at_next_level_start_idx + 4 * (0 - nodes_at_level_start_idx);
			nodes[0].com_x += nodes[nodes[0].children[i]].m * nodes[nodes[0].children[i]].com_x;
			nodes[0].com_y += nodes[nodes[0].children[i]].m * nodes[nodes[0].children[i]].com_y;
			nodes[0].m += nodes[nodes[0].children[i]].m;
		}

		if (nodes[0].m != 0)
		{
			nodes[0].com_x /= nodes[0].m;
			nodes[0].com_y /= nodes[0].m;
		}
	}

}

__device__ void octree_recursion_cuda(size_t idx, body* bodies, size_t node_idx, cu_octree_node* nodes, const float g, const float t, const int N, const float tolerance)
{

	cu_octree_node node = nodes[node_idx];
	// largely the same as the CPU code in body.cpp
	if (node.is_leaf && node.num_bodies == 0)
	{
		return;
	}

	if (node.is_leaf && in_bounds(node, bodies[idx]))
	{
		for (size_t idx2 = node.body_start_idx; idx2 < node.body_start_idx + node.num_bodies; idx2++)
		{
			float d_x = bodies[idx].x - node.com_x;
			float d_y = bodies[idx].y - node.com_y;
			float dist = d_x * d_x + d_y * d_y;

			if (bodies[idx].r * bodies[idx].r + bodies[idx2].r + bodies[idx2].r >= dist)
				break;

			float dist_inv = rsqrtf(dist);

			float force = -(bodies[idx2].m * g) / dist;
			d_x *= dist_inv;
			d_y *= dist_inv;
			bodies[idx].v_x += d_x * force * t;
			bodies[idx].v_y += d_y * force * t;
		}
	}
	else
	{
		float d_x = bodies[idx].x - node.com_x;
		float d_y = bodies[idx].y - node.com_y;

		float dist = d_x * d_x + d_y * d_y;
		float dist_inv = rsqrtf(dist);

		if (dist_inv * node.size < tolerance)
		{
			if (bodies[idx].r * bodies[idx].r >= dist) return;
			float force = -(node.m * g) / dist;
			d_x *= dist_inv;
			d_y *= dist_inv;
			bodies[idx].v_x += d_x * force * t;
			bodies[idx].v_y += d_y * force * t;
		}
		else if(!node.is_leaf)
		{
			for (size_t idx3 = 0; idx3 < 4; idx3++)
			{
				octree_recursion_cuda(idx, bodies, node.children[idx3], nodes, g, t, N, tolerance);
			}
		}
	}

}

__global__ void octree_calc_force_cuda(body* bodies, cu_octree_node* nodes, const float g, const float t, const int N, const float tolerance)
{

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N)
	{
		octree_recursion_cuda(tid, bodies, 0, nodes, g, t, N, tolerance);
	}

}

__global__ void set_up_sort(sort_help* help, size_t* indices, const int N)
{

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N)
	{
		help[tid].node = indices[tid];
		help[tid].body = tid;
	}

}

__global__ void arrange_bodies(body* bodies_orig, body* bodies_new, cu_octree_node* nodes, sort_help* help, const int N)
{

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N)
	{
		bodies_new[tid] = bodies_orig[help[tid].body];
		nodes[help[tid].node].body_start_idx = tid;
	}

}

void process_bodies_octree_cuda(std::vector<body>& bodies, body* d_bodies, body* d_bodies_bu, cu_octree_node* d_nodes, sort_help* d_help, size_t* d_bodies_idxes, const sim_settings& ss)
{

	// determine the dimensions of the grid the octree will cover
	// such as size, number of nodes and location
	float min_x = FLT_MAX;
	float max_x = -FLT_MAX;
	float min_y = FLT_MAX;
	float max_y = -FLT_MAX;
	#ifdef THREED
	float min_z = FLT_MAX;
	float max_z = -FLT_MAX;
	#endif
	for (const body& it : bodies)
	{
		if (it.x > max_x) max_x = it.x;
		if (it.x < min_x) min_x = it.x;
		if (it.y > max_y) max_y = it.y;
		if (it.y < min_y) min_y = it.y;
		#ifdef THREED
		if (it.z > max_z) max_z = it.z;
		if (it.z < min_z) min_z = it.z;
		#endif
	}
	#ifndef THREED
	float largest_span = std::max(max_x - min_x, max_y - min_y);
	#else
	float largest_span = std::max(max_x - min_x, max_y - min_y, max_z - min_z);
	#endif

	const size_t max_depth = 6;  // temp
	size_t n_total_nodes = 0;
	size_t nodes_per_level[max_depth] = {0};

	for (size_t i = 1; i < max_depth; i++)
	{
		size_t incr = std::pow(4, i);
		nodes_per_level[i] = incr;
		n_total_nodes += incr;
	}
	nodes_per_level[0] = 1;
	n_total_nodes += 1;

	// now initialize all the nodes at the bottom of the octree and assign all bodies to them
	size_t cur_grid_num_cells_side = 2 << max_depth;  // how many nodes there are along one side of the lowest level of the octree
	float cur_grid_size = largest_span / cur_grid_num_cells_side;  // how large one cell at the bottom is in terms of world coordinates
	size_t n_threads_per_dim = 32;
	size_t n_blocks_per_dim = (cur_grid_num_cells_side + n_threads_per_dim - 1) / n_threads_per_dim;
	dim3 threads_2d(n_threads_per_dim, n_threads_per_dim);
	dim3 blocks_2d(n_blocks_per_dim, n_blocks_per_dim);
	size_t cur_level_nodes_start_idx = n_total_nodes - 1 - (cur_grid_num_cells_side << 2);
	build_grid_bottom<<<blocks_2d, threads_2d>>>(d_bodies, d_nodes, d_bodies_idxes, bodies.size(), n_total_nodes, cur_grid_num_cells_side, n_total_nodes - (cur_grid_size << 2) - 1, bottom_grid_cell_size, min_x, min_y);
	cudaDeviceSynchronize();

	// now that the bottom of the tree is complete, we have to sort the bodies such that they're located adjacent to each other in memory per node
	size_t n_threads = 1024;
	size_t n_blocks = (bodies.size() + n_threads - 1) / n_threads;
	set_up_sort<<<n_blocks, n_threads>>>(d_help, d_bodies_idxes, bodies.size());
	cudaDeviceSynchronize();
	// sort via Thrust library
	thrust::sort(d_help, d_help + bodies.size());
	// move the bodies in memory
	arrange_bodies << <n_blocks, n_threads >> > (d_bodies, d_bodies_bu, d_nodes, d_help, bodies.size());
	cudaDeviceSynchronize();
	// swap the pointers between current and backup bodies
	body* temp = d_bodies;
	d_bodies = d_bodies_bu;
	d_bodies_bu = temp;


	// build up the rest of the octree up to the root
	// note: the root is handled by the level == 1 iteration
	for (size_t level = max_depth - 1; level > 0; level--)
	{
		cur_grid_size *= 2;
		cur_grid_num_cells_side = 2 << (level - 1);
		n_blocks_per_dim = (cur_grid_num_cells_side + n_threads_per_dim - 1) / n_threads_per_dim;
		dim3 blocks_2d(n_blocks_per_dim, n_blocks_per_dim);
		build_grid_at_level<<<blocks_2d, threads_2d>>>(d_bodies, d_nodes, cur_grid_num_cells_side, cur_level_nodes_start_idx - (cur_grid_num_cells_side << 2), cur_level_nodes_start_idx, cur_grid_size, min_x, min_y);
		cudaDeviceSynchronize();
		cur_level_nodes_start_idx -= (cur_grid_num_cells_side << 2);
	}

	// finally, run the calculations with the octree on every single body
	size_t n_threads = 1024;
	size_t n_blocks = (bodies.size() + n_threads - 1) / n_threads;
	octree_calc_force_cuda<<<n_blocks, n_threads>>>(d_bodies, d_nodes, ss.g, ss.delta_t, bodies.size(), ss.octree_tolerance);
	cudaDeviceSynchronize();

	// apply movement to bodies
	calc_movement_cuda<<<n_blocks, n_threads>>>(d_bodies, ss.timestep, bodies.size());

	// copy data back onto the CPU
	const size_t bytes = sizeof(body) * bodies.size();
	cudaMemcpy(bodies.data(), d_bodies, bytes, cudaMemcpyDeviceToHost);
}

#endif