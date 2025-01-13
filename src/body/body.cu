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

__global__ void build_grid_bottom(body* bodies, cu_octree_node* nodes, size_t* bodies_nodes_assignment, size_t n_bodies, size_t n_nodes, size_t n_nodes_bottom, size_t bottom_nodes_idx, float size, float min_x, float max_x, float min_y, float max_y)
{

	// initializes the octree nodes that cover the grid at deepest level determined by the user
	// the next function then builds up from that until the root

	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	int tid_node = tid_y * n_nodes_bottom + tid_x;

	if (tid_node < n_nodes)
	{
		// set up the bottom level of the grid
		nodes[bottom_nodes_idx + tid_node].c_x = size * (tid_x + 0.5);
		nodes[bottom_nodes_idx + tid_node].c_y = size * (tid_y + 0.5);
		nodes[bottom_nodes_idx + tid_node].is_leaf = true;
		nodes[bottom_nodes_idx + tid_node].node_id = bottom_nodes_idx + tid_node;
		nodes[bottom_nodes_idx + tid_node].m = 0;
		nodes[bottom_nodes_idx + tid_node].com_x = 0;
		nodes[bottom_nodes_idx + tid_node].com_y = 0;
		nodes[bottom_nodes_idx + tid_node].num_bodies = 0;

		// for all bodies check if they fall into this node's cell
		// and update the cell accordingly
		for (size_t it = 0; it < n_bodies; it++)
		{
			if (in_bounds(nodes[bottom_nodes_idx + tid_node], bodies[it]))
			{
				bodies_nodes_assignment[it] = tid_node;  // the assignment array will only consider nodes at the smallest grid size, i.e. leafs of the octree
				nodes[bottom_nodes_idx + tid_node].com_x += bodies[it].x * bodies[it].m;
				nodes[bottom_nodes_idx + tid_node].m += bodies[it].m;
				nodes[bottom_nodes_idx + tid_node].num_bodies += 1;
			}
		}

		if (nodes[bottom_nodes_idx + tid_node].m != 0)
		{
			nodes[bottom_nodes_idx + tid_node].com_x /= nodes[bottom_nodes_idx + tid_node].m;
			nodes[bottom_nodes_idx + tid_node].com_y /= nodes[bottom_nodes_idx + tid_node].m;
		}
	}

}

void process_bodies_octree_cuda(std::vector<body>& bodies, body* d_bodies, cu_octree_node* d_nodes, size_t* d_bodies_idxes, const sim_settings& ss)
{

	const size_t max_depth = 4;  // temp
	size_t n_total_nodes = 0;
	size_t nodes_per_level[max_depth] = {0};

	for (size_t i = 1; i < max_depth; i++)
	{
		size_t incr = std::pow(4, i);
		nodes_per_level[i] = incr;
		n_total_nodes += incr;
	}

	size_t bottom_grid_side_size = 2 << max_depth;
	size_t n_threads_per_dim = 32;
	size_t n_blocks_per_dim = (bottom_grid_side_size + n_threads_per_dim - 1) / n_threads_per_dim;

	dim3 threads_2d(n_threads_per_dim, n_threads_per_dim);
	dim3 blocks_2d(n_blocks_per_dim, n_blocks_per_dim);

	build_grid_bottom<<<blocks_2d, threads_2d>>>(d_bodies, d_nodes, bodies.size(), n_total_nodes, bottom_grid_side_size, n_total_nodes - (bottom_grid_side_size << 2))

}

#endif