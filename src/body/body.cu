#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "body.hpp"

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
	int tid = tid_x + tid_y;

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

	int n_threads = 32;
	int n_blocks = (bodies.size() + n_threads - 1) / n_threads;

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