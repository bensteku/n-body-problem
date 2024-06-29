#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "body.hpp"

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

__global__ void calc_force_cuda_full(body* bodies, float g, float t, int N)
{

	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = tid_x + tid_y;

	if (tid_x < N && tid_y < N)
	{
		//runs Newton's formula
		float dx = bodies[tid_x].x - bodies[tid_y].x;
		float dy = bodies[tid_x].y - bodies[tid_y].y;
		if (tid < 0)
			printf("%f %f", dx, dy);

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

		dx *= dist_inv;
		// no writing on the iterator output field
		// a) would not work anyway, because thousands of GPU threads will try to do it simultaneously and corrupt the result and
		// b) we get the correct result anyway, because when the iterator changes the previous iterator's values will be changed then
		//v_x[it] += dx * force * t / m[it];
		bodies[tid_y].v_x -= dx * force * t / bodies[tid_y].m;

		dy *= dist_inv;
		//v_y[it] += dy * force * t / m[it];
		bodies[tid_y].v_y -= dy * force * t / bodies[tid_y].m;
	}

}

__global__ void calc_movement_cuda(body* bodies, float t, int N)
{

	// global thread ID using the formula from the CUDA docs
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N)
	{
		bodies[tid].x += 0.5 * bodies[tid].v_x * t;
		bodies[tid].y += 0.5 * bodies[tid].v_y * t;
	}	

}

void process_bodies_cuda(std::vector<body>& bodies,	body* d_bodies,	sim_settings& ss)
{

	int n_threads = 32;
	int n_blocks = (bodies.size() + n_threads - 1) / n_threads;

	/*
	// run calculation on the GPU
	for (size_t i = 0; i < bodies.size(); i++)
	{
		calc_force_cuda<<<n_blocks, n_threads>>>(d_bodies, ss.g, ss.timestep, bodies.size(), i);
	}
	*/
	dim3 threads_2d(n_threads, n_threads);
	dim3 blocks_2d(n_blocks, n_blocks);
	calc_force_cuda_full<<<blocks_2d, threads_2d>>> (d_bodies, ss.g, ss.timestep, bodies.size());
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
	}
	// wait for all calculations to finish
	cudaDeviceSynchronize();
	// run movement calc on the GPU
	calc_movement_cuda<<<n_blocks, n_threads>>>(d_bodies, ss.timestep, bodies.size());

	// copy data back onto the CPU
	const size_t bytes = sizeof(body) * bodies.size();
	cudaMemcpy(bodies.data(), d_bodies, bytes, cudaMemcpyDeviceToHost);

}