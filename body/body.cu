#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "body.hpp"

__global__ void calc_force_cuda(float* x, float* y, float* m, float* r, float* v_x, float* v_y, float g, float t, int N, int it)
{

	// global thread ID using the formula from the CUDA docs
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// check if this is actually a valid calcuation
	if (tid < N)
	{
		//printf("x %f, y %f\n", x[0], y[0]);
		//runs Newton's formula
		float dx = x[it] - x[tid];
		float dy = y[it] - y[tid];
		//printf("dx %f, dy %f\n", dx, dy);

		float dist = dx * dx + dy * dy;
		//printf("dist %f\n", dist);

		float force, dist_inv;
		if ((r[it] + r[tid]) * (r[it] + r[tid]) >= dist)
		{
			force = 0;
			dist_inv = 0;
		}
		else
		{
			force = -(m[it] * m[tid] * g) / dist;
			dist_inv = rsqrtf(dist);
		}
		//printf("force %f\n", force);

		//float dist_inv = rsqrtf(dist);  // ignore the squiggle lines, this is a CUDA inbuilt function
		//printf("distinv %f\n", dist_inv);
		dx *= dist_inv;

		// no writing on the iterator output field
		// a) would not work anyway, because thousands of GPU threads will try to do it simultaneously and corrupt the result and
		// b) we get the correct result anyway, because when the iterator changes the previous iterator's values will be changed then
		//v_x[it] += dx * force * t / m[it];
		v_x[tid] -= dx * force * t / m[tid];

		dy *= dist_inv;

		//v_y[it] += dy * force * t / m[it];
		v_y[tid] -= dy * force * t / m[tid];
	}

}

void process_bodies_cuda(std::vector<body>& bodies,
	std::vector<float>& x,
	std::vector<float>& y,
	std::vector<float>& mass,
	std::vector<float>& radius,
	std::vector<float>& v_x,
	std::vector<float>& v_y,
	float* d_x,
	float* d_y,
	float* d_mass,
	float* d_radius,
	float* d_v_x,
	float* d_v_y,
	sim_settings& ss)
{

	// read in the data from the bodies struct
	// TODO: think about changing the bodies struct (and all the code it touches) so that this overhead is not necessary
	for (size_t i = 0; i < bodies.size(); i++)
	{
		x[i] = bodies[i].x;
		y[i] = bodies[i].y;
		mass[i] = bodies[i].m;
		radius[i] = bodies[i].r;
		v_x[i] = bodies[i].v_x;
		v_y[i] = bodies[i].v_y;
	}
	// memcopy the data onto the GPU
	const size_t bytes = sizeof(float) * bodies.size();
	cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_radius, radius.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v_x, v_x.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v_y, v_y.data(), bytes, cudaMemcpyHostToDevice);

	int n_threads = 1024;
	int n_blocks = (bodies.size() + n_threads - 1) / n_threads;

	for (size_t i = 0; i < bodies.size(); i++)
	{
		calc_force_cuda <<<n_blocks, n_threads >>> (d_x, d_y, d_mass, d_radius, d_v_x, d_v_y, ss.g, ss.timestep, bodies.size(), i);
	}
	// copy data back onto the CPU
	cudaMemcpy(v_x.data(), d_v_x, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(v_y.data(), d_v_y, bytes, cudaMemcpyDeviceToHost);
	// update body velocities and positions
	for (size_t i = 0; i < bodies.size(); i++)
	{
		bodies[i].v_x = v_x[i];
		bodies[i].x += 0.5 * v_x[i] * ss.timestep;
		bodies[i].v_y = v_y[i];
		bodies[i].y += 0.5 * v_y[i] * ss.timestep;
	}
	//exit(0);

}