#pragma once

#include <vector>
#include <immintrin.h>
#include <random>
#include <iostream>
#include <limits>
#include <thread>
#include <barrier>
#include <condition_variable>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include "../misc/util.hpp"
#include "../misc/settings.hpp"

#ifndef THREED
struct body
{

	float m;
	float r;

	float x;
	float y;

	float v_x;
	float v_y;

};
#else
struct body
{

	float m;
	float r;

	float x;
	float y;
	float z;

	float v_x;
	float v_y;
	float v_z;

};
#endif



struct sim_settings
{

	float timestep;
	float g;
	size_t n;

	// init settings
	float circle_radius = 25;
	float circle_deviation = 0.0;
	float x_range = 25;
	float y_range = 25;
	#ifdef THREED
	float z_range = 25;
	#endif
	float max_mass = 2000;
	float min_mass = 2000;

};

// initialization methods
void init_bodies_uniform(std::vector<body>& bodies, sim_settings& ss);
void init_bodies_circle(std::vector<body>& bodies, sim_settings& ss);
void init_bodies_normal(std::vector<body>& bodies, sim_settings& ss);

void process_bodies_simd(std::vector<body>& bodies, sim_settings& ss, std::vector<__m256>& x_vec, std::vector<__m256>& y_vec, std::vector<__m256>& m_vec, std::vector<__m256>& r_vec);
void process_bodies(std::vector<body>& bodies, sim_settings& ss);

#ifdef USE_THREADS
// methods when using multi-threading
void process_bodies_mt(std::unique_ptr<std::barrier<>>& compute_barrier1, std::unique_ptr<std::barrier<>>& compute_barrier2, std::unique_ptr<std::barrier<>>& render_barrier, std::atomic<bool>& terminate, bool& run, std::condition_variable& run_cv, std::mutex& run_mtx, size_t thread_id, size_t num_threads, std::vector<body>& bodies, sim_settings& ss);
#endif