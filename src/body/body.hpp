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
#ifdef USE_OCTREE
	size_t octree_max_node_size = 20;
	float octree_tolerance = 0.5;
#endif

};

// initialization methods
void init_bodies_uniform(std::vector<body>& bodies, sim_settings& ss);
void init_bodies_circle(std::vector<body>& bodies, sim_settings& ss);
void init_bodies_normal(std::vector<body>& bodies, sim_settings& ss);

void process_bodies_simd(std::vector<body>& bodies, sim_settings& ss, std::vector<__m256>& x_vec, std::vector<__m256>& y_vec, std::vector<__m256>& m_vec, std::vector<__m256>& r_vec);
void process_bodies(std::vector<body>& bodies, sim_settings& ss);

#ifdef USE_THREADS
	// methods when using multi-threading
	#ifdef USE_SIMD
		void process_bodies_simd_mt(std::unique_ptr<std::barrier<>>& compute_barrier1, std::unique_ptr<std::barrier<>>& compute_barrier2, std::unique_ptr<std::barrier<>>& render_barrier, std::atomic<bool>& terminate, bool& run, std::condition_variable& run_cv, std::mutex& run_mtx, size_t thread_id, size_t num_threads,	std::vector<body>& bodies, sim_settings& ss, std::vector<__m256>& x_vec, std::vector<__m256>& y_vec, std::vector<__m256>& mass_vec, std::vector<__m256>& r_vec);
	#else
		void process_bodies_mt(std::unique_ptr<std::barrier<>>& compute_barrier1, std::unique_ptr<std::barrier<>>& compute_barrier2, std::unique_ptr<std::barrier<>>& render_barrier, std::atomic<bool>& terminate, bool& run, std::condition_variable& run_cv, std::mutex& run_mtx, size_t thread_id, size_t num_threads, std::vector<body>& bodies, sim_settings& ss);
	#endif
#endif

#ifdef USE_OCTREE
	// node for an octree data structure
	struct octree  // also doubles as quadtree for the 2D case
	{
		
		size_t max_occupancy;
		bool is_leaf;

		float com_x, com_y;
		float c_x, c_y;
		#ifdef THREED
			float com_z;
			float c_z;
		#endif
		float size;
		float m;

		std::vector<const body*> bodies;

		// children order:
		// 0: nw, 1: ne, 2: sw, 3: se,
		// for 3D: 4: nw2, 5: ne2, 6: sw2, 7: se2
		#ifndef THREED
			std::array<octree*, 4> children;
		#else
			std::array<octree*, 8> children;
		#endif

	#ifndef THREED
		octree(float x, float y, float s, size_t max_occupancy)
			: c_x(x), c_y(y), size(s), max_occupancy(max_occupancy), is_leaf(true)
	#else
		octree(float x, float y, float z, float s, size_t max_occupancy)
			: c_x(x), c_y(y), c_z(z), size(s), max_occupancy(max_occupancy), isleaf(true)
	#endif
		{

			for (octree*& oc : children) oc = nullptr;
			
		}

		~octree()
		{

			for (octree* oc : children) delete oc;

		}

		void build(std::vector<body>& start_bodies, sim_settings& ss);
	#ifdef USE_SIMD
		void calc_force_simd();
	#elif defined(USE_CUDA)

	#else
		void calc_force(body& test_body, sim_settings& ss);
	#endif

		private:
			void subdivide(sim_settings& ss);
			void update_com();
			bool in_bounds(const body* test_body) const;

	};
#endif