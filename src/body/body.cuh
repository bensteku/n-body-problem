#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "body.hpp"

void process_bodies_cuda(std::vector<body>& bodies, body* d_bodies, float* d_interactions_x, float* d_interactions_y, const sim_settings& ss);

#ifdef USE_OCTREE

// define CUDA specific version of the octree node here

struct cu_octree_node
{
	
	// struct members copied from CPU octree
	bool is_leaf;

	float com_x, com_y;
	float c_x, c_y;
	#ifdef THREED
	float com_z;
	float c_z;
	#endif
	float size;
	float m;

	// CUDA-specific:
	size_t node_id;  // for identifying this node when we sort the bodies array
	size_t body_start_idx;  // index in the sorted bodies array where bodies belonging to this node start
	size_t num_bodies;  // how far we can go in the bodies array until we start hitting bodies belonging to other nodes

};

#endif