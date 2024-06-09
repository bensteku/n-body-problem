#pragma once

#include <vector>
#include <immintrin.h>

#include "../misc/settings.hpp"

struct body
{

	float m;
	float r;

	float x;
	float y;

	float v_x;
	float v_y;

};

struct processing_args
{

	float timestep;
	float g;
	size_t n;

};

// initialization methods
void init_bodies_uniform(std::vector<body>& bodies, float min_mass, float max_mass);
void init_bodies_circle(std::vector<body>& bodies, float min_mass, float max_mass, float radius = 50, float deviation = 0);
void init_bodies_normal(std::vector<body>& bodies, float min_mass, float max_mass, float center_x = 0, float center_y = 0, float std_x = 25, float std_y = 25);

void process_bodies_simd(std::vector<body>& bodies, processing_args& pa, __m256* x_vec, __m256* y_vec, __m256* m_vec, __m256* r_vec);
void process_bodies(std::vector<body>& bodies, processing_args& pa);