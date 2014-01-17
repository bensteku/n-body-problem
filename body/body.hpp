#pragma once

#include <vector>

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

// initialization methods
std::vector<body> init_bodies_uniform(size_t num_bodies, float min_mass, float max_mass);
std::vector<body> init_bodies_circle(size_t num_bodies, float min_mass, float max_mass, float radius = 50, float deviation = 0);
std::vector<body> init_bodies_normal(size_t num_bodies, float min_mass, float max_mass, float center_x = 0, float center_y = 0, float std_x = 25, float std_y = 25);

void calc_force(body& body1, body& body2, float timestep);
void process_bodies(std::vector<body>& bodies, float timestep);
void update_positions(std::vector<body>& bodies, float timestep);