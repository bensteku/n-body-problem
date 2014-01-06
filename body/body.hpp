#pragma once

struct body
{
	float m;
	float r;

	float x;
	float y;

	float v_x;
	float v_y;
};

enum init_policy {
	uniform,
	gaussian,
	circle
};

void calc_force(body& body1, body& body2, float timestep);
void process_bodies(body* bodies, size_t num_bodies, float timestep);
void update_positions(body* bodies, size_t num_bodies, float timestep);

body* init_bodies(size_t num_bodies, init_policy policy, float min_mass, float max_mass);