//#include <cmath>

#include "body.hpp"

#include <random>
#include <iostream>

#include "../misc/settings.hpp"
#include "../misc/util.hpp"

body* init_bodies(size_t num_bodies, init_policy policy, float min_mass, float max_mass)
{
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> pos_dist(-1, 1);
	std::uniform_real_distribution<float> m_dist(min_mass, max_mass);
	std::normal_distribution<double> pos_dist_normal(0, 0.5);

	body* bodies = new body[num_bodies]; // no free atm because there's no need

	switch (policy)
	{
	case uniform:
		for (size_t i = 0; i < num_bodies; i++)
		{
			bodies[i].x = pos_dist(rng);
			bodies[i].y = pos_dist(rng);
			bodies[i].m = m_dist(rng);
			bodies[i].v_x = 0;
			bodies[i].v_y = 0;
		}
		break;
	case gaussian:
		for (size_t i = 0; i < num_bodies; i++)
		{
			bodies[i].x = pos_dist_normal(rng);
			bodies[i].y = pos_dist_normal(rng);
			bodies[i].m = m_dist(rng);
			bodies[i].v_x = 0;
			bodies[i].v_y = 0;
		}
		break;
	}


	return bodies;
}

void calc_force(body& body1, body& body2, float timestep)
{
	float d_x = body1.x - body2.x;
	float d_y = body1.y - body2.y;

	float dist = d_x * d_x + d_y * d_y; // no square root needed here because the formula for the force squares it anyway

	// Newton's formula
	float force = -(body1.m * body2.m * settings::g) / dist;

	// get the inverse of the distance
	float dist_inv = rsqrt(dist);

	// and normalize the x and y components with it
	d_x *= dist_inv;
	// then multiply the normalized direction component with the force and the timestep and divide by mass to get
	// the velocity component caused by the force in that direction
	body1.v_x += d_x * force * timestep / body1.m;
	body2.v_x -= d_x * force * timestep / body2.m;
	
	// same for y
	d_y *= dist_inv;
	body1.v_y += d_y * force * timestep / body1.m;
	body2.v_y -= d_y * force * timestep / body2.m;
}

void process_bodies(body* bodies, size_t num_bodies, float timestep)
{
	for (size_t i = 0; i < num_bodies - 1; i++)
	{
		for (size_t j = i + 1; j < num_bodies; j++)
		{
			calc_force(bodies[i], bodies[j], timestep);
		}
	}
}

void update_positions(body* bodies, size_t num_bodies, float timestep)
{
	for (size_t i = 0; i < num_bodies; i++)
	{
		bodies[i].x += 0.5 * bodies[i].v_x * timestep;
		bodies[i].y += 0.5 * bodies[i].v_y * timestep;
	}
}
