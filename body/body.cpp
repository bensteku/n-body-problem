#include "body.hpp"

#include <random>
#include <iostream>

#include "../misc/util.hpp"

/*
	For initialization, the x-axis will always initially be assumed as ranging from -100 to 100.
	The y-axis will be cropped according to the aspect ratio.
*/

std::vector<body> init_bodies_uniform(size_t num_bodies, float min_mass, float max_mass)
{

	// set up rng
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> m_dist(min_mass, max_mass);
	// determine which dimension is bigger
	std::uniform_real_distribution<float> x_uniform(-100, 100);
	std::uniform_real_distribution<float> y_uniform(-100 * settings::aspect_ratio, 100 * settings::aspect_ratio);	

	std::vector<body> bodies;
	bodies.resize(num_bodies);

	for (body& it : bodies)
	{
		it.x = x_uniform(rng);
		it.y = y_uniform(rng);
		it.m = m_dist(rng);
		it.r = it.m / settings::mass_radius_factor;
		it.v_x = 0;
		it.v_y = 0;
	}

	return bodies;

}

std::vector<body> init_bodies_circle(size_t num_bodies, float min_mass, float max_mass, float radius, float deviation)
{

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> m_dist(min_mass, max_mass);
	std::uniform_real_distribution<float> random_radius((1 - deviation) * radius, (1 + deviation) * radius);

	std::vector<body> bodies;
	bodies.resize(num_bodies);

	float theta = 0;
	float t_increment = 2 * settings::pi / num_bodies;
	for (body& it : bodies)
	{
		float r = random_radius(rng);
		it.x = r * cos(theta);
		it.y = r * sin(theta);
		it.m = m_dist(rng);
		it.r = it.m / settings::mass_radius_factor;
		it.v_x = 0;
		it.v_y = 0;
		theta += t_increment;
	}

	return bodies;

}

std::vector<body> init_bodies_normal(size_t num_bodies, float min_mass, float max_mass, float center_x, float center_y, float std_x, float std_y)
{

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> m_dist(min_mass, max_mass);
	std::normal_distribution<float> x_normal(center_x, std_x);
	std::normal_distribution<float> y_normal(center_y, std_y);

	std::vector<body> bodies;
	bodies.resize(num_bodies);
	
	for (body& it : bodies)
	{
		it.x = x_normal(rng);
		it.y = y_normal(rng);
		it.m = m_dist(rng);
		it.r = it.m / settings::mass_radius_factor;
		it.v_x = 0;
		it.v_y = 0;
	}

	return bodies;

}

void calc_force(body& body1, body& body2, float timestep)
{

	float d_x = body1.x - body2.x;
	float d_y = body1.y - body2.y;

	float dist = d_x * d_x + d_y * d_y; // no square root needed here because the formula for the force squares it anyway

	float force;
	if ((body1.r + body2.r) * (body1.r + body2.r) >= dist)
	{
		// in case of collision, to avoid NaNs and infinite acceleration
		force = 0;
	}
	else
	{
		// Newton's formula
		force = -(body1.m * body2.m * settings::g) / dist;
	}
	
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

void process_bodies(std::vector<body>& bodies, float timestep)
{

	for (auto it = bodies.begin(); it != bodies.end() - 1; ++it)
	{
		for (auto it2 = it + 1; it2 != bodies.end(); ++it2)
		{
			calc_force(*it, *it2, timestep);
		}
	}

	for (body& it : bodies)
	{
		it.x += 0.5 * it.v_x * timestep;
		it.y += 0.5 * it.v_y * timestep;
	}

}

void process_bodies_simd(std::vector<body>& bodies, float timestep)
{

}