#include "body.hpp"

#include <random>
#include <iostream>
#include <limits>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include "../misc/util.hpp"

/*
	For initialization, the x-axis will always initially be assumed as ranging from -100 to 100.
	The y-axis will be cropped according to the aspect ratio.
*/

void init_bodies_uniform(std::vector<body>& bodies, sim_settings& ss)
{

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> m_dist(ss.min_mass, ss.max_mass);
	std::uniform_real_distribution<float> x_uniform(-ss.x_range, ss.x_range);
	std::uniform_real_distribution<float> y_uniform(-ss.y_range * settings::aspect_ratio, ss.y_range * settings::aspect_ratio);	
	#ifdef THREED
	std::uniform_real_distribution<float> z_uniform(-ss.z_range, ss.z_range);
	#endif

	for (body& it : bodies)
	{
		it.x = x_uniform(rng);
		it.y = y_uniform(rng);
		it.m = m_dist(rng);
		it.r = it.m / settings::mass_radius_factor;
		it.v_x = 0;
		it.v_y = 0;
		#ifdef THREED
		it.z = z_uniform(rng);
		it.v_z = 0;
		#endif
	}

}

void init_bodies_circle(std::vector<body>& bodies, sim_settings& ss)
{

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> m_dist(ss.min_mass, ss.max_mass);
	std::uniform_real_distribution<float> random_radius((1 - ss.circle_deviation) * ss.circle_radius, (1 + ss.circle_deviation) * ss.circle_radius);

	#ifndef THREED
	float theta = 0;
	const float t_increment = 2 * settings::pi / bodies.size();
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
	#else
	constexpr float phi = settings::pi * 0.763932;
	for (int i = 0; i < bodies.size(); ++i)
	{
		float r = random_radius(rng);

		float y = 1 - (i / static_cast<float>(bodies.size() - 1)) * 2;
		float r_norm = std::sqrt(1 - y * y);
		float theta = phi * i;

		float x = cos(theta) * r_norm;
		float z = sin(theta) * r_norm;

		bodies[i].x = r * x;
		bodies[i].y = r * y;
		bodies[i].z = r * z;

		bodies[i].m = m_dist(rng);
		bodies[i].r = bodies[i].m / settings::mass_radius_factor;

		bodies[i].v_x = 0;
		bodies[i].v_y = 0;
		bodies[i].v_z = 0;
	}
	#endif

}

void init_bodies_normal(std::vector<body>& bodies, sim_settings& ss)
{

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> m_dist(ss.min_mass, ss.max_mass);
	std::normal_distribution<float> x_normal(0, ss.x_range);
	std::normal_distribution<float> y_normal(0, ss.y_range);
	#ifdef THREED
	std::normal_distribution<float> z_normal(0, ss.z_range);
	#endif

	for (body& it : bodies)
	{
		it.x = x_normal(rng);
		it.y = y_normal(rng);
		it.m = m_dist(rng);
		it.r = it.m / settings::mass_radius_factor;
		it.v_x = 0;
		it.v_y = 0;
		#ifdef THREED
		it.z = z_normal(rng);
		it.v_z = 0;
		#endif
	}

}

void calc_force(body& body1, body& body2, sim_settings& ss)
{

	float d_x = body1.x - body2.x;
	float d_y = body1.y - body2.y;

	#ifndef THREED
	float dist = d_x * d_x + d_y * d_y;  // no square root needed here because the formula for the force squares it anyway
	#else
	float d_z = body1.z - body2.z;
	float dist = d_x * d_x + d_y * d_y + d_z * d_z;
	#endif

	float force;
	if ((body1.r + body2.r) * (body1.r + body2.r) >= dist)
	{
		// in case of collision, to avoid NaNs and infinite acceleration
		force = 0;
	}
	else
	{
		// Newton's formula
		force = -(body1.m * body2.m * ss.g) / dist;
	}
	
	// get the inverse of the distance
	float dist_inv = rsqrt(dist);

	// and normalize the x and y components with it
	d_x *= dist_inv;
	// then multiply the normalized direction component with the force and the timestep and divide by mass to get
	// the velocity component caused by the force in that direction
	body1.v_x += d_x * force * ss.timestep / body1.m;
	body2.v_x -= d_x * force * ss.timestep / body2.m;
	
	// same for y
	d_y *= dist_inv;
	body1.v_y += d_y * force * ss.timestep / body1.m;
	body2.v_y -= d_y * force * ss.timestep / body2.m;

	#ifdef THREED
	// same for z
	body1.v_z += d_z * force * ss.timestep / body1.m;
	body1.v_z -= d_z * force * ss.timestep / body2.m;
	#endif

}

void process_bodies_simd(std::vector<body>& bodies, sim_settings& ss, std::vector<__m256>& x_vec, std::vector<__m256>& y_vec, std::vector<__m256>& mass_vec, std::vector<__m256>& r_vec)
{

	const size_t num_elements = bodies.size();
	const size_t num_packed_elements = x_vec.size();
	const size_t num_elements_over = num_packed_elements * 8 - num_elements;

	// init registers (maybe move this outside so this is only done once)
	// single and tmp registers for the element the iteration is currently at
	const __m256 g_reg = _mm256_set_ps(ss.g, ss.g, ss.g, ss.g, ss.g, ss.g, ss.g, ss.g);	
	const __m256 delta_t_reg = _mm256_set_ps(ss.timestep, ss.timestep, ss.timestep, ss.timestep, ss.timestep, ss.timestep, ss.timestep, ss.timestep);
	

	// load current values into the register arrays
	for (size_t i = 0; i < num_packed_elements; i++)
	{
		// array of indices to ensure that we don't access invalid indices in case the bodies array's size is not divisible by 8
		const size_t indices[8] = { std::min(i * 8, num_elements - 1), std::min(i * 8 + 1, num_elements - 1), std::min(i * 8 + 2, num_elements - 1), std::min(i * 8 + 3, num_elements - 1), std::min(i * 8 + 4, num_elements - 1), std::min(i * 8 + 5, num_elements - 1), std::min(i * 8 + 6, num_elements - 1), std::min(i * 8 + 7, num_elements - 1) };
		x_vec[i] = _mm256_set_ps(bodies[indices[7]].x, bodies[indices[6]].x, bodies[indices[5]].x, bodies[indices[4]].x, bodies[indices[3]].x, bodies[indices[2]].x, bodies[indices[1]].x, bodies[indices[0]].x);
		y_vec[i] = _mm256_set_ps(bodies[indices[7]].y, bodies[indices[6]].y, bodies[indices[5]].y, bodies[indices[4]].y, bodies[indices[3]].y, bodies[indices[2]].y, bodies[indices[1]].y, bodies[indices[0]].y);
		#ifdef THREED
		z_vec[i] = _mm256_set_ps(bodies[indices[7]].z, bodies[indices[6]].z, bodies[indices[5]].z, bodies[indices[4]].z, bodies[indices[3]].z, bodies[indices[2]].z, bodies[indices[1]].z, bodies[indices[0]].z);
		#endif
		mass_vec[i] = _mm256_set_ps(bodies[indices[7]].m, bodies[indices[6]].m, bodies[indices[5]].m, bodies[indices[4]].m, bodies[indices[3]].m, bodies[indices[2]].m, bodies[indices[1]].m, bodies[indices[0]].m);
		r_vec[i] = _mm256_set_ps(bodies[indices[7]].r, bodies[indices[6]].r, bodies[indices[5]].r, bodies[indices[4]].r, bodies[indices[3]].r, bodies[indices[2]].r, bodies[indices[1]].r, bodies[indices[0]].r);
	}

	for (int j = 0; j < num_elements; j++)
	{
		__m256 nan_check_reg;
		__m256 size_check_reg;
		__m256 mass_it_reg;
		__m256 x_it_reg;
		__m256 y_it_reg;
		__m256 r_it_reg;
		__m256 r_add_it_reg;
		__m256 dx_reg;
		__m256 dx_squ_reg;
		__m256 dy_reg;
		__m256 dy_squ_reg;
		__m256 dist_reg;
		__m256 force_reg;
		__m256 vel_reg;
		// load the current body's values into the tmp registers
		mass_it_reg = _mm256_set_ps(bodies[j].m, bodies[j].m, bodies[j].m, bodies[j].m, bodies[j].m, bodies[j].m, bodies[j].m, bodies[j].m);
		x_it_reg = _mm256_set_ps(bodies[j].x, bodies[j].x, bodies[j].x, bodies[j].x, bodies[j].x, bodies[j].x, bodies[j].x, bodies[j].x);
		y_it_reg = _mm256_set_ps(bodies[j].y, bodies[j].y, bodies[j].y, bodies[j].y, bodies[j].y, bodies[j].y, bodies[j].y, bodies[j].y);
		r_it_reg = _mm256_set_ps(bodies[j].r, bodies[j].r, bodies[j].r, bodies[j].r, bodies[j].r, bodies[j].r, bodies[j].r, bodies[j].r);

		// run the formulas for 8 bodies at a time
		for (int i = 0; i < num_packed_elements; i++)
		{
			// set up mask for nulling out duplicate bodies at the end of the array (only happens if the amount of bodies is not divisible by 8)
			// I suppose this is not the smartest way of doing this, but I can't be bothered to look up how to do it better
			if (num_elements_over != 0 and i == num_packed_elements - 1)
			{
				float tmp[8] = { 0.0 };
				for (size_t j = 0; j < 8 - num_elements_over; j++)
				{
					tmp[j] = 1.0;
				}
				size_check_reg = _mm256_set_ps(tmp[7], tmp[6], tmp[5], tmp[4], tmp[3], tmp[2], tmp[1], tmp[0]);
			}
			else
			{
				size_check_reg = _mm256_set1_ps(1.0);
			}

			// distances
			dx_reg = _mm256_sub_ps(x_it_reg, x_vec[i]);
			dx_squ_reg = _mm256_mul_ps(dx_reg, dx_reg);
			dy_reg = _mm256_sub_ps(y_it_reg, y_vec[i]);
			dy_squ_reg = _mm256_mul_ps(dy_reg, dy_reg);
			dist_reg = _mm256_add_ps(dx_squ_reg, dy_squ_reg);

			// collision check to avoid NaNs
			// this will conveniently also zero out all calculations between
			// the iterator element and its version from the overall array
			// add up the two radiuses
			r_add_it_reg = _mm256_add_ps(r_it_reg, r_vec[i]);
			// square them to make them equivalent to the squared distances above
			r_add_it_reg = _mm256_mul_ps(r_add_it_reg, r_add_it_reg);
			// create a mask that is 0 for all body pairs whose distance is smaller than the sum of their radii
			nan_check_reg = _mm256_cmp_ps(r_add_it_reg, dist_reg, _CMP_LT_OQ);

			// calculate force (using the force register as a tmp as well for the intermediate steps)
			force_reg = _mm256_mul_ps(mass_it_reg, mass_vec[i]);
			force_reg = _mm256_mul_ps(force_reg, g_reg);
			force_reg = _mm256_div_ps(force_reg, dist_reg);
			// null out all entries based on our mask
			force_reg = _mm256_and_ps(force_reg, nan_check_reg);
			// null out values that don't exist (because our last register has more values than are left over in bodies (in case it's not divisible by 8))
			force_reg = _mm256_mul_ps(force_reg, size_check_reg);

			// get the inverse square root of the distance
			dist_reg = _mm256_invsqrt_ps(dist_reg);
			// the inverse square root puts out nans/infs for 0, so we have to deal with that
			//dist_reg = _mm256_and_ps(dist_reg, _mm256_xor_ps(is_infinity(dist_reg), _mm256_castsi256_ps(_mm256_set1_epi64x(-1))));
			dist_reg = _mm256_and_ps(dist_reg, nan_check_reg);  // should achieve the same purpose as the line above
			// normalize the x and y components with it
			dx_reg = _mm256_mul_ps(dx_reg, dist_reg);
			dy_reg = _mm256_mul_ps(dy_reg, dist_reg);

			// calculating the velocity
			// x, for the iterator body
			vel_reg = _mm256_mul_ps(force_reg, dx_reg);
			vel_reg = _mm256_mul_ps(vel_reg, delta_t_reg);
			dx_reg = _mm256_div_ps(vel_reg, mass_it_reg);  // overwriting the dx_reg, as vel_reg already contains the results we need it for
			bodies[j].v_x -= reduce_register(dx_reg);
			// y, for the iterator body
			vel_reg = _mm256_mul_ps(force_reg, dy_reg);
			vel_reg = _mm256_mul_ps(vel_reg, delta_t_reg);
			dy_reg = _mm256_div_ps(vel_reg, mass_it_reg);  // same as above
			bodies[j].v_y -= reduce_register(dy_reg);
		}
	}

	// the compiler vectorizes this on its own, looking at the assembly
	for (body& it : bodies)
	{
		it.x += 0.5 * it.v_x * ss.timestep;
		it.y += 0.5 * it.v_y * ss.timestep;
	}

}

void process_bodies(std::vector<body>& bodies, sim_settings& ss)
{
	
	for (auto it = bodies.begin(); it != bodies.end() - 1; it++)
	{
		for (auto it2 = it + 1; it2 != bodies.end(); it2++)
		{
			calc_force(*it, *it2, ss);
		}
	}
	
	for (body& it : bodies)
	{
		it.x += 0.5 * it.v_x * ss.timestep;
		it.y += 0.5 * it.v_y * ss.timestep;
	}

}