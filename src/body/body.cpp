#include "body.hpp"

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

	// registers for constants
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
#ifdef USE_OCTREE
	// get min max of current bodies
	float min_x = FLT_MAX;
	float max_x = -FLT_MAX;
	float min_y = FLT_MAX;
	float max_y = -FLT_MAX;
	#ifdef THREED
		float min_z = FLT_MAX;
		float max_z = -FLT_MAX;
	#endif
	for (const body& it : bodies)
	{
		if (it.x > max_x) max_x = it.x;
		if (it.x < min_x) min_x = it.x;
		if (it.y > max_y) max_y = it.y;
		if (it.y < min_y) min_y = it.y;
	#ifdef THREED
		if (it.z > max_z) max_z = it.z;
		if (it.z < min_z) min_z = it.z;
	#endif
	}
	#ifndef THREED
		float size = std::max(max_x - min_x, max_y - min_y) + 0.005;
	#else
		float size = std::max(max_x - min_x, max_y - min_y, max_z - min_z);
	#endif

	// create octree that is adapted to the current simulation space
	octree* it_octree = 
	#ifndef THREED
		new octree(min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2, size, ss.octree_max_node_size);
	#else
		new octree(min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2, min_z + (max_z - min_z) / 2, size, ss.octree_max_node_size);
	#endif
	it_octree->build(bodies, ss);

	// run calculations
	for (body& it : bodies)
	{
		it_octree->calc_force(it, ss);
	}

	// destroy octree
	delete it_octree;

#else
	for (auto it = bodies.begin(); it != bodies.end() - 1; it++)
	{
		for (auto it2 = it + 1; it2 != bodies.end(); it2++)
		{
			calc_force(*it, *it2, ss);
		}
	}
#endif
	
	for (body& it : bodies)
	{
		it.x += 0.5 * it.v_x * ss.timestep;
		it.y += 0.5 * it.v_y * ss.timestep;
	}

}

#ifdef USE_THREADS

void process_bodies_mt(std::unique_ptr<std::barrier<>>& compute_barrier1, std::unique_ptr<std::barrier<>>& compute_barrier2, std::unique_ptr<std::barrier<>>& render_barrier, 
					   std::atomic<bool>& terminate, bool& run,
					   std::condition_variable& run_cv, std::mutex& run_mtx,
					   size_t thread_id, size_t num_threads,
					   std::vector<body>& bodies, sim_settings& ss)
{
	
	// store last bodies size in order to avoid having to continually recalculate the work indices if nothing has changed
	size_t last_size = 0;
	size_t start_body1, start_body2;
	size_t interactions_per_thread;
	size_t bodies_per_thread;  // for the velocity calculation

	while (!terminate)
	{
		// mutex and condition variable guard to ensure that the threads don't do anything as long as the simulation isn't running
		{
			std::unique_lock<std::mutex> lock(run_mtx);
			run_cv.wait(lock, [&run] {return run; });
		}	

		// work section
		if (!terminate)
		{
			// run calculations that determine which bodies this thread will work on
			if (last_size == 0 || last_size != bodies.size())
			{
				size_t num_interactions = bodies.size() * (bodies.size() - 1) / 2;
				interactions_per_thread = std::ceil(static_cast<float>(num_interactions) / num_threads);
				bodies_per_thread = std::ceil(static_cast<float>(bodies.size()) / num_threads);
				size_t first_interaction_id = thread_id * interactions_per_thread;
				size_t counter = 0;
				last_size = bodies.size();
				bool found = false;
				for (size_t i = 0; i < bodies.size() - 1; i++)
				{
					for (size_t j = i + 1; j < bodies.size(); j++)
					{
						if (counter == first_interaction_id)
						{
							found = true;
							start_body1 = i;
							start_body2 = j;
							break;
						}
						counter += 1;
					}
					if (found)
						break;
				}
			}

			// start calculations
			int calculated = 0;
			size_t body1 = start_body1;
			size_t body2 = start_body2;
			while (calculated < interactions_per_thread)
			{
				calc_force(bodies[body1], bodies[body2], ss);
				if (body2 == bodies.size() - 1)
				{
					body1 += 1;
					body2 = body1 + 1;
				}
				else
				{
					body2 += 1;
				}
				if (body1 == bodies.size() - 1)
				{
					// when this is the case, this thread has run out of bodies to calculate
					calculated = interactions_per_thread;
				}
				calculated += 1;
			}
		}

		// wait for all other threads to finish their gravity calculations
		if (!terminate)
		{
			compute_barrier1->arrive_and_wait();
		}
		else
		{
			break;
		}

		// run the velocity calculation in parallel
		for (size_t idx = thread_id * bodies_per_thread; idx < std::min((thread_id + 1) * bodies_per_thread, bodies.size()); idx++)
		{
			bodies[idx].x += 0.5 * bodies[idx].v_x * ss.timestep;
			bodies[idx].y += 0.5 * bodies[idx].v_y * ss.timestep;
		}

		// send signal to main thread
		if (!terminate)
		{
			compute_barrier2->arrive_and_wait();
		}
		else
		{
			break;
		}

		// wait until rendering is done
		if (!terminate)
		{
			render_barrier->arrive_and_wait();
		}
		else
		{
			break;
		}
	}
	
}

void process_bodies_simd_mt(std::unique_ptr<std::barrier<>>& compute_barrier1, std::unique_ptr<std::barrier<>>& compute_barrier2, std::unique_ptr<std::barrier<>>& render_barrier,
							std::atomic<bool>& terminate, bool& run,
							std::condition_variable& run_cv, std::mutex& run_mtx,
							size_t thread_id, size_t num_threads,
							std::vector<body>& bodies, sim_settings& ss,
							std::vector<__m256>& x_vec, std::vector<__m256>& y_vec, std::vector<__m256>& mass_vec, std::vector<__m256>& r_vec)
{

	// store last bodies size in order to avoid having to continually recalculate the work indices if nothing has changed
	size_t last_size = 0;
	size_t num_elements_over;
	size_t num_packed_elements;
	size_t num_packed_elements_per_thread, num_elements_per_thread;
	size_t packed_start_idx, body_start_idx;
	size_t bodies_per_thread;  // for the velocity calculation
	
	// declare all the registers
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
	__m256 g_reg;
	__m256 delta_t_reg;

	while (!terminate)
	{
		// mutex and condition variable guard to ensure that the threads don't do anything as long as the simulation isn't running
		{
			std::unique_lock<std::mutex> lock(run_mtx);
			run_cv.wait(lock, [&run] {return run; });
		}

		// pre-work section
		if (!terminate)
		{
			// run calculations that determine which registers this thread will work on
			if (last_size == 0 || last_size != bodies.size())
			{
				last_size = bodies.size();
				num_packed_elements = x_vec.size();
				num_elements_over = num_packed_elements * 8 - bodies.size();
				num_packed_elements_per_thread = std::ceil(static_cast<float>(num_packed_elements) / num_threads);
				num_elements_per_thread = std::ceil(static_cast<float>(last_size) / num_threads);
				packed_start_idx = thread_id * num_packed_elements_per_thread;
				body_start_idx = thread_id * num_elements_per_thread;
				bodies_per_thread = std::ceil(static_cast<float>(bodies.size()) / num_threads);
			}

			// registers for constants
			g_reg = _mm256_set_ps(ss.g, ss.g, ss.g, ss.g, ss.g, ss.g, ss.g, ss.g);
			delta_t_reg = _mm256_set_ps(ss.timestep, ss.timestep, ss.timestep, ss.timestep, ss.timestep, ss.timestep, ss.timestep, ss.timestep);

			// load current values into the register arrays
			for (size_t i = packed_start_idx; i < std::min(packed_start_idx + num_packed_elements_per_thread, num_packed_elements); i++)
			{
				// array of indices to ensure that we don't access invalid indices in case the bodies array's size is not divisible by 8
				const size_t indices[8] = { std::min(i * 8, last_size - 1), std::min(i * 8 + 1, last_size - 1), std::min(i * 8 + 2, last_size - 1), std::min(i * 8 + 3, last_size - 1), std::min(i * 8 + 4, last_size - 1), std::min(i * 8 + 5, last_size - 1), std::min(i * 8 + 6, last_size - 1), std::min(i * 8 + 7, last_size - 1) };
				x_vec[i] = _mm256_set_ps(bodies[indices[7]].x, bodies[indices[6]].x, bodies[indices[5]].x, bodies[indices[4]].x, bodies[indices[3]].x, bodies[indices[2]].x, bodies[indices[1]].x, bodies[indices[0]].x);
				y_vec[i] = _mm256_set_ps(bodies[indices[7]].y, bodies[indices[6]].y, bodies[indices[5]].y, bodies[indices[4]].y, bodies[indices[3]].y, bodies[indices[2]].y, bodies[indices[1]].y, bodies[indices[0]].y);
				#ifdef THREED
				z_vec[i] = _mm256_set_ps(bodies[indices[7]].z, bodies[indices[6]].z, bodies[indices[5]].z, bodies[indices[4]].z, bodies[indices[3]].z, bodies[indices[2]].z, bodies[indices[1]].z, bodies[indices[0]].z);
				#endif
				mass_vec[i] = _mm256_set_ps(bodies[indices[7]].m, bodies[indices[6]].m, bodies[indices[5]].m, bodies[indices[4]].m, bodies[indices[3]].m, bodies[indices[2]].m, bodies[indices[1]].m, bodies[indices[0]].m);
				r_vec[i] = _mm256_set_ps(bodies[indices[7]].r, bodies[indices[6]].r, bodies[indices[5]].r, bodies[indices[4]].r, bodies[indices[3]].r, bodies[indices[2]].r, bodies[indices[1]].r, bodies[indices[0]].r);
			}
		}

		// synchronize threads after loading
		if (!terminate)
		{
			compute_barrier1->arrive_and_wait();
		}
		else
		{
			break;
		}

		// start calculations
		if (!terminate)
		{
			for (size_t j = body_start_idx; j < std::min(body_start_idx + num_elements_per_thread, last_size); j++)
			{
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
				
				// calculate new position
				bodies[j].x += 0.5 * bodies[j].v_x * ss.timestep;
				bodies[j].y += 0.5 * bodies[j].v_y * ss.timestep;
			}
		}

		// send signal to main thread
		if (!terminate)
		{
			compute_barrier2->arrive_and_wait();
		}
		else
		{
			break;
		}

		// wait until rendering is done
		if (!terminate)
		{
			render_barrier->arrive_and_wait();
		}
		else
		{
			break;
		}
	}

}

#endif

#ifdef USE_OCTREE

/*
	functions that implement an octree for the problem
*/

void octree::build(std::vector<body>& start_bodies, sim_settings& ss)
{

	for (const body& it : start_bodies)
	{
		bodies.push_back(&it);
	}
	subdivide(ss);
	update_com();

}

void octree::calc_force(body& test_body, sim_settings& ss)
{
	
	if (is_leaf && bodies.size() == 0)
	{
		return;
	}

	// if this node is a leaf and the test body is in its domain, we run the calculation as normal
	// i.e. the O^2 algorithm
	if (is_leaf && in_bounds(&test_body))
	{
		for (const body*& it : bodies)
		{
			float d_x = test_body.x - it->x;
			float d_y = test_body.y - it->y;
			
		#ifndef THREED
			float dist = d_x * d_x + d_y * d_y;
		#else
			float d_z = test_body.z - it->z;
			float dist = d_x * d_x + d_y * d_y + d_z * d_z;
		#endif
			float dist_inv = rsqrt(dist);

			if (test_body.r * test_body.r + it->r * it->r >= dist)
				break;

			float force = -(it->m * ss.g) / dist;
			d_x *= dist_inv;
			test_body.v_x += d_x * force * ss.timestep;
			d_y *= dist_inv;
			test_body.v_y += d_y * force * ss.timestep;
		#ifdef THREED
			d_z *= dist_inv;
			test_body.v_z += d_z * force * ss.timestep;
		#endif 
		}
	}
	// otherwise we run the octree comparison and do a simplified calculation with the center of mass
	else
	{
		float d_x = test_body.x - com_x;
		float d_y = test_body.y - com_y;

	#ifndef THREED
		float dist = d_x * d_x + d_y * d_y;
	#else
		float d_z = test_body.z - com_z;
		float dist = d_x * d_x + d_y * d_y + d_z * d_z;
	#endif

		float dist_inv = rsqrt(dist);

		// calculate force either with the current com or all child coms
		if (dist_inv * size < ss.octree_tolerance)
		{
			// early exit if the body is too near the com to prevent inf and nan values
			if (test_body.r * test_body.r >= dist) return;
			float force = -(m * ss.g) / dist;
			d_x *= dist_inv;
			test_body.v_x += d_x * force * ss.timestep;
			d_y *= dist_inv;
			test_body.v_y += d_y * force * ss.timestep;
	#ifdef THREED
			d_z *= dist_inv;
			test_body.v_z += d_z * force * ss.timestep;
	#endif
		}
		else if (!is_leaf)
		{
			for (octree*& child : children)
			{
				child->calc_force(test_body, ss);
			}
		}
	}

}

void octree::subdivide(sim_settings& ss)
{

	// if there is enough space here, then this is a leaf node in the octree
	if (bodies.size() <= max_occupancy)
	{
		is_leaf = true;
		return;
	}
	// otherwise, distribute bodies to child nodes
	else
	{
	#ifndef THREED
		children[0] = new octree(c_x - size / 4, c_y + size / 4, size / 2, ss.octree_max_node_size);
		children[1] = new octree(c_x + size / 4, c_y + size / 4, size / 2, ss.octree_max_node_size);
		children[2] = new octree(c_x - size / 4, c_y - size / 4, size / 2, ss.octree_max_node_size);
		children[3] = new octree(c_x + size / 4, c_y - size / 4, size / 2, ss.octree_max_node_size);
	#else
		children[0] = new octree(c_x - size / 2, c_y + size / 2, c_z - size / 2, size / 2, ss.octree_max_node_size);
		children[1] = new octree(c_x + size / 2, c_y + size / 2, c_z - size / 2, size / 2, ss.octree_max_node_size);
		children[2] = new octree(c_x - size / 2, c_y - size / 2, c_z - size / 2, size / 2, ss.octree_max_node_size);
		children[3] = new octree(c_x + size / 2, c_y - size / 2, c_z - size / 2, size / 2, ss.octree_max_node_size);
		children[4] = new octree(c_x - size / 2, c_y + size / 2, c_z + size / 2, size / 2, ss.octree_max_node_size);
		children[5] = new octree(c_x + size / 2, c_y + size / 2, c_z + size / 2, size / 2, ss.octree_max_node_size);
		children[6] = new octree(c_x - size / 2, c_y - size / 2, c_z + size / 2, size / 2, ss.octree_max_node_size);
		children[7] = new octree(c_x + size / 2, c_y - size / 2, c_z + size / 2, size / 2, ss.octree_max_node_size);
	#endif
		for (const body*& it : bodies)
		{
			for (octree*& child : children)
			{
				if (child->in_bounds(it))
				{
					child->bodies.push_back(it);
					break;
				}
			}
		}
		bodies.clear();
		for (octree*& child : children)
		{
			child->subdivide(ss);
		}
		is_leaf = false;
	}

}

bool octree::in_bounds(const body* test_body) const
{

	// the 1e-3 here is a fudge factor to make sure that no bodies slip the net
	float box = size / 2;
	if (test_body->x >= c_x + box + 1e-3 || test_body->x < c_x - box - 1e-3) return false;
	if (test_body->y >= c_y + box + 1e-3 || test_body->y < c_y - box - 1e-3) return false;
#ifdef THREED
	if (test_body->x > c_x + box + 1e-3 || test_body->x < c_x - box - 1e-3) return false;
#endif
	return true;

}

void octree::update_com()
{

	com_x = 0;
	com_y = 0;
	m = 0;
	if (is_leaf)
	{
		if (bodies.size() == 0)
		{
			com_x = c_x;
			com_y = c_y;
			return;
		}
		for (const body* it : bodies)
		{
			com_x += it->x * it->m;
			com_y += it->y * it->m;
			m += it->m;
		}
	}
	else
	{
		for (octree*& it : children)
		{
			it->update_com();
			com_x += it->com_x * it->m;
			com_y += it->com_y * it->m;
			m += it->m;
		}
	}
	if (m != 0)
	{
		com_x /= m;
		com_y /= m;
	}

}

#endif