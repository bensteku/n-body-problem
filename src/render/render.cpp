#include "render.hpp"

std::vector<sf::CircleShape> init_shapes(const std::vector<body>& bodies)
{
	std::vector<sf::CircleShape> shapes;
	shapes.resize(bodies.size());
	
	for (size_t i = 0; i < bodies.size(); i++)
	{
		shapes[i].setRadius(bodies[i].m / settings::mass_radius_factor);
		// divide by 100 here is safe because at init time the scale will always be 100
		float x_pos = settings::window_width_h + settings::window_width_h * bodies[i].x / 100;
		float y_pos = settings::window_height_h - settings::aspect_ratio * settings::window_height_h * bodies[i].y / 100;
		shapes[i].setPosition(x_pos, y_pos);
	}
	
	return shapes;
}

// base Scene class

Scene::Scene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref) :
	m_window_ref(window_ref), m_bodies_ref(bodies_ref), m_shapes_ref(shapes_ref), m_is_ref(is_ref), m_ss_ref(ss_ref)
{

	if (!m_font.loadFromFile("./data/fonts/routed-gothic.ttf"))
		exit(-1);
	m_upper_left_text.setFont(m_font);
	m_upper_left_text.setCharacterSize(12);
	m_upper_left_text.setFillColor(sf::Color::Yellow);
	m_clock.restart();

}

void Scene::update_shapes()
{
	
	for (size_t i = 0; i < m_bodies_ref.size(); i++)
	{
		const float x_pos = settings::window_width_h + settings::window_width_h * (m_bodies_ref[i].x + m_is_ref.center_x) / (100 * m_is_ref.zoom);
		const float y_pos = settings::window_height_h - settings::aspect_ratio * settings::window_height_h * (m_bodies_ref[i].y + m_is_ref.center_y) / (100 * m_is_ref.zoom);
		m_shapes_ref[i].setPosition(x_pos, y_pos);
		const float radius = m_bodies_ref[i].m / (settings::mass_radius_factor * m_is_ref.zoom);
		m_shapes_ref[i].setOrigin(radius, radius);
		m_shapes_ref[i].setRadius(radius);
		m_window_ref.draw(m_shapes_ref[i]);
	}

}

void Scene::render_fps_info()
{
	if (m_is_ref.f_toggle)
	{
		if (m_frame_counter % settings::fps_smoother == 0)
		{
			float frame_time = m_clock.getElapsedTime().asSeconds();
			frame_time = (float)m_frame_counter / frame_time;
			m_upper_left_text.setString(m_fps_str + std::to_string(int(frame_time)) + "\n" +
				m_processing_type + "\n" +
				m_gravity_str + std::to_string(m_ss_ref.g) + "\n" +
				m_timestep_str + std::to_string(m_ss_ref.timestep) + "\n" +
				m_bodies_str + std::to_string(m_ss_ref.n)
				#ifdef USE_OCTREE
				+ "\n" + m_octree_str1 + std::to_string(m_ss_ref.octree_max_node_size)
				+ "\n" + m_octree_str2 + std::to_string(m_ss_ref.octree_tolerance)
				#endif
			);
			m_clock.restart();
			m_frame_counter = 1;
		}
		else
		{
			m_frame_counter += 1;
		}
		m_window_ref.draw(m_upper_left_text);
	}
}

// Scene class for the init screen

SetupScene::SetupScene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref)
{

	m_setup_text.setFont(m_font);
	m_setup_text.setCharacterSize(14);
	m_setup_text.setFillColor(sf::Color::Cyan);
	m_setup_text.setString(m_settings_str);
	m_setup_text.setOrigin(m_setup_text.getGlobalBounds().getSize() / 2.f + m_setup_text.getLocalBounds().getPosition());
	m_setup_text.setPosition(settings::window_width_h, settings::window_height_h);

}

State SetupScene::process_inputs()
{
	
	process_inputs_setup(m_window_ref, m_bodies_ref, m_shapes_ref, m_is_ref, m_ss_ref);
	return setup;

}

void SetupScene::render(State state_before)
{

	m_window_ref.draw(m_setup_text);

}

// Scene class for the init screen with a circle

SetupSceneCircle::SetupSceneCircle(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref)
{

	m_setup_circle_text.setFont(m_font);
	m_setup_circle_text.setCharacterSize(14);
	m_setup_circle_text.setFillColor(sf::Color::Cyan);
	m_setup_circle_text.setString(m_settings_circle_str);
	m_setup_circle_text.setOrigin(m_setup_circle_text.getGlobalBounds().getSize() / 2.f + m_setup_circle_text.getLocalBounds().getPosition());
	m_setup_circle_text.setPosition(settings::window_width_h, 55);

	m_previous_size = bodies_ref.size();

}

State SetupSceneCircle::process_inputs()
{

	process_inputs_setup_circle(m_window_ref, m_is_ref, m_ss_ref);
	return setup_circle;

}

void SetupSceneCircle::render(State state_before)
{
	
	// if the desired amount of bodies changed, resize all relevant vectors
	if (m_previous_size != m_ss_ref.n)
	{
		m_previous_size = m_ss_ref.n;
		m_bodies_ref.resize(m_ss_ref.n);
		m_shapes_ref.resize(m_ss_ref.n);
		#ifdef USE_SIMD
		const size_t num_packed_elements = std::ceil(static_cast<float>(m_ss_ref.n) / 8.0);
		for (auto& it : m_registers)
			it.resize(num_packed_elements);
		#endif
	}
	// set up bodies in the circle as determined by user inputs
	init_bodies_circle(m_bodies_ref, m_ss_ref);
	
	update_shapes();

	m_window_ref.draw(m_setup_circle_text);
	render_fps_info();

}

// Scene class for uniform random initialization

SetupSceneUniform::SetupSceneUniform(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref)
{

	m_setup_uniform_text.setFont(m_font);
	m_setup_uniform_text.setCharacterSize(14);
	m_setup_uniform_text.setFillColor(sf::Color::Cyan);
	m_setup_uniform_text.setString(m_settings_uniform_str);
	m_setup_uniform_text.setOrigin(m_setup_uniform_text.getGlobalBounds().getSize() / 2.f + m_setup_uniform_text.getLocalBounds().getPosition());
	m_setup_uniform_text.setPosition(settings::window_width_h, 55);

	m_previous_size = bodies_ref.size();

}

State SetupSceneUniform::process_inputs()
{

	process_inputs_setup_uniform(m_window_ref, m_is_ref, m_ss_ref);
	return setup_uniform;

}

void SetupSceneUniform::render(State state_before)
{

	// if the desired amount of bodies changed, resize all relevant vectors
	if (m_previous_size != m_ss_ref.n)
	{
		m_previous_size = m_ss_ref.n;
		m_bodies_ref.resize(m_ss_ref.n);
		m_shapes_ref.resize(m_ss_ref.n);
		#ifdef USE_SIMD
		const size_t num_packed_elements = std::ceil(static_cast<float>(m_ss_ref.n) / 8.0);
		for (auto& it : m_registers)
			it.resize(num_packed_elements);
		#endif
	}
	// set up bodies uniformly random
	init_bodies_uniform(m_bodies_ref, m_ss_ref);

	update_shapes();

	m_window_ref.draw(m_setup_uniform_text);
	render_fps_info();

}

// Scene class for the normal distributed intialization

SetupSceneNormal::SetupSceneNormal(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref)
{

	m_setup_normal_text.setFont(m_font);
	m_setup_normal_text.setCharacterSize(14);
	m_setup_normal_text.setFillColor(sf::Color::Cyan);
	m_setup_normal_text.setString(m_settings_normal_str);
	m_setup_normal_text.setOrigin(m_setup_normal_text.getGlobalBounds().getSize() / 2.f + m_setup_normal_text.getLocalBounds().getPosition());
	m_setup_normal_text.setPosition(settings::window_width_h, 55);

	m_previous_size = bodies_ref.size();

}

State SetupSceneNormal::process_inputs()
{

	// works for just as well for normal distribution
	process_inputs_setup_uniform(m_window_ref, m_is_ref, m_ss_ref);
	return setup_normal;

}

void SetupSceneNormal::render(State state_before)
{

	// if the desired amount of bodies changed, resize all relevant vectors
	if (m_previous_size != m_ss_ref.n)
	{
		m_previous_size = m_ss_ref.n;
		m_bodies_ref.resize(m_ss_ref.n);
		m_shapes_ref.resize(m_ss_ref.n);
		#ifdef USE_SIMD
		const size_t num_packed_elements = std::ceil(static_cast<float>(m_ss_ref.n) / 8.0);
		for (auto& it : m_registers)
			it.resize(num_packed_elements);
		#endif
	}
	// set up bodies uniformly random
	init_bodies_normal(m_bodies_ref, m_ss_ref);

	update_shapes();

	m_window_ref.draw(m_setup_normal_text);
	render_fps_info();

}

// Scene class for custom initialization

SetupSceneCustom::SetupSceneCustom(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref)
{

	m_setup_custom_text.setFont(m_font);
	m_setup_custom_text.setCharacterSize(14);
	m_setup_custom_text.setFillColor(sf::Color::Cyan);
	m_setup_custom_text.setString(m_settings_custom_str);
	m_setup_custom_text.setOrigin(m_setup_custom_text.getGlobalBounds().getSize() / 2.f + m_setup_custom_text.getLocalBounds().getPosition());
	m_setup_custom_text.setPosition(settings::window_width_h, 55);

	m_previous_size = bodies_ref.size();

}

State SetupSceneCustom::process_inputs()
{

	process_inputs_setup_custom(m_window_ref, m_bodies_ref, m_shapes_ref, m_is_ref, m_ss_ref);
	return setup_custom;

}

void SetupSceneCustom::render(State state_before)
{

	// if the desired amount of bodies changed, resize all relevant vectors
	if (m_previous_size != m_ss_ref.n)
	{
		m_previous_size = m_ss_ref.n;
		m_bodies_ref.resize(m_ss_ref.n);
		m_shapes_ref.resize(m_ss_ref.n);
		#ifdef USE_SIMD
		const size_t num_packed_elements = std::ceil(static_cast<float>(m_ss_ref.n) / 8.0);
		for (auto& it : m_registers)
			it.resize(num_packed_elements);
		#endif
	}

	update_shapes();

	m_window_ref.draw(m_setup_custom_text);
	render_fps_info();

}

// Scene class for the actual simulation

SimScene::SimScene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref)
{
	
	#ifdef USE_THREADS
	size_t num_threads = std::thread::hardware_concurrency() / 2;

	// uses unique pointer to avoid having to change the constructor's interface or main.cpp in case the USE_THREADS define is true
	m_compute_barrier1 = std::make_unique<std::barrier<>>(num_threads);
	m_compute_barrier2 = std::make_unique<std::barrier<>>(num_threads + 1);
	m_render_barrier = std::make_unique<std::barrier<>>(num_threads + 1);
	#ifdef USE_OCTREE
	m_shared_octree = std::make_unique<octree>();
	#endif

	m_threads.resize(num_threads);
	for (size_t i = 0; i < num_threads; i++)
	{
	#ifdef USE_SIMD
		m_threads[i] = std::thread(process_bodies_simd_mt,
								   std::ref(m_compute_barrier1), std::ref(m_compute_barrier2), std::ref(m_render_barrier),
								   std::ref(m_terminate_signal), std::ref(m_run_signal),
								   std::ref(m_run_cv), std::ref(m_run_mutex),
								   i, num_threads,
								   std::ref(m_bodies_ref), std::ref(m_ss_ref),
								   std::ref(m_registers[0]), std::ref(m_registers[1]), std::ref(m_registers[2]), std::ref(m_registers[3]));
	#else
		#ifdef USE_OCTREE
			m_threads[i] = std::thread(octree_calc_force_mt,
									   std::ref(m_compute_barrier1), std::ref(m_compute_barrier2), std::ref(m_render_barrier), std::ref(m_terminate_signal), std::ref(m_run_signal),
									   std::ref(m_run_cv), std::ref(m_run_mutex),
									   i, num_threads,
									   std::ref(m_bodies_ref), std::ref(m_shared_octree), std::ref(m_ss_ref));
		#else
			m_threads[i] = std::thread(process_bodies_mt,
									   std::ref(m_compute_barrier1), std::ref(m_compute_barrier2), std::ref(m_render_barrier), std::ref(m_terminate_signal), std::ref(m_run_signal),
									   std::ref(m_run_cv), std::ref(m_run_mutex),
									   i, num_threads,
									   std::ref(m_bodies_ref), std::ref(m_ss_ref));
		#endif
	#endif
	}
	#endif

}

#ifdef USE_THREADS

SimScene::~SimScene()
{

	// set terminate signal and wake up any sleeping threads so that they can end
	m_terminate_signal = true;
	m_compute_barrier2->arrive_and_drop();
	m_render_barrier->arrive_and_drop();
	for (std::thread& thread : m_threads)
	{
		thread.join();
	}

}

#endif

State SimScene::process_inputs()
{

	process_inputs_sim(m_window_ref, m_is_ref, m_ss_ref);
	#ifdef USE_THREADS
	// suspend the work in the thread pool when use exits sim
	if (m_is_ref.program_state != sim)
	{
		m_run_signal = false;
	}
	#endif
	return sim;

}

void SimScene::render(State state_before)
{

	#ifdef USE_THREADS
	if (state_before != sim)
	{
		#ifdef USE_OCTREE
		m_shared_octree->build(m_bodies_ref, m_ss_ref);
		#endif
		{
			std::lock_guard<std::mutex> lock(m_run_mutex);
			m_run_signal = true;
		}
		m_run_cv.notify_all();
	}
	#endif

	#ifdef USE_CUDA
	// if CUDA is enabled and the user changed the number of bodies or we're going into this method for the first time since setup
	// we have to reallocate memory for our GPU pointers
	if (state_before != sim)
	{
		cudaFree(m_d_bodies);
		cudaFree(m_d_interactions_x);
		cudaFree(m_d_interactions_y);
		const size_t bytes = sizeof(body) * m_bodies_ref.size();
		cudaMalloc(&m_d_bodies, bytes);
		const size_t bytes2 = sizeof(float) * m_bodies_ref.size() * m_bodies_ref.size();
		cudaMalloc(&m_d_interactions_x, bytes2);
		cudaMalloc(&m_d_interactions_y, bytes2);
		cudaMemcpy(m_d_bodies, m_bodies_ref.data(), bytes, cudaMemcpyHostToDevice);

		#ifdef USE_OCTREE
		cudaFree(m_d_bodies_bu);
		cudaFree(m_d_help);
		cudaFree(m_d_indices);
		cudaFree(m_d_nodes);
		cudaMalloc(&m_d_bodies_bu, bytes);
		const size_t bytes3 = sizeof(sort_help) * m_bodies_ref.size();
		const size_t bytes4 = sizeof(size_t) * m_bodies_ref.size();
		const size_t bytes5 = sizeof(cu_octree_node) * ((std::pow(4, m_ss_ref.octree_max_depth + 1) - 1) / (m_ss_ref.octree_max_depth - 1));
		cudaMalloc(&m_d_help, bytes3);
		cudaMalloc(&m_d_indices, bytes4);
		cudaMalloc(&m_d_nodes, bytes5);
		#endif
	}
	#ifndef USE_OCTREE
	process_bodies_cuda(m_bodies_ref, m_d_bodies, m_d_interactions_x, m_d_interactions_y, m_ss_ref);
	#else
	process_bodies_octree_cuda(m_bodies_ref, m_d_bodies, m_d_bodies_bu, m_d_nodes, m_d_help, m_d_indices, m_ss_ref);
	#endif
	#elif defined(USE_SIMD)
	#ifdef USE_THREADS
	m_compute_barrier2->arrive_and_wait();
	#else
	process_bodies_simd(m_bodies_ref, m_ss_ref, m_registers[0], m_registers[1], m_registers[2], m_registers[3]);
	#endif
	#else
	#ifdef USE_THREADS
	m_compute_barrier2->arrive_and_wait();
	#else
	process_bodies(m_bodies_ref, m_ss_ref);
	#endif
	#endif
	// update the shapes afterwards
	update_shapes();
	// optionally display FPS counter
	render_fps_info();
	#ifdef USE_THREADS
	#ifdef USE_OCTREE
	m_shared_octree->build(m_bodies_ref, m_ss_ref);
	#endif
	m_render_barrier->arrive_and_wait();
	#endif

}