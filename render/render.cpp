#include "render.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "../body/body.cuh"
#endif
#include <ranges>

#include "../misc/settings.hpp"

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
	
	const std::vector<body>& const_bodies_ref = const_cast<const std::vector<body>&>(m_bodies_ref);
	for (size_t i = 0; i < m_bodies_ref.size(); i++)
	{
		const float x_pos = settings::window_width_h + settings::window_width_h * (const_bodies_ref[i].x + m_is_ref.center_x) / (100 * m_is_ref.zoom);
		const float y_pos = settings::window_height_h - settings::aspect_ratio * settings::window_height_h * (const_bodies_ref[i].y + m_is_ref.center_y) / (100 * m_is_ref.zoom);
		m_shapes_ref[i].setPosition(x_pos, y_pos);
		const float radius = const_bodies_ref[i].m / (settings::mass_radius_factor * m_is_ref.zoom);
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
				m_bodies_str + std::to_string(m_ss_ref.n));
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

void SetupScene::process_inputs()
{
	
	process_inputs_setup(m_window_ref, m_bodies_ref, m_shapes_ref, m_is_ref, m_ss_ref);

}

void SetupScene::render()
{

	m_window_ref.draw(m_setup_text);

}

// Scene class for the init screen with a circle

SetupSceneCircle::SetupSceneCircle(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref), m_registers_ref(registers)
{

	m_setup_circle_text.setFont(m_font);
	m_setup_circle_text.setCharacterSize(14);
	m_setup_circle_text.setFillColor(sf::Color::Cyan);
	m_setup_circle_text.setString(m_settings_circle_str);
	m_setup_circle_text.setOrigin(m_setup_circle_text.getGlobalBounds().getSize() / 2.f + m_setup_circle_text.getLocalBounds().getPosition());
	m_setup_circle_text.setPosition(settings::window_width_h, 55);

	m_previous_size = bodies_ref.size();

}

void SetupSceneCircle::process_inputs()
{

	process_inputs_setup_circle(m_window_ref, m_is_ref, m_ss_ref);

}

void SetupSceneCircle::render()
{
	
	// if the desired amount of bodies changed, resize all relevant vectors
	if (m_previous_size != m_ss_ref.n)
	{
		m_previous_size = m_ss_ref.n;
		m_bodies_ref.resize(m_ss_ref.n);
		m_shapes_ref.resize(m_ss_ref.n);
		const size_t num_packed_elements = 1 + m_ss_ref.n / 8;
		for (auto& it : m_registers_ref)
			it.resize(num_packed_elements);
	}
	// set up bodies in the circle as determined by user inputs
	init_bodies_circle(m_bodies_ref, m_ss_ref.min_mass, m_ss_ref.max_mass, m_ss_ref.circle_radius, m_ss_ref.circle_deviation);
	
	update_shapes();

	m_window_ref.draw(m_setup_circle_text);
	render_fps_info();

}

// Scene class for uniform random initialization

SetupSceneUniform::SetupSceneUniform(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref), m_registers_ref(registers)
{

	m_setup_uniform_text.setFont(m_font);
	m_setup_uniform_text.setCharacterSize(14);
	m_setup_uniform_text.setFillColor(sf::Color::Cyan);
	m_setup_uniform_text.setString(m_settings_uniform_str);
	m_setup_uniform_text.setOrigin(m_setup_uniform_text.getGlobalBounds().getSize() / 2.f + m_setup_uniform_text.getLocalBounds().getPosition());
	m_setup_uniform_text.setPosition(settings::window_width_h, 55);

	m_previous_size = bodies_ref.size();

}

void SetupSceneUniform::process_inputs()
{

	process_inputs_setup_uniform(m_window_ref, m_is_ref, m_ss_ref);

}

void SetupSceneUniform::render()
{

	// if the desired amount of bodies changed, resize all relevant vectors
	if (m_previous_size != m_ss_ref.n)
	{
		m_previous_size = m_ss_ref.n;
		m_bodies_ref.resize(m_ss_ref.n);
		m_shapes_ref.resize(m_ss_ref.n);
		const size_t num_packed_elements = 1 + m_ss_ref.n / 8;
		for (auto& it : m_registers_ref)
			it.resize(num_packed_elements);
	}
	// set up bodies uniformly random
	init_bodies_uniform(m_bodies_ref, m_ss_ref.min_mass, m_ss_ref.max_mass, m_ss_ref.x_range, m_ss_ref.y_range);

	update_shapes();

	m_window_ref.draw(m_setup_uniform_text);
	render_fps_info();

}

// Scene class for the normal distributed intialization

SetupSceneNormal::SetupSceneNormal(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref), m_registers_ref(registers)
{

	m_setup_normal_text.setFont(m_font);
	m_setup_normal_text.setCharacterSize(14);
	m_setup_normal_text.setFillColor(sf::Color::Cyan);
	m_setup_normal_text.setString(m_settings_normal_str);
	m_setup_normal_text.setOrigin(m_setup_normal_text.getGlobalBounds().getSize() / 2.f + m_setup_normal_text.getLocalBounds().getPosition());
	m_setup_normal_text.setPosition(settings::window_width_h, 55);

	m_previous_size = bodies_ref.size();

}

void SetupSceneNormal::process_inputs()
{

	// works for just as well for normal distribution
	process_inputs_setup_uniform(m_window_ref, m_is_ref, m_ss_ref);

}

void SetupSceneNormal::render()
{

	// if the desired amount of bodies changed, resize all relevant vectors
	if (m_previous_size != m_ss_ref.n)
	{
		m_previous_size = m_ss_ref.n;
		m_bodies_ref.resize(m_ss_ref.n);
		m_shapes_ref.resize(m_ss_ref.n);
		const size_t num_packed_elements = 1 + m_ss_ref.n / 8;
		for (auto& it : m_registers_ref)
			it.resize(num_packed_elements);
	}
	// set up bodies uniformly random
	init_bodies_normal(m_bodies_ref, m_ss_ref.min_mass, m_ss_ref.max_mass, 0, 0, m_ss_ref.x_range, m_ss_ref.y_range);

	update_shapes();

	m_window_ref.draw(m_setup_normal_text);
	render_fps_info();

}

// Scene class for custom initialization

SetupSceneCustom::SetupSceneCustom(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref), m_registers_ref(registers)
{

	m_setup_custom_text.setFont(m_font);
	m_setup_custom_text.setCharacterSize(14);
	m_setup_custom_text.setFillColor(sf::Color::Cyan);
	m_setup_custom_text.setString(m_settings_custom_str);
	m_setup_custom_text.setOrigin(m_setup_custom_text.getGlobalBounds().getSize() / 2.f + m_setup_custom_text.getLocalBounds().getPosition());
	m_setup_custom_text.setPosition(settings::window_width_h, 55);

	m_previous_size = bodies_ref.size();

}

void SetupSceneCustom::process_inputs()
{

	process_inputs_setup_custom(m_window_ref, m_bodies_ref, m_shapes_ref, m_is_ref, m_ss_ref);

}

void SetupSceneCustom::render()
{

	// if the desired amount of bodies changed, resize all relevant vectors
	if (m_previous_size != m_ss_ref.n)
	{
		m_previous_size = m_ss_ref.n;
		m_bodies_ref.resize(m_ss_ref.n);
		m_shapes_ref.resize(m_ss_ref.n);
		const size_t num_packed_elements = 1 + m_ss_ref.n / 8;
		for (auto& it : m_registers_ref)
			it.resize(num_packed_elements);
	}

	update_shapes();

	m_window_ref.draw(m_setup_custom_text);
	render_fps_info();

}

// Scene class for the actual simulation

SimScene::SimScene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers) :
	Scene(window_ref, bodies_ref, shapes_ref, is_ref, ss_ref), m_registers_ref(registers)
{

#	ifdef USE_CUDA
	// if CUDA is enabled, we allocate memory for our pointers on the GPU
	m_last_allocation = m_bodies_ref.size();
	const size_t bytes = sizeof(float) * m_last_allocation;
	cudaMalloc(&m_d_x, bytes);
	cudaMalloc(&m_d_y, bytes);
	cudaMalloc(&m_d_mass, bytes);
	cudaMalloc(&m_d_radius, bytes);
	cudaMalloc(&m_d_v_x, bytes);
	cudaMalloc(&m_d_v_y, bytes);
	m_h_x.resize(m_last_allocation);
	m_h_y.resize(m_last_allocation);
	m_h_mass.resize(m_last_allocation);
	m_h_radius.resize(m_last_allocation);
	m_h_v_x.resize(m_last_allocation);
	m_h_v_y.resize(m_last_allocation);
#	endif

}

void SimScene::process_inputs()
{

	process_inputs_sim(m_window_ref, m_is_ref, m_ss_ref);

}

void SimScene::render()
{

#	ifdef USE_CUDA
	// if CUDA is enabled and the user changed the number of bodies,
	// we have to reallocate memory for our GPU pointers
	if (m_bodies_ref.size() != m_last_allocation)
	{
		cudaFree(m_d_x);
		cudaFree(m_d_y);
		cudaFree(m_d_mass);
		cudaFree(m_d_radius);
		cudaFree(m_d_v_x);
		cudaFree(m_d_v_y);
		m_last_allocation = m_bodies_ref.size();
		const size_t bytes = sizeof(float) * m_last_allocation;
		cudaMalloc(&m_d_x, bytes);
		cudaMalloc(&m_d_y, bytes);
		cudaMalloc(&m_d_mass, bytes);
		cudaMalloc(&m_d_radius, bytes);
		cudaMalloc(&m_d_v_x, bytes);
		cudaMalloc(&m_d_v_y, bytes);
		// the same for our CPU-side holding space
		m_h_x.resize(m_last_allocation);
		m_h_y.resize(m_last_allocation);
		m_h_mass.resize(m_last_allocation);
		m_h_radius.resize(m_last_allocation);
		m_h_v_x.resize(m_last_allocation);
		m_h_v_y.resize(m_last_allocation);
	}

	process_bodies_cuda(m_bodies_ref,
						m_h_x,
						m_h_y,
						m_h_mass,
						m_h_radius,
						m_h_v_x,
						m_h_v_y,
						m_d_x,
						m_d_y,
						m_d_mass,
						m_d_radius,
						m_d_v_x,
						m_d_v_y,
						m_ss_ref);
#	elif defined(USE_SIMD)
	process_bodies_simd(m_bodies_ref, m_ss_ref, m_registers_ref[0], m_registers_ref[1], m_registers_ref[2], m_registers_ref[3]);
#	else
	process_bodies(m_bodies_ref, m_ss_ref);
#	endif
	// update the shapes afterwards
	update_shapes();
	// optionally display FPS counter
	render_fps_info();

}