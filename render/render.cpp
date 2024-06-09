#include "render.hpp"

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

Renderer::Renderer(const std::string file_path)
{
	if (!m_font.loadFromFile(file_path))
		exit(-1);
	m_fps_text.setFont(m_font);
	m_fps_text.setCharacterSize(12);
	m_fps_text.setFillColor(sf::Color::Yellow);
	m_clock.restart();
}

void Renderer::render(sf::RenderWindow& window, const std::vector<sf::CircleShape>& shapes, input_settings& is, processing_args& pa)
{
	for (const sf::CircleShape& it : shapes)
	{
		window.draw(it);
	}
	if (is.f_pressed)
	{
		switch (settings::pt)
		{
			case SISD:
				m_processing_type = "SISD";
				break;
			case SIMD:
				m_processing_type = "SIMD";
				break;
			case CUDA:
				m_processing_type = "CUDA";
				break;
		}
		if (m_frame_counter % settings::fps_smoother == 0)
		{
			float frame_time = m_clock.getElapsedTime().asSeconds();
			frame_time = (float)m_frame_counter / frame_time;
			m_fps_text.setString(m_fps_string + std::to_string(int(frame_time)) + "\n" +
								m_processing_type + "\n" +
								m_gravity + std::to_string(pa.g) + "\n" +
								m_timestep + std::to_string(pa.timestep) + "\n" +
								m_bodies + std::to_string(pa.n));
			m_clock.restart();
			m_frame_counter = 1;
		}
		else
		{
			m_frame_counter += 1;
		}
		window.draw(m_fps_text);
	}
}

void Renderer::update_shapes(sf::RenderWindow& window, std::vector<sf::CircleShape>& shapes, const std::vector<body>& bodies, float x_offset, float y_offset, float zoom)
{
	for (size_t i = 0; i < bodies.size(); i++)
	{
		shapes[i].setRadius(bodies[i].m / (settings::mass_radius_factor * zoom));
		float x_pos = settings::window_width_h + settings::window_width_h * (bodies[i].x + x_offset) / (100 * zoom);
		float y_pos = settings::window_height_h - settings::aspect_ratio * settings::window_height_h * (bodies[i].y + y_offset) / (100 * zoom);
		shapes[i].setPosition(x_pos, y_pos);
	}
}
