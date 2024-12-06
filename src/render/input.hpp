#pragma once

#include <SFML/Graphics.hpp>

#include "../body/body.hpp"
#include "../misc/settings.hpp"
#include <iostream>

enum State
{

	setup,
	setup_circle,
	setup_uniform,
	setup_normal,
	setup_custom,
	sim

};

struct input_settings
{

	// state of the program's UI
	State program_state = setup;
	// values for worldspace to screenspace conversion
	float zoom = 1.0;
	float center_x = 0.0;
	float center_y = 0.0;
	// buffer for the dragging functionality
	int mouse_x = 0;
	int mouse_y = 0;
	bool mouse_clicked = false;
	// modifier key for zoom and dragging
	bool ctrl_pressed = false;
	// modifier for circle init
	bool d_pressed = false;
	// modifiers for all inits
	bool m_pressed = false;
	bool n_pressed = false;
	bool b_pressed = false;
	bool c_pressed = false;
	// toggle for showing fps
	bool f_toggle = false;
	// storage for new bodies in custom mode
	size_t new_body_index = 0;

};

inline void process_inputs_sim(sf::RenderWindow& window, input_settings& is, sim_settings& ss)
{

	sf::Event event;
	while (window.pollEvent(event))
	{
		switch (event.type)
		{
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::MouseWheelMoved:
				is.zoom *= 1 - ((0.05 + (int)is.ctrl_pressed * settings::zoom_modifier) * event.mouseWheel.delta);
				break;
			case sf::Event::MouseButtonPressed:
				if (event.mouseButton.button == sf::Mouse::Left)
				{
					is.mouse_x = event.mouseButton.x;
					is.mouse_y = event.mouseButton.y;
					is.mouse_clicked = true;
				}
				break;
			case sf::Event::MouseMoved:
				if (is.mouse_clicked)
				{
					is.center_x += (is.zoom + settings::drag_modifier * (int)is.ctrl_pressed) * (event.mouseMove.x - is.mouse_x) / (100);
					is.center_y -= (is.zoom + settings::drag_modifier * (int)is.ctrl_pressed) * (event.mouseMove.y - is.mouse_y) / (100);
					is.mouse_x = event.mouseMove.x;
					is.mouse_y = event.mouseMove.y;
				}
				break;
			case  sf::Event::MouseButtonReleased:
				if (event.mouseButton.button == sf::Mouse::Left)
				{
					// TODO: make this a bit more refined
					is.center_x += (is.zoom + settings::drag_modifier * (int)is.ctrl_pressed) * (event.mouseButton.x - is.mouse_x) / (100);
					is.center_y -= (is.zoom + settings::drag_modifier * (int)is.ctrl_pressed) * (event.mouseButton.y - is.mouse_y) / (100);
					is.mouse_clicked = false;
				}
				break;
			case sf::Event::KeyPressed:
				switch (event.key.scancode)
				{
				case sf::Keyboard::Scan::R:
					is.zoom = 1.0;
					is.center_x = 0;
					is.center_y = 0;
					break;
				case sf::Keyboard::Scan::LControl:
					is.ctrl_pressed = true;
					break;
				case sf::Keyboard::Scan::F:
					is.f_toggle = !is.f_toggle;
					break;
				case sf::Keyboard::Scan::NumpadPlus:
					if (is.ctrl_pressed)
						ss.timestep *= 1.1;
					else
						ss.g *= 1.1;
					break;
				case sf::Keyboard::Scan::NumpadMinus:
					if (is.ctrl_pressed)
						ss.timestep *= 0.9;
					else
						ss.g *= 0.9;
					break;
				case sf::Keyboard::Scan::S:
					ss.g *= -1;
					break;
				case sf::Keyboard::Scan::T:
					is.program_state = setup;
					// resetting the settings struct to default
					// in case it was overwritten by user interaction
					ss.max_mass = 2000;
					ss.min_mass = 2000;
					ss.x_range = 25;
					ss.y_range = 25;
					break;
				default:
					break;
				}
				break;
			case sf::Event::KeyReleased:
				if (event.key.scancode == sf::Keyboard::Scan::LControl)
				{
					is.ctrl_pressed = false;
				}
				break;
		}
	}

}

inline void process_inputs_setup(sf::RenderWindow& window, std::vector<body>& bodies, std::vector<sf::CircleShape>& shapes, input_settings& is, sim_settings& ss)
{

	sf::Event event;
	while (window.pollEvent(event))
	{
		switch (event.type)
		{
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::KeyPressed:
				switch (event.key.scancode)
				{
					case sf::Keyboard::Scan::Num1:
					case sf::Keyboard::Scan::Numpad1:
						is.program_state = setup_circle;
						break;
					case sf::Keyboard::Scan::Num2:
					case sf::Keyboard::Scan::Numpad2:
						is.program_state = setup_uniform;
						break;
					case sf::Keyboard::Scan::Num3:
					case sf::Keyboard::Scan::Numpad3:
						is.program_state = setup_normal;
						break;
					case sf::Keyboard::Scan::Num4:
					case sf::Keyboard::Scan::Numpad4:
						is.program_state = setup_custom;
						bodies.resize(0);
						shapes.resize(0);
						ss.n = 0;
						break;
				}
		}
	}

}

inline void process_inputs_setup_circle(sf::RenderWindow& window, input_settings& is, sim_settings& ss)
{

	sf::Event event;
	while (window.pollEvent(event))
	{
		switch (event.type)
		{
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::MouseButtonPressed:
				if (event.mouseButton.button == sf::Mouse::Left)
					is.mouse_clicked = true;
				break;
			case sf::Event::MouseButtonReleased:
				if (event.mouseButton.button == sf::Mouse::Left)
					is.mouse_clicked = false;
				break;
			case sf::Event::MouseMoved:
				if (is.mouse_clicked)
				{
					float dx = 100 * is.zoom * (event.mouseMove.x / settings::window_width_h - 1) - is.center_x;
					float dy = (100 / settings::aspect_ratio) * is.zoom * (event.mouseMove.y / settings::window_height_h - 1) - is.center_y;
					ss.circle_radius = sqrt(dx * dx + dy * dy);
				}
				break;
			case sf::Event::MouseWheelMoved:
				if (is.d_pressed)
					ss.circle_deviation = std::max(0.0, ss.circle_deviation + 0.01 * event.mouseWheel.delta);
				else if (is.m_pressed)
					ss.max_mass = std::max((double)ss.min_mass, ss.max_mass * (1 + 0.01 * event.mouseWheel.delta));
				else if (is.n_pressed)
					ss.min_mass = std::max(0.0, std::min((double)ss.max_mass, ss.min_mass * (1 + 0.01 * event.mouseWheel.delta)));
				else if (is.b_pressed)
				{
					if (event.mouseWheel.delta >= 0)
						ss.n = std::max(ss.n + 1, (size_t)(ss.n * 1.1));
					else
						ss.n = std::max((size_t)0, std::min(ss.n - 1, (size_t)(ss.n * 0.9)));
				}
				else
					is.zoom *= 1 - ((0.05 + (int)is.ctrl_pressed * settings::zoom_modifier) * event.mouseWheel.delta);
				break;
			case sf::Event::KeyPressed:
				switch (event.key.scancode)
				{
					case sf::Keyboard::Scan::D:
						is.d_pressed = true;
						break;
					case sf::Keyboard::Scan::Enter:
					case sf::Keyboard::Scan::NumpadEnter:
						is.program_state = sim;
						break;
					case sf::Keyboard::Scan::LControl:
						is.ctrl_pressed = true;
						break;
					case sf::Keyboard::Scan::M:
						is.m_pressed = true;
						break;
					case sf::Keyboard::Scan::N:
						is.n_pressed = true;
						break;
					case sf::Keyboard::Scan::B:
						is.b_pressed = true;
						break;
				}
				break;
			case sf::Event::KeyReleased:
				switch (event.key.scancode)
				{
					case sf::Keyboard::Scan::D:
						is.d_pressed = false;
						break;
					case sf::Keyboard::Scan::LControl:
						is.ctrl_pressed = false;
						break;
					case sf::Keyboard::Scan::M:
						is.m_pressed = false;
						break;
					case sf::Keyboard::Scan::N:
						is.n_pressed = false;
						break;
					case sf::Keyboard::Scan::B:
						is.b_pressed = false;
						break;
				}
				break;
		}
	}

}

inline void process_inputs_setup_uniform(sf::RenderWindow& window, input_settings& is, sim_settings& ss)
{

	sf::Event event;
	while (window.pollEvent(event))
	{
		switch (event.type)
		{
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::MouseWheelMoved:
				if (is.d_pressed)
					ss.x_range = std::max(0.0, ss.x_range * (1 + 0.01 * event.mouseWheel.delta));
				else if (is.c_pressed)
					ss.y_range = std::max(0.0, ss.y_range * (1 + 0.01 * event.mouseWheel.delta));
				else if (is.m_pressed)
					ss.max_mass = std::max((double)ss.min_mass, ss.max_mass * (1 + 0.01 * event.mouseWheel.delta));
				else if (is.n_pressed)
					ss.min_mass = std::max(0.0, std::min((double)ss.max_mass, ss.min_mass * (1 + 0.01 * event.mouseWheel.delta)));
				else if (is.b_pressed)
				{
					if (event.mouseWheel.delta >= 0)
						ss.n = std::max(ss.n + 1, (size_t)(ss.n * 1.1));
					else
						ss.n = std::max((size_t)0, std::min(ss.n - 1, (size_t)(ss.n * 0.9)));
				}
				else
					is.zoom *= 1 - ((0.05 + (int)is.ctrl_pressed * settings::zoom_modifier) * event.mouseWheel.delta);
				break;
			case sf::Event::KeyPressed:
				switch (event.key.scancode)
				{
					case sf::Keyboard::Scan::D:
						is.d_pressed = true;
						break;
					case sf::Keyboard::Scan::Enter:
					case sf::Keyboard::Scan::NumpadEnter:
						is.program_state = sim;
						break;
					case sf::Keyboard::Scan::LControl:
						is.ctrl_pressed = true;
						break;
					case sf::Keyboard::Scan::M:
						is.m_pressed = true;
						break;
					case sf::Keyboard::Scan::N:
						is.n_pressed = true;
						break;
					case sf::Keyboard::Scan::B:
						is.b_pressed = true;
						break;
					case sf::Keyboard::Scan::C:
						is.c_pressed = true;
						break;
				}
				break;
			case sf::Event::KeyReleased:
				switch (event.key.scancode)
				{
					case sf::Keyboard::Scan::D:
						is.d_pressed = false;
						break;
					case sf::Keyboard::Scan::LControl:
						is.ctrl_pressed = false;
						break;
					case sf::Keyboard::Scan::M:
						is.m_pressed = false;
						break;
					case sf::Keyboard::Scan::N:
						is.n_pressed = false;
						break;
					case sf::Keyboard::Scan::B:
						is.b_pressed = false;
						break;
					case sf::Keyboard::Scan::C:
						is.c_pressed = false;
						break;
				}
				break;
		}
	}

}

inline void process_inputs_setup_custom(sf::RenderWindow& window, std::vector<body>& bodies, std::vector<sf::CircleShape>& shapes, input_settings& is, sim_settings& ss)
{

	sf::Event event;
	while (window.pollEvent(event))
	{
		switch (event.type)
		{
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::MouseButtonPressed:
				if (event.mouseButton.button == sf::Mouse::Left)
				{
					is.mouse_clicked = true;
					// abusing the ss struct here instead of adding new fields
					// the reset to the setup screen will reset it anyway
					float x = 100 * is.zoom * (event.mouseButton.x / settings::window_width_h - 1) - is.center_x;
					float y = -(100 / settings::aspect_ratio) * is.zoom * (event.mouseButton.y / settings::window_height_h - 1) - is.center_y;
					constexpr float radius = 2000 / settings::mass_radius_factor;
					body new_body {2000, radius, x, y, 0, 0};
					bodies.push_back(new_body);
					shapes.resize(bodies.size());
					is.new_body_index = bodies.size() - 1;
					ss.n += 1;
				}
				break;
			case sf::Event::MouseButtonReleased:
				if (event.mouseButton.button == sf::Mouse::Left)
					is.mouse_clicked = false;
				break;
			case sf::Event::MouseWheelMoved:
				if (is.mouse_clicked)
				{
					bodies[is.new_body_index].m = std::max(0.0, bodies[is.new_body_index].m * (1 + 0.1 * event.mouseWheel.delta));
					bodies[is.new_body_index].r = bodies[is.new_body_index].m / settings::mass_radius_factor;
				}
				else
					is.zoom *= 1 - ((0.05 + (int)is.ctrl_pressed * settings::zoom_modifier) * event.mouseWheel.delta);
				break;
			case sf::Event::KeyPressed:
				switch (event.key.scancode)
				{
					case sf::Keyboard::Scan::R:
						bodies.resize(0);
						shapes.resize(0);
						ss.n = 0;
						break;
					case sf::Keyboard::Scan::Enter:
					case sf::Keyboard::Scan::NumpadEnter:
						is.program_state = sim;
						break;
					case sf::Keyboard::Scan::LControl:
						is.ctrl_pressed = true;
						break;
				}
				break;
			case sf::Event::KeyReleased:
				switch (event.key.scancode)
				{
					case sf::Keyboard::Scan::LControl:
						is.ctrl_pressed = false;
						break;
				}
				break;
		}
	}

}