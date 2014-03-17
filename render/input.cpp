#include "input.hpp"

#include "../misc/settings.hpp"

void process_inputs(sf::Window& window, input_settings& is)
{

	sf::Event event;
	while (window.pollEvent(event))
	{
		if (event.type == sf::Event::Closed)
			window.close();
		else if (event.type == sf::Event::MouseWheelMoved)
			is.zoom *= 1 - ((0.05 + (int)is.ctrl_pressed * settings::zoom_modifier) * event.mouseWheel.delta);
		else if (event.type == sf::Event::MouseButtonPressed)
		{
			if (event.mouseButton.button == sf::Mouse::Left)
			{
				is.mouse_x = event.mouseButton.x;
				is.mouse_y = event.mouseButton.y;
				is.mouse_clicked = true;
			}
		}
		else if (event.type == sf::Event::MouseMoved)
		{
			if (is.mouse_clicked)
			{
				is.center_x += (is.zoom + settings::drag_modifier * (int)is.ctrl_pressed) * (event.mouseMove.x - is.mouse_x) / (100);
				is.center_y -= (is.zoom + settings::drag_modifier * (int)is.ctrl_pressed) * (event.mouseMove.y - is.mouse_y) / (100);
				is.mouse_x = event.mouseMove.x;
				is.mouse_y = event.mouseMove.y;
			}
		}
		else if (event.type == sf::Event::MouseButtonReleased)
		{
			if (event.mouseButton.button == sf::Mouse::Left)
			{
				// TODO: make this a bit more refined
				is.center_x += (is.zoom + settings::drag_modifier * (int)is.ctrl_pressed) * (event.mouseButton.x - is.mouse_x) / (100);
				is.center_y -= (is.zoom + settings::drag_modifier * (int)is.ctrl_pressed) * (event.mouseButton.y - is.mouse_y) / (100);
				is.mouse_clicked = false;
			}
		}
		else if (event.type == sf::Event::KeyPressed)
		{
			if (event.key.scancode == sf::Keyboard::Scan::R)
			{
				is.zoom = 1.0;
				is.center_x = 0;
				is.center_y = 0;
			}
			else if (event.key.scancode == sf::Keyboard::Scan::LControl)
			{
				is.ctrl_pressed = true;
			}
		}
		else if (event.type == sf::Event::KeyReleased)
		{
			if (event.key.scancode == sf::Keyboard::Scan::LControl)
			{
				is.ctrl_pressed = false;
			}
		}
	}
	
}