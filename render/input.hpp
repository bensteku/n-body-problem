#pragma once

#include <SFML/Graphics.hpp>

struct input_settings
{
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
};

void process_inputs(sf::Window& window, input_settings& is);