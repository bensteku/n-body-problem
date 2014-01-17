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

void render_shapes(sf::RenderWindow& window, const std::vector<sf::CircleShape>& shapes)
{
	for (const sf::CircleShape& it : shapes)
	{
		window.draw(it);
	}
}

void update_shapes(sf::RenderWindow& window, std::vector<sf::CircleShape>& shapes, const std::vector<body>& bodies, float zoom)
{
	for (size_t i = 0; i < bodies.size(); i++)
	{
		shapes[i].setRadius(bodies[i].m / (settings::mass_radius_factor * zoom));
		float x_pos = settings::window_width_h + settings::window_width_h * bodies[i].x / (100 * zoom);
		float y_pos = settings::window_height_h - settings::aspect_ratio * settings::window_height_h * bodies[i].y / (100 * zoom);
		shapes[i].setPosition(x_pos, y_pos);
	}
}
