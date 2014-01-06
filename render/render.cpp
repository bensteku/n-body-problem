#include "render.hpp"

#include "../misc/settings.hpp"

sf::CircleShape* init_shapes(body* bodies, size_t num_bodies)
{
	sf::CircleShape* shapes = new sf::CircleShape[num_bodies];
	
	for (size_t i = 0; i < num_bodies; i++)
	{
		shapes[i].setRadius(bodies[i].m / settings::mass_radius_factor);
		float x_pos = settings::window_width_h + settings::window_width_h * bodies[i].x;
		float y_pos = settings::window_height_h - settings::window_height_h * bodies[i].y;
		shapes[i].setPosition(x_pos, y_pos);
	}
	
	return shapes;
}

void render_shapes(sf::RenderWindow& window, sf::CircleShape* shapes, size_t num_bodies)
{
	for (size_t i = 0; i < num_bodies; i++)
	{
		window.draw(shapes[i]);
	}
}

void update_shapes(sf::RenderWindow& window, sf::CircleShape* shapes, body* bodies, size_t num_bodies)
{
	for (size_t i = 0; i < num_bodies; i++)
	{
		float x_pos = settings::window_width_h + settings::window_width_h * bodies[i].x;
		float y_pos = settings::window_height_h - settings::window_height_h * bodies[i].y;
		shapes[i].setPosition(x_pos, y_pos);
	}
}
