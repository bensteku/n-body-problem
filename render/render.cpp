#include "render.hpp"

#include "../misc/settings.hpp"

sf::CircleShape* init_shapes(body* bodies, size_t num_bodies)
{
	sf::CircleShape* shapes = new sf::CircleShape[num_bodies];
	
	for (size_t i = 0; i < num_bodies; i++)
	{
		shapes[i].setRadius(bodies[i].m / settings::mass_radius_factor);
		shapes[i].setPosition(bodies[i].x, bodies[i].y);
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
		shapes[i].setPosition(bodies[i].x, bodies[i].y);
	}
}
