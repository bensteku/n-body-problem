#include <SFML/Graphics.hpp>

#include "misc/settings.hpp"
#include "body/body.hpp"
#include "render/render.hpp"

#include <iostream>

// TODO: set max and min radius according to screen size and only then map it onto mass
// TODO: prevent bodies from going out of range
// TODO: deal with collision
// TODO: run processing with SIMD and/or GPU

int main(int argc, char* argv[]) 
{
	sf::RenderWindow window(sf::VideoMode(settings::window_width, settings::window_height), "n body problem");
	window.setFramerateLimit(settings::frame_rate);

	// creates a number of bodies in a virtual workspace from -1 to 1 on both axes
	body* bodies = init_bodies(settings::n_bodies, gaussian, 1000, 2000);
	sf::CircleShape* shapes = init_shapes(bodies, settings::n_bodies);

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		window.clear();
		render_shapes(window, shapes, settings::n_bodies);
		window.display();

		process_bodies(bodies, settings::n_bodies, settings::timestep);
		update_positions(bodies, settings::n_bodies, settings::timestep);
		update_shapes(window, shapes, bodies, settings::n_bodies);

		//std::cout << bodies[0].x << " " << bodies[0].y << std::endl;
	}
	return 0;
}