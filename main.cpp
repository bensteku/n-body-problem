#include <SFML/Graphics.hpp>

#include "misc/settings.hpp"
#include "body/body.hpp"
#include "render/render.hpp"

#include <iostream>

// TODO:
// - set max and min radius according to screen size and only then map it onto mass
// - prevent bodies from going out of range
// - or make it so that the screenspace can be zoomed out (which would make worldspace calcuations necessary again)
// - migrate to Qt
// - deal with collision
// - add fixed bodies
// - add colored bodies and add some way to track their trajectories
// - seperate gui and calculations into threads
// - run processing with SIMD and/or GPU

int main(int argc, char* argv[]) 
{

	sf::RenderWindow window(sf::VideoMode(settings::window_width, settings::window_height), "n body problem");
	window.setFramerateLimit(settings::frame_rate);

	// creates a number of bodies in a virtual workspace from -1 to 1 on both axes
	//std::vector<body> bodies = init_bodies_uniform(settings::n_bodies, 2000, 2000);
	//std::vector<body> bodies = init_bodies_circle(settings::n_bodies, 2000, 2000, 20);
	std::vector<body> bodies = init_bodies_normal(settings::n_bodies, 2000, 2000);
	std::vector<sf::CircleShape> shapes = init_shapes(bodies);

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		window.clear();
		render_shapes(window, shapes);
		window.display();

		process_bodies(bodies, settings::timestep);
		update_positions(bodies, settings::timestep);
		update_shapes(window, shapes, bodies);

		//std::cout << bodies[0].x << " " << bodies[0].y << std::endl;
	}

	return 0;

}