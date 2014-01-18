#include <SFML/Graphics.hpp>

#include "misc/settings.hpp"
#include "body/body.hpp"
#include "render/render.hpp"

#include <iostream>

// TODO:
// - seperate out keyboard and mouse handling
// - deal with collision
// - add fixed bodies
// - add colored bodies and add some way to track their trajectories
// - seperate gui and calculations into threads
// - eventually: have a small gui where user can pick between initialization methods or add new bodies by clicking, with mass/size being determined by the duration of the click; also bodies should be destroyable by clicking on them
// - run processing with SIMD and/or GPU

int main(int argc, char* argv[]) 
{

	sf::RenderWindow window(sf::VideoMode(settings::window_width, settings::window_height), "n body problem");
	window.setFramerateLimit(settings::frame_rate);

	// rendering values at start
	float zoom = 1.0;
	float center_x = 50.0;
	float center_y = 25.0;

	// bodies will be created in world space from -100 to 100 initially
	//std::vector<body> bodies = init_bodies_uniform(settings::n_bodies, 2000, 2000);
	std::vector<body> bodies = init_bodies_circle(settings::n_bodies, 2000, 2000, 20);
	//std::vector<body> bodies = init_bodies_normal(settings::n_bodies, 2000, 2000);
	std::vector<sf::CircleShape> shapes = init_shapes(bodies);

	// mouse drag variables
	int mouse_x = 0;
	int mouse_y = 0;
	bool mouse_clicked = false;

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
			else if (event.type == sf::Event::MouseWheelMoved)
				zoom *= 1 + (0.05 * event.mouseWheel.delta);
			else if (event.type == sf::Event::MouseButtonPressed)
			{
				if (event.mouseButton.button == sf::Mouse::Left)
				{
						mouse_x = event.mouseButton.x;
						mouse_y = event.mouseButton.y;
						mouse_clicked = true;
				}
			}
			else if (event.type == sf::Event::MouseMoved)
			{
				if (mouse_clicked)
				{
					center_x += zoom * (event.mouseMove.x - mouse_x) / (100);
					center_y += zoom * (event.mouseMove.y - mouse_y) / (100);
					mouse_x = event.mouseMove.x;
					mouse_y = event.mouseMove.y;
				}
				std::cout << event.mouseMove.x << " " << event.mouseMove.y << std::endl;
			}
			else if (event.type == sf::Event::MouseButtonReleased)
			{
				if (event.mouseButton.button == sf::Mouse::Left)
				{
					// TODO: make this a bit more refined
					center_x += zoom * (event.mouseButton.x - mouse_x) / (100);
					center_y += zoom * (event.mouseButton.y - mouse_y) / (100);
					mouse_clicked = false;
				}
			}
			else if (event.type == sf::Event::KeyPressed)
			{
				if (event.key.scancode == sf::Keyboard::Scan::R)
				{
					zoom = 1.0;
					center_x = 0;
					center_y = 0;
				}
			}
		}

		window.clear();
		render_shapes(window, shapes);
		window.display();

		process_bodies(bodies, settings::timestep);
		update_positions(bodies, settings::timestep);
		update_shapes(window, shapes, bodies, center_x, center_y, zoom);
	}

	return 0;

}