#include <SFML/Graphics.hpp>

#include <thread>
#include <iostream>

#include "misc/settings.hpp"
#include "body/body.hpp"
#include "render/render.hpp"
#include "render/input.hpp"

// TODO:
// - propagate FPS
// - add coordinate system (and an on/off toggle for it)
// - add fixed bodies
// - add colored bodies and add some way to track their trajectories
// - seperate gui and calculations into threads
// - eventually: have a small gui where user can pick between initialization methods or add new bodies by clicking, with mass/size being determined by the duration of the click; also bodies should be destroyable by clicking on them
// - run processing with some combination of SIMD, multi-threading and GPU

int main(int argc, char* argv[]) 
{

	sf::RenderWindow window(sf::VideoMode(settings::window_width, settings::window_height), "n body problem");
	window.setFramerateLimit(settings::frame_rate_cap);

	Renderer renderer;

	// bodies will be created in world space from -100 to 100 initially
	//std::vector<body> bodies = init_bodies_uniform(settings::n_bodies, 2000, 2000);
	std::vector<body> bodies = init_bodies_circle(settings::n_bodies, 2000, 2000, 50);
	//std::vector<body> bodies = init_bodies_normal(settings::n_bodies, 2000, 2000);
	std::vector<sf::CircleShape> shapes = init_shapes(bodies);

	// set up the struct that will contain information about the user's inputs that are relevant
	// the main thread, which is handling input events, will write into it, the rendering read from it
	// (events being handled by the main thread is the recommended way for SFML)
	input_settings input_info;

	while (window.isOpen())
	{
		process_inputs(window, input_info);

		window.clear();
		renderer.render(window, shapes, input_info);
		window.display();

		process_bodies(bodies, settings::timestep);

		renderer.update_shapes(window, shapes, bodies, input_info.center_x, input_info.center_y, input_info.zoom);
	}

	return 0;

}