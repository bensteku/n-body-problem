#include <SFML/Graphics.hpp>

#include <thread>
#include <iostream>
#include <immintrin.h>

#include "misc/settings.hpp"
#include "body/body.hpp"
#include "render/render.hpp"
#include "render/input.hpp"

// TODO:
// - add coordinate system (and an on/off toggle for it)
// - add fixed bodies
// - add colored bodies and add some way to track their trajectories
// - seperate calculations into threads
// - eventually: have a small gui where user can pick between initialization methods or add new bodies by clicking, with mass/size being determined by the duration of the click; also bodies should be destroyable by clicking on them
// - run processing with some combination of SIMD, multi-threading and GPU

int main(int argc, char* argv[]) 
{

	sf::RenderWindow window(sf::VideoMode(settings::window_width, settings::window_height), "n body problem");
	window.setFramerateLimit(settings::frame_rate_cap);

	// renderer class, holds some information about text labels
	Renderer renderer;

	// bodies will be created in world space from -100 to 100 initially
	//std::vector<body> bodies = init_bodies_uniform(settings::n_bodies, 2000, 2000);
	std::vector<body> bodies = init_bodies_circle(settings::n_bodies, 2000, 2000, 50);
	//std::vector<body> bodies = init_bodies_normal(settings::n_bodies, 2000, 2000);
	
	// SFML shapes
	std::vector<sf::CircleShape> shapes = init_shapes(bodies);

	// struct that will contain information about the user's inputs
	// the main thread, which is handling input events, will write into it, the rendering read from it
	// (events being handled by the main thread is the recommended way for SFML)
	input_settings input_info;

	// struct that will contain values for the physical calculations
	// modifiable by user input, uses values from settings as default
	processing_args processing_info{ settings::timestep, settings::g };

	// in case we're running SIMD processing, we allocate a bunch of registers beforehand
	__m256* x_vec;
	__m256* y_vec;
	__m256* m_vec;
	__m256* r_vec;
	if (settings::pt == SIMD)
	{
		const size_t num_elements = bodies.size();
		const size_t num_packed_elements = 1 + num_elements / 8;
		// not freeing those as they die with the program anyway
		x_vec = new __m256[num_packed_elements];
		y_vec = new __m256[num_packed_elements];
		m_vec = new __m256[num_packed_elements];
		r_vec = new __m256[num_packed_elements];
	}
	else
	{
		x_vec = nullptr;
		y_vec = nullptr;
		m_vec = nullptr;
		r_vec = nullptr;
	}

	while (window.isOpen())
	{
		process_inputs(window, input_info, processing_info);

		window.clear();
		renderer.render(window, shapes, input_info, processing_info);
		window.display();

		switch (settings::pt)
		{
			case SISD:
				process_bodies(bodies, processing_info);
				break;
			case SIMD:
				process_bodies_simd(bodies, processing_info, x_vec, y_vec, m_vec, r_vec);
				break;
			case CUDA:
				exit(0);
		}

		renderer.update_shapes(window, shapes, bodies, input_info.center_x, input_info.center_y, input_info.zoom);
	}

	return 0;

}