#include <SFML/Graphics.hpp>

#include <thread>
#include <iostream>
#include <immintrin.h>

#include "misc/settings.hpp"
#include "misc/util.hpp"
#include "body/body.hpp"
#include "render/render.hpp"
#include "render/input.hpp"

// TODO:
// - add colored bodies and add some way to track their trajectories
// - run processing with some combination of SIMD, multi-threading and GPU

int main(int argc, char* argv[]) 
{

	sf::RenderWindow window(sf::VideoMode(settings::window_width, settings::window_height), "n body problem");
	window.setFramerateLimit(settings::frame_rate_cap);

	// body structs holding the info for the simulation
	std::vector<body> bodies;
	bodies.resize(settings::n_bodies);
	
	// SFML shapes
	std::vector<sf::CircleShape> shapes;
	shapes.resize(settings::n_bodies);

	// struct that will contain information about the user's inputs
	input_settings input_info;

	// struct that will contain settings for the physical calculations
	// modifiable by user input, uses values from settings.hpp as default
	sim_settings sim_info{ settings::timestep, settings::g, settings::n_bodies };

	// array of registers that we hand off to the simulation in case SIMD is used
	// we handle them this way such that resizing them all is easier in case the user
	// increases the amount of bodies floating around
	std::array<std::vector<__m256>, 4> registers = set_up_simd_registers();

	// set up the various scenes that make up the program
	SetupScene setup_scene(window, bodies, shapes, input_info, sim_info);
	SetupSceneCircle setup_circle_scene(window, bodies, shapes, input_info, sim_info, registers);
	SetupSceneUniform setup_uniform_scene(window, bodies, shapes, input_info, sim_info, registers);
	SetupSceneNormal setup_normal_scene(window, bodies, shapes, input_info, sim_info, registers);
	SetupSceneCustom setup_custom_scene(window, bodies, shapes, input_info, sim_info, registers);
	SimScene sim_scene(window, bodies, shapes, input_info, sim_info, registers);
	Scene* scenes[] = {&setup_scene, &setup_circle_scene, &setup_uniform_scene, &setup_normal_scene, &setup_custom_scene, &sim_scene};

	// main loop
	while (window.isOpen())
	{
		scenes[input_info.program_state]->process_inputs();

		window.clear();
		scenes[input_info.program_state]->render();
		window.display();
	}

	return 0;

}