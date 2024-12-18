#include <SFML/Graphics.hpp>

#ifdef THREED
#include <GL/glew.h>
#endif

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

int main(int argc, char* argv[]) 
{

	sf::RenderWindow window(sf::VideoMode(settings::window_width, settings::window_height), "n body problem");
	window.setFramerateLimit(settings::frame_rate_cap);

#ifdef THREED
	// initialize GLEW
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		return -1;
	}
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glViewport(0, 0, settings::window_width, settings::window_height);
#endif

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
	sim_settings sim_info {settings::timestep, settings::g, settings::n_bodies};

	// set up the various scenes that make up the program
	SetupScene setup_scene(window, bodies, shapes, input_info, sim_info);
	SetupSceneCircle setup_circle_scene(window, bodies, shapes, input_info, sim_info);
	SetupSceneUniform setup_uniform_scene(window, bodies, shapes, input_info, sim_info);
	SetupSceneNormal setup_normal_scene(window, bodies, shapes, input_info, sim_info);
	SetupSceneCustom setup_custom_scene(window, bodies, shapes, input_info, sim_info);
	SimScene sim_scene(window, bodies, shapes, input_info, sim_info);

	// array that will work as our state switch
	Scene* scenes[] = {&setup_scene, &setup_circle_scene, &setup_uniform_scene, &setup_normal_scene, &setup_custom_scene, &sim_scene};

	// main loop
	while (window.isOpen())
	{
		State state_before = scenes[input_info.program_state]->process_inputs();

	#ifdef THREED
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	#else
		window.clear();
	#endif
		scenes[input_info.program_state]->render(state_before);
		window.display();
	}

	return 0;

}