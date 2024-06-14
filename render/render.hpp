#pragma once

#include <vector>
#include <array>
#include <string>
#include <SFML/Graphics.hpp>

#include "../body/body.hpp"
#include "input.hpp"
#include "../misc/util.hpp"

std::vector<sf::CircleShape> init_shapes(const std::vector<body>& bodies);

class Scene
{
	private:
		static inline const std::string m_fps_str = "FPS: ";
		static inline const std::string m_gravity_str = "Gravity: ";
		static inline const std::string m_timestep_str = "Timestep: ";
		static inline const std::string m_bodies_str = "Number of bodies: ";
		static inline const std::string m_processing_type =
		#if defined(USE_CUDA)
		"CUDA"
		#elif defined(USE_SIMD)
		"SIMD"
		#else
		"SISD"
		#endif
		;

		int m_frame_counter = 1;
	protected:
		sf::RenderWindow& m_window_ref;
		std::vector<body>& m_bodies_ref;
		std::vector<sf::CircleShape>& m_shapes_ref;
		input_settings& m_is_ref;
		sim_settings& m_ss_ref;

		static inline sf::Font m_font;
		static inline sf::Text m_upper_left_text;
		static inline sf::Clock m_clock;

		void render_fps_info();
		void update_shapes();
	public:
		Scene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref);
		
		virtual void process_inputs() = 0;
		virtual void render() = 0;
		
};

class SetupScene : public Scene
{

	private:
		sf::Text m_setup_text;

		static inline const std::string m_settings_str = \
			"Press 1 to initialize bodies in a circle.\nPress 2 to initialize bodies uniformly random.\nPress 3 to initialize bodies in a random normal distribution.\nPress 4 to place bodies yourself.";
	public:
		SetupScene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref);

		void process_inputs();
		void render();

};

class SetupSceneCircle : public Scene
{
	
	private:
		std::array<std::vector<__m256>, 4>& m_registers_ref;

		sf::Text m_setup_circle_text;

		const std::string m_settings_circle_str = \
			"Click and hold to determine radius.\nHold d + scroll mouse wheel to change random deviation from circle.\nHold m + scroll mouse wheel to change max mass, n + scroll for min mass.\nHold b + scroll mouse wheel to change number of bodies.\nPress Enter to start simulation.";
		// buffer when the user changes the amount of bodies
		size_t m_previous_size = 0;
	public:
		SetupSceneCircle(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers);

		void process_inputs();
		void render();

};

class SetupSceneUniform : public Scene
{

	private:
		std::array<std::vector<__m256>, 4>& m_registers_ref;

		sf::Text m_setup_uniform_text;

		 const std::string m_settings_uniform_str = \
			"Hold d/c + scroll mouse wheel to change x/y range.\nHold m + scroll mouse wheel to change max mass, n + scroll for min mass.\nHold b + scroll mouse wheel to change number of bodies.\nPress Enter to start simulation.";
		 // buffer when the user changes the amount of bodies
		 size_t m_previous_size = 0;
	public:
		SetupSceneUniform(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers);

		void process_inputs();
		void render();

};

class SetupSceneNormal : public Scene
{

	private:
		std::array<std::vector<__m256>, 4>& m_registers_ref;

		sf::Text m_setup_normal_text;

		const std::string m_settings_normal_str = \
			"Hold d/c + scroll mouse wheel to change x/y std.\nHold m + scroll mouse wheel to change max mass, n + scroll for min mass.\nHold b + scroll mouse wheel to change number of bodies.\nPress Enter to start simulation.";
		// buffer when the user changes the amount of bodies
		size_t m_previous_size = 0;
	public:
		SetupSceneNormal(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers);

		void process_inputs();
		void render();

};

class SetupSceneCustom : public Scene
{

	private:
		std::array<std::vector<__m256>, 4>& m_registers_ref;

		sf::Text m_setup_custom_text;

		const std::string m_settings_custom_str = \
			"Click to add a body.\nHolding the click and scrolling allows to determine the body's mass.\nPress R to delete all bodies placed so far.\nPress Enter to start simulation.";
		// buffer when the user changes the amount of bodies
		size_t m_previous_size = 0;
	public:
		SetupSceneCustom(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers);

		void process_inputs();
		void render();

};

class SimScene : public Scene
{

	private:
		// registers for the SIMD case
		// 0:x, 1:y, 2:mass, 3:radius
		std::array<std::vector<__m256>, 4>& m_registers_ref;

		// if CUDA is used, we create several pointers for memory on the GPU
#		ifdef USE_CUDA
		float* m_d_x;
		float* m_d_y;
		float* m_d_mass;
		float* m_d_radius;
		float* m_d_v_x;
		float* m_d_v_y;
		// ... and because the data is laid out in terms of body structs
		// (as opposed to in terms arrays of the struct's members)
		// we need floats to hold the same data on the cpu as well
		std::vector<float> m_h_x;
		std::vector<float> m_h_y;
		std::vector<float> m_h_mass;
		std::vector<float> m_h_radius;
		std::vector<float> m_h_v_x;
		std::vector<float> m_h_v_y;
		// memory to keep track of size of last allocation
		size_t m_last_allocation = 0;
#		endif
	public:
		SimScene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers);

		void process_inputs();
		void render();

};