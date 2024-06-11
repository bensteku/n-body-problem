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
		static inline std::string m_processing_type;

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
			"Press 1 to initialize bodies in a circle.\nPress 2 to initialize bodies uniformly random.\nPress 3 to initialize bodies in a random normal distribution.";
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

		static inline const std::string m_settings_circle_str = \
			"Click and hold to determine radius.\nHold m + scroll mouse wheel to change max mass, n + scroll for min mass.\nHold b + scroll mouse wheel to change number of bodies.\nHold d + scroll mouse wheel to change random deviation from circle.\nPress Enter to start simulation.";
		// buffer when the user changes the amount of bodies
		size_t m_previous_size = 0;
	public:
		SetupSceneCircle(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers);

		void process_inputs();
		void render();

};

class SimScene : public Scene
{

	private:
		// registers for the SIMD case
		// 0:x, 1:y, 2:mass, 3:radius
		std::array<std::vector<__m256>, 4>& m_registers_ref;
	public:
		SimScene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref, std::array<std::vector<__m256>, 4>& registers);

		void process_inputs();
		void render();

};