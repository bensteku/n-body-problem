#pragma once

#include <vector>
#include <array>
#include <string>
#include <thread>
#include <barrier>
#include <SFML/Graphics.hpp>

#include "../body/body.hpp"
#include "input.hpp"
#include "../misc/util.hpp"

std::vector<sf::CircleShape> init_shapes(const std::vector<body>& bodies);

const std::string ui_string1 =
	#ifdef USE_CUDA
		"CUDA";
	#elif defined(USE_SIMD)
		"SIMD";
	#else
		"SISD";
	#endif


const std::string ui_string2 =
	#ifdef USE_THREADS
		" (multi-threaded)";
	#else
		"";
	#endif

const std::string ui_string3 =
	#ifdef USE_OCTREE
		" (Octree)";
	#else
		"";
	#endif

class Scene
{
	private:
		static inline const std::string m_fps_str = "FPS: ";
		static inline const std::string m_gravity_str = "Gravity: ";
		static inline const std::string m_timestep_str = "Timestep: ";
		static inline const std::string m_bodies_str = "Number of bodies: ";
	#ifdef USE_OCTREE
		static inline const std::string m_octree_str1 = "Octree Node Size: ";
		static inline const std::string m_octree_str2 = "Octree Node Tolerance: ";
	#endif
		static inline const std::string m_processing_type = ui_string1 + ui_string2 + ui_string3;

		int m_frame_counter = 1;
	protected:
		sf::RenderWindow& m_window_ref;
		std::vector<body>& m_bodies_ref;
		std::vector<sf::CircleShape>& m_shapes_ref;
		input_settings& m_is_ref;
		sim_settings& m_ss_ref;
	#ifdef USE_SIMD
		// reference to the vector that will contain our SIMD registers
		static inline std::array<std::vector<__m256>, 4> m_registers = set_up_simd_registers(settings::n_bodies);
	#endif


		static inline sf::Font m_font;
		static inline sf::Text m_upper_left_text;
		static inline sf::Clock m_clock;

		void render_fps_info();
		void update_shapes();
	public:
		Scene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref);
		
		virtual State process_inputs() = 0;
		virtual void render(State state_before) = 0;

};

class SetupScene : public Scene
{

	private:
		sf::Text m_setup_text;

		static inline const std::string m_settings_str = \
			"Press 1 to initialize bodies in a circle.\nPress 2 to initialize bodies uniformly random.\nPress 3 to initialize bodies in a random normal distribution.\nPress 4 to place bodies yourself.";
	public:
		SetupScene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref);

		State process_inputs();
		void render(State state_before);

};

class SetupSceneCircle : public Scene
{
	
	private:
		sf::Text m_setup_circle_text;

		const std::string m_settings_circle_str = \
			"Click and hold to determine radius.\nHold d + scroll mouse wheel to change random deviation from circle.\nHold m + scroll mouse wheel to change max mass, n + scroll for min mass.\nHold b + scroll mouse wheel to change number of bodies.\nPress Enter to start simulation.";
		// buffer when the user changes the amount of bodies
		size_t m_previous_size = 0;
	public:
		SetupSceneCircle(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref);

		State process_inputs();
		void render(State state_before);

};

class SetupSceneUniform : public Scene
{

	private:
		sf::Text m_setup_uniform_text;

		 const std::string m_settings_uniform_str = \
			"Hold d/c + scroll mouse wheel to change x/y range.\nHold m + scroll mouse wheel to change max mass, n + scroll for min mass.\nHold b + scroll mouse wheel to change number of bodies.\nPress Enter to start simulation.";
		 // buffer when the user changes the amount of bodies
		 size_t m_previous_size = 0;
	public:
		SetupSceneUniform(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref);

		State process_inputs();
		void render(State state_before);

};

class SetupSceneNormal : public Scene
{

	private:
		sf::Text m_setup_normal_text;

		const std::string m_settings_normal_str = \
			"Hold d/c + scroll mouse wheel to change x/y std.\nHold m + scroll mouse wheel to change max mass, n + scroll for min mass.\nHold b + scroll mouse wheel to change number of bodies.\nPress Enter to start simulation.";
		// buffer when the user changes the amount of bodies
		size_t m_previous_size = 0;
	public:
		SetupSceneNormal(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref);

		State process_inputs();
		void render(State state_before);

};

class SetupSceneCustom : public Scene
{

	private:
		sf::Text m_setup_custom_text;

		const std::string m_settings_custom_str = \
			"Click to add a body.\nHolding the click and scrolling allows to determine the body's mass.\nPress R to delete all bodies placed so far.\nPress Enter to start simulation.";
		// buffer when the user changes the amount of bodies
		size_t m_previous_size = 0;
	public:
		SetupSceneCustom(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref);

		State process_inputs();
		void render(State state_before);

};

class SimScene : public Scene
{

	private:
		// if CUDA is used, we create pointers for memory on the GPU
	#ifdef USE_CUDA
		body* m_d_bodies;
		float* m_d_interactions_x;
		float* m_d_interactions_y;
	#endif

		// if we run the program multi-threaded, we create a pool of threads to use
	#ifdef USE_THREADS
		std::vector<std::thread> m_threads;
		// have to allocate on heap to avoid changing the constructor's interface
		std::unique_ptr<std::barrier<>> m_compute_barrier1;
		std::unique_ptr<std::barrier<>> m_compute_barrier2;
		std::unique_ptr<std::barrier<>> m_render_barrier;
		// sychronization helpers
		std::atomic<bool> m_terminate_signal = false;
		bool m_run_signal = false;
		std::mutex m_run_mutex;
		std::condition_variable m_run_cv;
	#endif
	public:
		SimScene(sf::RenderWindow& window_ref, std::vector<body>& bodies_ref, std::vector<sf::CircleShape>& shapes_ref, input_settings& is_ref, sim_settings& ss_ref);
	#ifdef USE_THREADS
		// only needed in the multi-threaded case to shut the threads safely
		~SimScene();
	#endif

		State process_inputs();
		void render(State state_before);

};