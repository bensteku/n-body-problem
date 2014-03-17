#pragma once

namespace settings
{

	// render settings
	static unsigned int constexpr window_width = 1280;
	static unsigned int constexpr window_height = 720;
	static float constexpr window_width_h = window_width / 2.0;
	static float constexpr window_height_h = window_height / 2.0;
	static float constexpr aspect_ratio = (float)window_width / window_height;
	static float constexpr frame_rate_cap = 10000;
	static float constexpr zoom_modifier = 0.1;
	static float constexpr drag_modifier = 3.0;

	// physics settings
	static size_t constexpr n_bodies = 60;
	static float constexpr timestep = 0.005;
	//static float constexpr g = 6.6743e-11;
	static float constexpr g = 6.6743e-3;

	// drawing settings
	static float constexpr mass_radius_factor = 200;

	// constants
	static float constexpr pi = 3.14159265358;

}