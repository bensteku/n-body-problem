#pragma once

namespace settings
{
	static unsigned int constexpr window_width = 1280;
	static unsigned int constexpr window_height = 720;
	static float constexpr window_width_h = window_width / 2.0;
	static float constexpr window_height_h = window_height / 2.0;

	static float constexpr frame_rate = 1000;
	static float constexpr frame_time = 1.0 / frame_rate;

	static size_t constexpr n_bodies = 10;
	static float constexpr timestep = 0.05;
	static float constexpr g = 6.6743e-11;

	static float constexpr mass_radius_factor = 100;
}