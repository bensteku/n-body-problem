#pragma once

namespace settings
{

	// render settings
	static constexpr unsigned int window_width = 1280;
	static constexpr unsigned int window_height = 720;
	static constexpr float window_width_h = window_width / 2.0;
	static constexpr float window_height_h = window_height / 2.0;
	static constexpr float aspect_ratio = (float)window_width / window_height;
	static constexpr float frame_rate_cap = 10000;
	static constexpr float zoom_modifier = 0.1;
	static constexpr float drag_modifier = 6.0;
	static constexpr int fps_smoother = 100;

	// physics settings
	static constexpr size_t n_bodies = 8000;
	static constexpr float timestep = 0.005;
	static constexpr float g = 6.6743e-3;

	// drawing settings
	static constexpr float mass_radius_factor = 200;

	// constants
	static constexpr float pi = 3.14159265358;

}