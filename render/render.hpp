#pragma once

#include <vector>
#include <string>
#include <SFML/Graphics.hpp>

#include "../body/body.hpp"
#include "input.hpp"

std::vector<sf::CircleShape> init_shapes(const std::vector<body>& bodies);

class Renderer
{
	private:
		sf::Font m_font;
		sf::Text m_fps_text;
		sf::Clock m_clock;
		std::string m_fps_string = "FPS: ";
		std::string m_processing_type = "SISD";

		int m_frame_counter = 1;
	public:
		Renderer(const std::string file_path="./data/fonts/routed-gothic.ttf");
		void render(sf::RenderWindow& window, const std::vector<sf::CircleShape>& shapes, input_settings& is);
		void update_shapes(sf::RenderWindow& window, std::vector<sf::CircleShape>& shapes, const std::vector<body>& bodies, float x_offset = 0, float y_offset = 0, float zoom = 1.0);
};