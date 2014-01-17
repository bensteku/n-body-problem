#pragma once

#include <vector>
#include <SFML/Graphics.hpp>

#include "../body/body.hpp"

std::vector<sf::CircleShape> init_shapes(const std::vector<body>& bodies);

void render_shapes(sf::RenderWindow& window, const std::vector<sf::CircleShape>& shapes);
void update_shapes(sf::RenderWindow& window, std::vector<sf::CircleShape>& shapes, const std::vector<body>& bodies, float zoom=1.0);