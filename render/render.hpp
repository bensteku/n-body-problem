#pragma once

#include <SFML/Graphics.hpp>

#include "../body/body.hpp"

sf::CircleShape* init_shapes(body* bodies, size_t num_bodies);

void render_shapes(sf::RenderWindow& window, sf::CircleShape* shapes, size_t num_bodies);
void update_shapes(sf::RenderWindow& window, sf::CircleShape* shapes, body* bodies, size_t num_bodies);