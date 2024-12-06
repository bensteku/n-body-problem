#pragma once

void process_bodies_cuda(std::vector<body>& bodies, body* d_bodies, float* d_interactions_x, float* d_interactions_y, const sim_settings& ss);