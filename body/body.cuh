#pragma once

void calc_force_cuda(float* x, float* y, float* m, float* r, float* v_x, float* v_y, float g, float t, int N, int it);