#include "util.hpp" 

std::array<std::vector<__m256>, 4> set_up_simd_registers()
{
	std::vector<__m256> m_x_vec;
	std::vector<__m256> m_y_vec;
	std::vector<__m256> m_mass_vec;
	std::vector<__m256> m_r_vec;

	return std::array{m_x_vec, m_y_vec, m_mass_vec, m_r_vec};

}