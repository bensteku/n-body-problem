#include "util.hpp" 

std::array<std::vector<__m256>, 4> set_up_simd_registers(size_t num_elements)
{

	const size_t num_packed_elements = std::ceil(static_cast<float>(num_elements) / 8);

	std::vector<__m256> x_vec;
	std::vector<__m256> y_vec;
	std::vector<__m256> mass_vec;
	std::vector<__m256> r_vec;

#	ifdef USE_SIMD
	x_vec.resize(num_packed_elements);
	y_vec.resize(num_packed_elements);
	mass_vec.resize(num_packed_elements);
	r_vec.resize(num_packed_elements);
#	endif

	return std::array{x_vec, y_vec, mass_vec, r_vec};

}