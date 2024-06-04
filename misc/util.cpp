#include "util.hpp" 

// function that returns a bitmask for values that are infinite
// taken from https://stackoverflow.com/a/30713827
__m256 is_infinity(__m256 x) 
{
	const __m256 SIGN_MASK = _mm256_set1_ps(-0.0);
	const __m256 INF = _mm256_set1_ps(std::numeric_limits<float>::infinity());

	x = _mm256_andnot_ps(SIGN_MASK, x);
	x = _mm256_cmp_ps(x, INF, _CMP_EQ_OQ);
	return x;
}

// function to reduce a register down to a single float
// modified by me, but originally taken from https://stackoverflow.com/a/49943540
float reduce_register(__m256 v)
{
	// step 1: split 8 floats into two registers of 4 floats and add them together
	__m128 vlow = _mm256_castps256_ps128(v);
	__m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
	vlow = _mm_add_ps(vlow, vhigh);     // reduce down to 128, 4 numbers

	// step 2: do the same again, after which the first 64 bits hold two 32bit floats
	__m128 high64 = _mm_unpackhi_ps(vlow, vlow);  // extract upper 64 bits
	// the first 64 bits now hold the two sums of the rest of the register
	vlow = _mm_add_ss(vlow, _mm_shuffle_ps(high64, high64, _MM_SHUFFLE(0, 2, 1, 3)));
	// add the register to a shuffled version of itself, where the shuffle moves the second 32 bits into the first 32 bits
	vlow = _mm_add_ss(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 0, 3, 4)));
	return  _mm_cvtss_f32(vlow);  // extract the overall sum from the first 32 bits
}