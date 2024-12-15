#pragma once

#include <bit>
#include <limits>
#include <cstdint>
#include <vector>
#include <array>
#include "immintrin.h"

std::array<std::vector<__m256>, 4> set_up_simd_registers(size_t num_elements);

constexpr float rsqrt(float number) noexcept
{

    static_assert(std::numeric_limits<float>::is_iec559); // (enable only on IEEE 754)

    float const y = std::bit_cast<float>(
        0x5f3759df - (std::bit_cast<std::uint32_t>(number) >> 1));
    return y * (1.5f - (number * 0.5f * y * y));

}

// function that returns a bitmask for values that are infinite
// taken from https://stackoverflow.com/a/30713827
inline __m256 is_infinity(__m256 x)
{

	const __m256 SIGN_MASK = _mm256_set1_ps(-0.0);
	const __m256 INF = _mm256_set1_ps(std::numeric_limits<float>::infinity());

	x = _mm256_andnot_ps(SIGN_MASK, x);
	x = _mm256_cmp_ps(x, INF, _CMP_EQ_OQ);
	return x;

}

// function to reduce a register down to a single float
// taken from https://stackoverflow.com/a/13222410
inline float reduce_register(__m256 v)
{

    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(v, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(v);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);

}