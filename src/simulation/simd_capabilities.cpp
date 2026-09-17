#include "simulation/simd_capabilities.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace nbody {

SimdCapabilities detectSimdCapabilities() {
    SimdCapabilities result;
#if defined(_MSC_VER)
    int registers[4]{};
    __cpuid(registers, 0);
    if (registers[0] < 1) return result;

    __cpuidex(registers, 1, 0);
    const bool osxsave = (registers[2] & (1 << 27)) != 0;
    const bool hardware_avx = (registers[2] & (1 << 28)) != 0;
    if (!osxsave || !hardware_avx) return result;
    const unsigned __int64 xcr0 = _xgetbv(0);
    result.avx = (xcr0 & 0x6) == 0x6;
    if (!result.avx) return result;
    result.fma = (registers[2] & (1 << 12)) != 0;

    __cpuid(registers, 0);
    if (registers[0] < 7) return result;
    __cpuidex(registers, 7, 0);
    result.avx2 = (registers[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    result.avx = __builtin_cpu_supports("avx") != 0;
    result.avx2 = __builtin_cpu_supports("avx2") != 0;
    result.fma = __builtin_cpu_supports("fma") != 0;
#endif
    return result;
}

}
