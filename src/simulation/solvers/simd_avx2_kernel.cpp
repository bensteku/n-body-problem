#include "simulation/solvers/simd_avx2_kernel.hpp"

#include <immintrin.h>
#include <cmath>

namespace nbody {

void calculateAvx2Accelerations(const WorldState& world, const SimulationParameters& parameters,
                                std::vector<Vec3>& output) {
    const BodyStorage& storage = world.bodyStorage();
    storage.synchronizePositionComponents();
    const std::size_t count = world.bodyCount();
    output.assign(count, Vec3{});
    const auto& x = storage.positionX();
    const auto& y = storage.positionY();
    const auto& z = storage.positionZ();
    const auto& masses = storage.masses();
    const bool three_dimensional = world.dimension() == Dimension::Three;
    const double softening_squared = parameters.softening_length * parameters.softening_length;
    const __m256d softening = _mm256_set1_pd(softening_squared);
    const __m256d gravitational_constant = _mm256_set1_pd(parameters.gravitational_constant);

    for (std::size_t target = 0; target < count; ++target) {
        if (world.body(target).is_static()) continue;
        const __m256d target_x = _mm256_set1_pd(x[target]);
        const __m256d target_y = _mm256_set1_pd(y[target]);
        const __m256d target_z = _mm256_set1_pd(z[target]);
        __m256d acceleration_x = _mm256_setzero_pd();
        __m256d acceleration_y = _mm256_setzero_pd();
        __m256d acceleration_z = _mm256_setzero_pd();
        double scalar_x = 0.0;
        double scalar_y = 0.0;
        double scalar_z = 0.0;

        std::size_t source = 0;
        for (; source + 3 < count; source += 4) {
            const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(x.data() + source), target_x);
            const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(y.data() + source), target_y);
            __m256d dz;
            __m256d distance_squared = _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy));
            if (three_dimensional) {
                dz = _mm256_sub_pd(_mm256_loadu_pd(z.data() + source), target_z);
                distance_squared = _mm256_add_pd(distance_squared, _mm256_mul_pd(dz, dz));
            }
            distance_squared = _mm256_max_pd(_mm256_add_pd(distance_squared, softening),
                                             _mm256_set1_pd(1e-30));
            const __m256d distance = _mm256_sqrt_pd(distance_squared);
            const __m256d inverse_cubed = _mm256_div_pd(_mm256_set1_pd(1.0),
                _mm256_mul_pd(distance_squared, distance));
            const __m256d scale = _mm256_mul_pd(gravitational_constant,
                _mm256_mul_pd(_mm256_loadu_pd(masses.data() + source), inverse_cubed));
            acceleration_x = _mm256_add_pd(acceleration_x, _mm256_mul_pd(dx, scale));
            acceleration_y = _mm256_add_pd(acceleration_y, _mm256_mul_pd(dy, scale));
            if (three_dimensional) acceleration_z = _mm256_add_pd(acceleration_z, _mm256_mul_pd(dz, scale));
        }

        alignas(32) double lanes_x[4];
        alignas(32) double lanes_y[4];
        alignas(32) double lanes_z[4];
        _mm256_store_pd(lanes_x, acceleration_x);
        _mm256_store_pd(lanes_y, acceleration_y);
        _mm256_store_pd(lanes_z, acceleration_z);
        for (double lane : lanes_x) scalar_x += lane;
        for (double lane : lanes_y) scalar_y += lane;
        for (double lane : lanes_z) scalar_z += lane;
        for (; source < count; ++source) {
            if (source == target) continue;
            const Vec3 displacement = world.body(source).position - world.body(target).position;
            const double distance_squared = displacement.lengthSquared(world.dimension()) + softening_squared;
            if (distance_squared == 0.0) continue;
            const double distance = std::sqrt(distance_squared);
            const double scale = parameters.gravitational_constant * masses[source]
                / (distance_squared * distance);
            scalar_x += displacement.x * scale;
            scalar_y += displacement.y * scale;
            if (world.dimension() == Dimension::Three) scalar_z += displacement.z * scale;
        }
        output[target] = {scalar_x, scalar_y, three_dimensional ? scalar_z : 0.0};
    }
}

}
