# Performance scratchpad

This file records plausible performance investigations that are deliberately
deferred until profiling shows they are worthwhile.

## Integration kernel

The integration pass is currently shared by all engines through the bulk
operations on `BodyStorage`. It is scalar and single-threaded. Potential future
variants are:

- SIMD integration using component-wise position, velocity, and acceleration
  data;
- multithreaded integration by partitioning disjoint body ranges;
- a backend-specific fused integration kernel for SIMD or GPU engines.

This is intentionally not an active optimization target. Recent measurements
show integration taking only approximately `0.02–0.03 ms` for a 2,000-body
scenario, while force calculation takes several milliseconds. SIMD setup,
additional component storage, synchronization, and worker wake-up costs could
therefore outweigh the benefit at ordinary body counts.

Revisit this item when profiling a representative large-body workload shows
integration becoming material relative to force and collision phases. Any
implementation must preserve the existing static-body behavior, 2D Z-axis
normalization, numerical parity, and the explicit single-threaded/MT runtime
setting.
