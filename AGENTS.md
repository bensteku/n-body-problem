This repo contains a simulation for gravitational attraction between bodies. The simulation is implemented with different computational backends and features an interactive UI for users to manipulate the simulation mid-flight.

# Rules

- PLAN.md is the canonical source desing document. Since it is quite large, you will not autonomously open or read it. You will receive explicit instructions whenever it is directly relevant and only then will you read it.
- You will not under any circumstances modify PLAN.md unless explicitly instructed to do so.
- A turn can leave code in non-compiling or inconsistent state, as long as the turn is part of a broader overhaul that will eventually leave the code in workable state again.
- You are working under the principles of Data Driven Design (DDD).
    - You will try to achieve good cache coherency and data efficiency.
    - You will try to write code that compiles into efficient instructions, e.g. vectorization.
    - In your responses you will highlight tradeoffs with regards to performance and make suggestions improve adherence to DDD-principles.
- Backend/solver combinations are integrated physics engines. They own the complete numerical step, including force calculation, integration, collision and boundary handling, deferred outcomes, diagnostics, and future physics modules. Do not assume that a module must share the same implementation or data structure across Scalar, SIMD, and GPU engines.
- Design each engine from first principles around its execution model, memory access patterns, vectorization, device transfer, and synchronization costs. Existing engines are semantic references, not implementation templates. Prefer backend-specific fusion and data layouts when profiling supports them. Despite internal specialization, engines must be externally interchangeable: expose the same lifecycle and stepping API, configuration semantics, diagnostics, events, state views, validation behavior, and failure/transition reporting to outside components.
- Numerical and behavioral parity is the binding contract between engines, especially relative to Scalar Full, within explicit documented tolerances. Validate positions, velocities, conserved quantities, body identities/counts, collision/fragmentation/absorption outcomes, and determinism guarantees as applicable. Do not trade away semantic parity for a benchmark result without explicit direction.
