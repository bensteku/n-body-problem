This repo contains a simulation for gravitational attraction between bodies. The simulation is implemented with different computational backends and features an interactive UI for users to manipulate the simulation mid-flight.

# Rules

- PLAN.md is the canonical source desing document. Since it is quite large, you will not autonomously open or read it. You will receive explicit instructions whenever it is directly relevant and only then will you read it.
- You will not under any circumstances modify PLAN.md unless explicitly instructed to do so.
- A turn can leave code in non-compiling or inconsistent state, as long as the turn is part of a broader overhaul that will eventually leave the code in workable state again.
- You are working under the principles of Data Driven Design (DDD).
    - You will try to achieve good cache coherency and data efficiency.
    - You will try to write code that compiles into efficient instructions, e.g. vectorization.
    - In your responses you will highlight tradeoffs with regards to performance and make suggestions improve adherence to DDD-principles.