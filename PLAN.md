# N-Body Simulator — Greenfield Rebuild Plan

Status: Living implementation guidance; completed slices are marked explicitly
Date: 2026-09-21

## 1. Purpose and product vision

This project will be rebuilt as a native Windows application for experimenting with, visualizing, and comparing implementations of the n-body problem.

The application has two equally important purposes:

1. It should be an enjoyable interactive simulator with a strong graphical presentation.
2. It should be a credible performance and numerical-comparison playground for different force-calculation implementations.

The long-term application should support:

- two- and three-dimensional Newtonian simulations;
- runtime selection of full pairwise or Barnes–Hut-style spatial approximation;
- runtime selection of scalar CPU, SIMD CPU, or Vulkan GPU computation;
- interactive body creation, inspection, and editing;
- configurable collision and boundary behavior;
- material colors, luminosity, lighting, spin, and axial tilt;
- native benchmark and telemetry facilities;
- large-scale scenarios using configurable units and numerical scaling;
- an experimental post-Newtonian physics model;
- visually convincing black holes, approximate absorption, and eventually approximate gravitational lensing.

The existing repository is not the foundation for this system. It is a source of historical ideas, equations, experiments, and benchmark intent. The rebuild should be greenfield, with only deliberately revalidated concepts carried over.

### General design preamble: unified physics engines

Each backend/solver-kind combination is an integrated physics engine, not merely a force solver. An engine owns the complete numerical step: force calculation, integration, collision detection and response, boundary handling, deferred physical outcomes, diagnostics, and any future physics modules added to that engine. Application code selects an engine through a common session-facing interface and does not need to know how that engine orders or fuses its internal phases.

The engine concept is an execution boundary, not a demand that every engine share the same algorithms or data structures. Scalar, SIMD, and Vulkan GPU engines should be designed from first principles around their respective execution models, memory systems, and synchronization costs. A SIMD engine may fuse phases or use layouts that have no useful Scalar equivalent; a GPU engine should use device-oriented flat buffers and kernel staging rather than mirror CPU object graphs. Existing implementations are semantic references and sources of validated behavior, not structural templates to be mechanically ported. Internally specialized engines must nevertheless be externally interchangeable: callers receive the same lifecycle and stepping API, configuration semantics, diagnostics, events, state views, validation behavior, and failure/transition reporting regardless of the selected engine.

The non-negotiable cross-engine contract is numerical and behavioral parity, especially against Scalar Full as the reference, within explicit tolerances. Parity includes positions, velocities, conserved quantities, body counts and identities, collision/fragmentation/absorption outcomes, and documented determinism guarantees. Tests must distinguish intended floating-point tolerance from genuine semantic divergence. Performance-oriented backend specialization is encouraged so long as it preserves that contract.

The engine boundary must also account for presentation handoff from the beginning. A completed simulation frame is published through a renderer-facing frame contract rather than by exposing mutable `WorldState` internals. CPU engines may publish read-only CPU views backed by reusable frame storage; a Vulkan engine should be able to publish device-local or shared Vulkan resources directly to the renderer, together with the synchronization information required for safe consumption. The common contract must not require GPU readback or an unnecessary CPU copy, while allowing the Scalar and SIMD implementations to use staging/upload paths appropriate to their memory domain.

## 2. Firm project constraints

### 2.1 Platform and toolchain

- The shipped application is a Windows application.
- The project is built with CMake.
- The primary compiler/toolchain is MSVC Build Tools.
- The build must work from the command line through CMake and Ninja.
- The Visual Studio IDE is not a project dependency.
- Vulkan GPU builds use the Vulkan SDK and supported shader toolchain.
- Vulkan is the settled native GPU API for both simulation compute and bespoke graphics.
- Windowing/input may use a platform library, but GPU rendering and compute should share one Vulkan device and resource model.
- The core simulation logic should remain portable C++ wherever practical.
- Windows-specific code should be isolated to application, windowing, file-system, or platform integration layers.

### 2.2 Runtime configuration

The following choices should be runtime-selectable where technically sensible:

- 2D versus 3D, during startup only;
- full pairwise versus Barnes–Hut approximation;
- scalar versus SIMD versus Vulkan GPU computation;
- collision behavior;
- boundary behavior;
- physics and numerical parameters;
- lighting and grid visualization;
- body and material properties.

The dimension cannot change once a simulation run has begun. Other solver choices may be changed at safe frame boundaries by pausing, transferring/repacking state, and reinitializing the selected backend.

### 2.3 Required solver implementations

The application should expose these six principal implementation choices:

| Force calculation | Scalar CPU | SIMD CPU | Vulkan GPU |
|---|---:|---:|---:|
| Full pairwise | Required | Required | Required |
| Barnes–Hut tree | Required | Required | Required |

The 2D and 3D versions should be supported by the same application architecture. The numerical kernels may be specialized internally for dimension and backend.

## 3. Guiding principles

### 3.1 Establish a trustworthy reference first

The scalar full pairwise solver is the initial correctness reference. All optimized implementations should be compared against it using identical initial states and parameters.

The project should not optimize a behavior that has not first been specified and tested.

### 3.2 Separate the simulation from the application

The simulation must be usable without creating a window. This enables:

- unit tests;
- deterministic regression tests;
- command-line benchmarks;
- automated backend comparisons;
- snapshot generation;
- Python-based offline analysis later.

### 3.3 Separate physical state from presentation state

Physics data, visual properties, selection, camera state, UI state, and benchmark state should not be mixed in one object.

### 3.4 Prefer policies over scattered conditionals

Collision handling, boundary responses, integration, force laws, units, and lighting should be explicit policy objects or enumerations. They should not be implemented as implicit behavior hidden in individual solver loops.

### 3.5 Preserve reproducibility

Every experiment should be reproducible from:

- a random seed;
- a serialized initial state;
- simulation parameters;
- solver configuration;
- a recorded sequence of edits and parameter changes.

### 3.6 Prefer data-oriented design in the numerical core

The numerical backend should be designed around data movement, cache behavior, vectorization, and device transfer efficiency. Human-readable object structures are useful at the UI and editing boundaries, but they are not the default representation for hot simulation loops.

In particular:

- hot kinematic data should be dense and contiguous;
- solver loops should operate on structure-of-arrays or similarly vector-friendly layouts;
- visual metadata should not be interleaved with positions, velocities, or masses unless measurements justify it;
- temporary force and acceleration buffers should be reused rather than allocated per step;
- storage should be organized to minimize cache misses, pointer chasing, and false sharing;
- GPU layouts should be designed for coalesced access rather than mirroring C++ object graphs;
- data should be reordered for tree traversal or device processing when that improves locality, with stable IDs preserving domain identity;
- allocations, ownership, and synchronization should be visible at subsystem boundaries.

This is a deliberate priority: the backend should optimize data and code arrangement even when that makes the internal representation less intuitive to a human reader.

The project should still maintain readable domain names, invariants, and documentation at the interfaces. Data-oriented design is not permission to make the domain model incomprehensible; it means that domain concepts and storage representation are allowed to be different things.

### 3.7 Apply DDD to boundaries and invariants, not to object shape

Domain-driven design should be used to define the language and boundaries of the system:

- simulation state owns physical state and simulation-time invariants;
- solver backends calculate forces and advance numerical state;
- collision and boundary systems own contact and world-constraint policies;
- editing owns validation and mutation commands;
- rendering owns presentation and visual interpretation;
- benchmarking owns measurements and experiment metadata.

These are domain boundaries and responsibilities, not demands for one class per noun. The numerical core may use flat arrays, indices, spans, views, and policies while still representing the domain faithfully.

Aggregates should be kept small and explicit. A body ID, world state, collision event, edit command, or benchmark result can be a domain concept without becoming a heap-allocated polymorphic object.

Where an invariant crosses a data-oriented layout, expose a named operation or validation function rather than relying on callers to manipulate raw arrays correctly.

## 4. Proposed high-level architecture

~~~text
Application
├── App state machine
├── Input and UI
├── Camera and interaction tools
├── Renderer
├── Benchmark/telemetry presentation
└── Simulation session
    ├── Editable world state
    ├── Physics and numerical parameters
    ├── Reference-point presentation model
    ├── Collision and boundary systems
    ├── Diagnostics
    └── Runtime-selected integrated physics engine
        ├── Scalar full
        ├── SIMD full
        ├── Vulkan GPU full
        ├── Scalar Barnes–Hut
        ├── SIMD Barnes–Hut
        └── Vulkan GPU Barnes–Hut
~~~

The renderer and UI communicate with the session through high-level commands and read-only views. The solver receives a simulation state and numerical configuration and returns an updated state or force/acceleration results.

### 4.1 Simulation-to-renderer frame publication

Simulation engines own mutable numerical state; renderers consume immutable published frames. The frame-publication boundary should support both CPU-backed and Vulkan-backed data without making the simulation core depend on Vulkan types.

The renderer-facing contract should provide, as applicable:

- positions and stable body IDs;
- body count, active/dead state, and compaction/remapping information;
- render-relevant physical and visual properties;
- simulation time and interpolation metadata;
- diagnostics and presentation-safe event summaries;
- a lifetime/lease guaranteeing that published data remains valid until rendering is complete;
- a synchronization token or equivalent readiness status.

The session should use reusable double- or triple-buffered frame storage so simulation and rendering can overlap without mutating data currently being consumed. A CPU engine may write a reusable host frame and arrange one upload into a Vulkan staging or persistently mapped buffer. A Vulkan engine using the same device should be able to export a renderer-consumable storage buffer or image directly, with explicit Vulkan barriers, queue ownership, and timeline-semaphore/fence requirements. The renderer must not need to know whether a frame originated from Scalar, SIMD, or GPU execution.

Dynamic body creation, fragmentation, absorption, and compaction must be reflected through stable IDs and frame metadata rather than assuming that a renderer can retain raw body-array indices across frames. GPU-capable paths should prefer capacity-managed buffers and append/compaction passes over synchronous readback. CPU-visible diagnostics and collision events may be transferred through a separate, compact asynchronous channel instead of forcing the complete numerical state back to the host.

This frame-publication design should precede further physics-module expansion. It establishes the ownership, lifetime, synchronization, and dynamic-cardinality rules that Scalar, SIMD, and Vulkan engines must all satisfy while preserving backend-specific internal layouts.

### 4.2 Frontend shell and presentation state

The frontend should be organized as a persistent application shell around the
active simulation state. The shell owns application mode, commands,
presentation state, panel layout, input routing, and transition animation;
the simulation session owns physical state and numerical progression.

The primary application flow is:

~~~text
Main Menu (`AppMode::MainMenu`)
  -> Simulation (`AppMode::Simulation`)
      -> Startup Edit (`SimulationMode::StartupEdit`)
      -> Paused / Running / Edit
  -> Benchmark (`AppMode::Benchmark`)
~~~

Initial Edit Mode starts in 2D by default. Its top bar contains the only
dimension-switch control. Switching between 2D and 3D is a destructive setup
operation: it discards the current setup completely, returns to an empty
Initial Edit Mode state in the selected dimension, and therefore requires
explicit confirmation. The control disappears when Initial Edit Mode ends and
is not available in normal Edit Mode during a running simulation. A total
reset/restart returns the session to Initial Edit Mode and makes the control
available again.

The simulation workspace should provide a consistent top bar, bottom bar,
left body/navigation panel, right inspector panel, and unobstructed viewport.
Each edge element is foldable. Top and bottom bars start expanded; left and
right panels start folded. A folded element leaves a small edge indicator.
Hovering the indicator for a configurable dwell time (initially about one
second) expands it with an eased transition. Folding and pinning should be
explicit so the UI does not unexpectedly disappear while being used.

The viewport rectangle must be derived from the animated panel bounds rather
than assuming that the renderer owns the complete window. Panel animation is
wall-clock UI state and must not affect simulation time.

~~~cpp
enum class AppMode { MainMenu, Simulation, Benchmark };

struct ApplicationState {
    AppMode mode;
    SimulationSession session;
    PresentationState presentation;
};

struct PresentationState {
    CameraState camera;
    CameraFocus focus;
    SelectionState selection;
    InspectorState inspector;
    GridSettings grid;
    UnitPreferences units;
    TrajectorySettings trajectories;
    ReferenceDisplaySettings references;
};

struct FoldablePanelState {
    PanelEdge edge;
    bool expanded;
    bool pinned;
    float animation_progress;
    float hover_time;
};
~~~

### 4.3 Inspector presentation and graphics boundary

Body inspection is one view model with multiple presentation modes. Selection
from the body list displays the inspector statically in the right panel.
Visual selection in the viewport may instead display an undocked inspector
anchored near the selected body and following it as it moves. The overlay must
be clamped to the viewport and may be pinned, dismissed, or returned to the
right panel. Both modes use the same commands and validation.

The current pixelated body renderer is a temporary presentation. The frontend
must publish render-relevant data through a graphics-ready scene/frame
boundary, not through widget-specific structures. That boundary should leave
room for:

- 2D and 3D camera projections;
- meshes or sprites instead of point circles;
- material and texture handles;
- per-body surface colors, luminosity, roughness, metallicity, and visual kind;
- ambient, unlit, emissive, and dynamic-lighting modes;
- depth, blending, post-processing, and debug overlays;
- GPU-resident body and material buffers where appropriate.

UI panels may request presentation changes, but renderer resources and Vulkan
objects remain inside rendering/application infrastructure. Simulation code
must only expose logical physical and visual properties.

### 4.4 Current architectural baseline

The first three architecture slices are implemented and are now constraints on
future work:

1. `ApplicationState` owns global `AppMode`, one `SimulationSession`, and one
   unified `PresentationState`. There is no separate `SimulationDocument` or
   `UIState` domain object.
2. `SimulationSession` owns `SimulationState`, the selected physics session,
   `SimulationMode`, and the optional in-memory initial-state checkpoint.
3. `CommandDispatcher` is the mutation boundary for lifecycle transitions,
   selection/focus, body creation, solver changes, display-unit preferences,
   checkpoint capture/restore, and snapshot load/save. The confirmed
   Initial-Edit-only dimension reset will be added through the same boundary.
4. `SnapshotSerializer` provides a versioned, human-readable exact physical
   snapshot format. Loading replaces the session and enters Startup Edit Mode;
   saving records the current editable/evolving state. A loaded snapshot is
   never a read-only scenario.
5. The renderer-facing frame contract now carries backend-neutral frame
   description, schema, storage classification, readiness, and resource
   lifetime semantics. CPU publication is implemented; target-specific GPU
   resources remain future backend work.
6. The logical render scene and render graph are now defined independently of
   graphics API calls. They describe camera, materials, overlays, resources,
   pass intent, and dependencies; target backends will compile them later.
7. The first Vulkan 2D renderer slice is implemented: shader compilation,
   Vulkan body pipeline, host-visible dynamic vertex storage, and CPU-frame
   body submission now render bodies through Vulkan before UI composition.

The current frontend is still a deliberately temporary shell. Its existing
widgets and renderer are not architectural commitments. Future UI work should
bind controls to commands and read models rather than adding direct mutation
paths around the dispatcher.

### 4.5 Shared workflows versus dimension-specific presentation

The application should have one shared workflow layer for commands, text
input, validation, snapshots, body lists, inspectors, simulation lifecycle,
and numerical editing. Those features must work regardless of whether the
current session is 2D or 3D.

Visual spatial affordances are different products in the two dimensions and
must be implemented and tested separately:

- 2D mouse picking, screen-to-plane mapping, grid lines, orbit overlays,
  trajectory presentation, and 2D camera behavior;
- 3D ray picking, depth-aware selection, plane/grid rendering, orbit-plane
  visualization, trajectory presentation, and 3D camera behavior.

They may share mathematical models, command types, read-only view models, and
renderer resource policies, but neither dimension should be treated as a
thin special case of the other. Text-based creation/editing is the universal
fallback and remains available in both modes.

### 4.6 Separate native and web products

Vulkan and WebGPU are never mixed in one binary. They are separate application
targets with separate renderer and GPU-simulation implementations:

~~~text
Native desktop binary
  portable simulation/domain code
    ├── Scalar CPU backend
    ├── SIMD CPU backend
    └── Vulkan GPU backend
  Vulkan renderer

Web application binary
  portable simulation/domain code compiled to WebAssembly
    ├── Scalar CPU backend
    ├── WebAssembly SIMD backend where supported
    └── WebGPU compute backend
  WebGPU renderer
~~~

The shared boundary is limited to simulation state schemas, published frame
semantics, logical render-scene data, commands, snapshots, and conformance
tests. Vulkan objects, WebGPU objects, synchronization mechanisms, shader
implementations, resource allocators, and device capability handling remain in
their respective targets.

Each target may use an integrated zero-copy simulation-to-renderer path when
both systems run on the same device/API. CPU simulation uses the same logical
frame contract but uploads through that target's renderer. No cross-API
resource-sharing abstraction is required.

## 5. Suggested source layout

This is deliberately provisional. Names and file boundaries may change during implementation.

~~~text
src/
  app/
    app_state.*
    simulation_session.*
    commands.*
    command_history.*

  simulation/
    world_state.*
    body_state.*
    body_id.*
    simulation_parameters.*
    unit_system.*
    integrator.*
    solver.hpp
    solver_factory.*
    solver_info.*
    diagnostics.*
    reference_frame.*

    solvers/
      scalar_full.*
      scalar_barnes_hut.*
      simd_full.*
      simd_barnes_hut.*
      vulkan_gpu_full.*
      vulkan_gpu_barnes_hut.*
      vulkan_compute_support.*

    spatial/
      quadtree.*
      octree.*
      flat_tree.*
      tree_builder.*

    physics/
      force_model.*
      newtonian.*
      post_newtonian.*
      collision_system.*
      boundary_system.*
      collision_events.*
      black_hole.*
      rotation.*

  rendering/
    renderer.hpp
    renderer_2d.*
    renderer_3d.*
    render_scene.*
    render_materials.*
    render_resources.*
    camera.*
    selection.*
    lighting.*
    materials.*
    debug_visualization.*
    grid_visualization.*
    trajectory_visualization.*

  ui/
    ui_context.*
    panel_layout.*
    panel_animation.*
    hud.*
    main_menu.*
    startup_editor.*
    edit_mode.*
    body_list.*
    body_inspector.*
    creation_dialog.*
    body_generators.*
    physics_panel.*
    benchmark_panel.*
    unit_governance.*
    notification.*

  io/
    snapshot.*
    snapshot_format.*
    configuration.*
    export.*

tests/
  simulation_tests.*
  solver_conformance_tests.*
  collision_tests.*
  tree_tests.*
  serialization_tests.*

tools/
  benchmark_main.*
  snapshot_inspector.*
~~~

The exact UI library remains an implementation choice, but the native renderer is Vulkan-based, with a windowing/input library and optional Dear ImGui integration wrapped behind project-owned interfaces. The renderer layer should not expose Vulkan, windowing, or UI-library types to simulation code.

## 6. World state and data model

### 6.1 Stable body identity

Bodies need stable identifiers independent of their storage index.

Roughly:

~~~cpp
struct BodyId {
    std::uint64_t value;
};
~~~

Storage may be reordered for SIMD, tree traversal, or GPU execution, but body IDs must remain stable for:

- selection;
- body lists;
- property windows;
- collision events;
- benchmark logs;
- undo/redo;
- snapshot references.

### 6.2 Vectors and dimensions

Use a common three-component vector representation at the application/state level:

~~~cpp
struct Vec3 {
    double x;
    double y;
    double z;
};
~~~

In 2D mode, z and z velocity are constrained to zero. Numerical kernels may still have template dimension specializations so that 2D kernels avoid unnecessary z arithmetic.

This avoids duplicating the whole application, UI, serialization, and renderer architecture while permitting genuinely different 2D and 3D kernels.

### 6.3 Body physical state

Rough draft:

~~~cpp
struct Kinematics {
    Vec3 position;
    Vec3 velocity;
};

struct RotationState {
    Quaternion orientation;
    Vec3 angular_velocity;
};

struct PhysicalProperties {
    double mass;
    double radius;
    bool is_static;
};

struct BodyState {
    BodyId id;
    Kinematics kinematics;
    RotationState rotation;
    PhysicalProperties physical;
    Appearance appearance;
    BodyKind kind;
};
~~~

The precise representation may become structure-of-arrays for performance. The important conceptual separation is between stable identity, kinematics, physical properties, rotation, and appearance.

### 6.4 Structure-of-arrays storage

The editable model may use a convenient body-oriented view, while solver storage should likely be structure-of-arrays:

~~~cpp
struct BodyStorage {
    std::vector<BodyId> ids;
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> mass;
    std::vector<double> radius;
    std::vector<std::uint8_t> static_flags;
    std::vector<Appearance> appearance;
    std::vector<RotationState> rotation;
};
~~~

The exact split between physical and visual arrays can be optimized later. SIMD and Vulkan GPU paths should not force the rest of the application to use backend-specific layouts.

The storage should be explicitly split into hot, warm, and cold data rather than treating every body field as equally important to every loop.

~~~text
Hot simulation data:
  position, velocity, mass, radius, static mask

Warm simulation data:
  accumulated force/acceleration, collision state, solver reorder indices

Cold/editing data:
  stable ID, names, three colors, material preset, luminosity,
  body kind, UI metadata, selection state, editor locks

Rotation/render data:
  orientation, angular velocity, debug markers, renderer handles
~~~

The exact grouping should be validated with profiling, but the intended direction is that a full gravity step does not pull appearance, strings, UI locks, or renderer handles into cache lines.

Potential implementation techniques include:

- dense arrays for hot fields;
- sparse-set or ID-to-dense-index mapping for stable identity;
- separate cold metadata tables keyed by BodyId;
- reusable double-buffered force/velocity storage;
- aligned allocations for SIMD data;
- explicit padding or chunking to avoid false sharing in threaded paths;
- backend-specific views over shared logical state;
- compact integer masks instead of scattered boolean objects;
- explicit permutation maps when tree/GPU layouts reorder bodies.

The editable world may offer a body-oriented façade, but that façade should be a view or command boundary over the dense storage rather than the storage used by the numerical loops.

### 6.5 Data ownership and mutation phases

Simulation stepping should have explicit phases so that storage remains stable while hot loops run:

~~~text
Read current state
    → calculate forces/accelerations
    → integrate into next-state buffers
    → resolve collisions and boundaries
    → apply deferred body mutations
    → commit/reorder state
    → publish a read-only render snapshot
~~~

Body creation, deletion, fragmentation, absorption, and reordering should not invalidate iterators or references during force calculation. The publish step should provide the renderer with a stable snapshot or read-only view, avoiding UI/render code racing with backend writes.

### 6.6 Body kinds

Initially, a body can be an ordinary body with appearance/material settings. The model should leave room for:

~~~cpp
enum class BodyKind {
    Ordinary,
    BlackHole
};
~~~

Future kinds such as gas body, star, or solid planet should preferably be material/physics property combinations rather than an explosion of subclasses.

The simulation may use a compact `BodyKind` classification for behavior that cannot
be inferred from material values alone. Ordinary bodies use material presets for
rocky, metallic, or icy behavior; gas bodies and stars can opt into accretion-like
absorption behavior; black holes are a separate protected kind. These kinds are
metadata, not subclasses, and should remain separate from editable material
properties.

## 7. Appearance, colors, and lighting

### 7.1 Three-color palette

Every body should have three colors from the beginning:

~~~cpp
struct Color {
    float r;
    float g;
    float b;
    float a;
};

struct Appearance {
    Color primary_color;
    Color secondary_color;
    Color tertiary_color;
    float luminosity;
    Color light_color;
    MaterialPreset material;
};
~~~

Initial behavior:

- primary_color colors both luminous and non-luminous bodies;
- primary_color determines emitted light color for luminous bodies;
- secondary and tertiary colors are editable, saved, and displayed but initially unused by the surface shader;
- luminosity zero means non-luminous.

Later material shaders should mix all three colors to produce rock, gas, and star surfaces.

There is no requirement to model multicolored stellar emission initially. A luminous body emits using its primary color.

### 7.2 Lighting activation

Lighting should be optional.

Suggested behavior:

- with no synthetic or luminous sources, use an ambient/unlit presentation so the scene never becomes pitch black;
- when the first light source or luminous body is introduced, dynamic lighting may activate automatically;
- the user can override this automatic behavior with a lighting mode setting.

Rough policy:

~~~cpp
enum class LightingMode {
    AmbientOnly,
    Dynamic,
    Automatic
};
~~~

### 7.3 Light sources

Light sources are visual objects, not gravitational bodies unless explicitly represented by a body.

Roughly:

~~~cpp
struct LightSource {
    LightId id;
    LightKind kind;       // point initially; directional/area later
    Vec3 position;
    Vec3 direction;
    Color color;
    float intensity;
    bool enabled;
};
~~~

Users may:

- create unsimulated synthetic lights;
- make one or more bodies luminous;
- edit luminosity and light color;
- toggle lighting and light visibility.

### 7.4 Material progression

The material system should progress in layers:

1. flat primary color;
2. primary-color lighting;
3. three-color debug pattern;
4. rock, gas, and star procedural presets;
5. configurable shader parameters;
6. richer textures and generated surface detail.

The procedural material system should use body-local coordinates so that the pattern rotates with the body.

## 8. Physics and numerical configuration

### 8.1 Physics model

The session should expose a physics-model choice:

~~~cpp
enum class PhysicsModel {
    Newtonian,
    PostNewtonian
};
~~~

Newtonian gravity is the initial reference. Post-Newtonian corrections should be added later as a separate model implementation that can reuse the body-centric architecture.

### 8.2 Simulation parameters

Rough draft:

~~~cpp
struct SimulationParameters {
    PhysicsModel physics_model;
    double gravitational_constant;
    double timestep;
    double display_timescale;
    IntegrationMethod integrator;
    CollisionSettings collision;
    BoundarySettings boundary;
    NumericalSettings numerical;
};
~~~

The distinction between simulation timestep and display timescale is important:

- timestep controls numerical integration;
- display timescale controls how quickly simulation time advances relative to wall-clock time;
- a paused simulation has display timescale zero;
- changing display timescale should not change the numerical timestep.

### 8.3 Integration

The current half-step update should not be carried over without validation. The initial candidate should be velocity Verlet or leapfrog, with a clearly documented convention.

The integrator should be a replaceable policy so that future comparisons can distinguish:

- force-backend performance;
- integration-method behavior;
- numerical stability.

### 8.4 Numerical settings

Potential settings include:

- scalar precision;
- force-softening length;
- Barnes–Hut tolerance;
- maximum tree depth;
- maximum bodies;
- timestep policy;
- Vulkan GPU workgroup configuration, where exposing it is useful;
- diagnostic frequency.

Numeric settings should be editable in Edit Mode and recorded as parameter-change events.

## 9. Collision system

Collision behavior is a runtime UI-accessible setting.

### 9.1 Collision models

~~~cpp
enum class CollisionModel {
    Transparent,
    HardBody
};
~~~

Fragmentation and absorption are hard-body outcome settings, not collision
models. They are only valid when `CollisionModel::HardBody` is selected.

### 9.2 Transparent bodies

Bodies glide through each other. The singularity handler must be explicit.

Possible initial implementation:

~~~text
effective_distance² = distance² + softening_length²
~~~

or a finite force cap when separation approaches zero.

The softening length should be editable and visible in the numerical settings.

Transparent mode should not silently use hard-body collision tests.

### 9.2.1 Runtime collision-mode transitions

Collision classifier settings, including damage, fragmentation, and absorption policy, are
valid only while hard-body collision mode is active. Transparent mode must not
silently classify contacts, generate fragments, or absorb bodies; attempting to
enable any of these settings while transparent mode is selected must be
rejected as an invalid runtime configuration.

Fragmentation is an outcome layered on top of hard-body contact resolution. It
is not a separate contact solver: a fragmenting contact still performs
hard-body overlap correction, impact calculation, impulse response, and event
classification before deferred body mutation.

When switching from transparent to hard-body mode at a runtime-safe boundary,
the collision system must first check for overlapping bodies. Overlaps are
resolved by moving the bodies apart along their contact axis until they merely
touch, using inverse-mass weighting and treating static bodies as immovable.
The correction must run in multiple passes because separating one pair can
create or expose another overlap. The pass count is capped at 50. If overlaps
remain after the cap, the transition fails: the world remains in transparent
mode, and diagnostics must expose the failed pass count and the IDs of bodies
still involved in overlaps so the frontend can highlight them.

The transition check must be dimension-aware. In 2D it operates in X/Y and
keeps Z constrained to zero. In 3D it operates in X/Y/Z without flattening
positions or velocities.

### 9.3 Hard bodies

Initial hard-body behavior should resemble billiard balls:

- detect contact/overlap;
- resolve penetration;
- calculate the collision normal;
- apply an impulse;
- support unequal masses;
- support both 2D and 3D;
- support configurable restitution;
- treat static bodies as infinite-mass objects.

The collision system should be shared across solver backends wherever possible. The force solver should calculate forces; a common post-integration collision phase should resolve hard contacts.

Continuous collision detection may be needed later when bodies move far enough in one timestep to tunnel through one another. The first implementation may use discrete overlap detection, provided diagnostics make the limitation visible.

### 9.4 Future heuristic outcomes

Future collision outcomes may include:

- solid + solid: merge, partial fragmentation, or a configurable hard-body outcome;
- small solid + large body: large body gains most mass and loses a small amount as ejecta;
- solid + gas/star body: the solid is absorbed when it is below both configurable
  mass and size ratios relative to the absorber; otherwise the solid destroys the
  absorber and receives a momentum-weighted velocity slowdown while retaining its
  own mass and radius. The default mass and size ratios are both 2.0 and are
  independently exposed as collision settings;
- gas/star absorbers are immune to damage and fragmentation. Absorber-versus-
  absorber selection compares mass first; near-equal masses use size as the
  tie-breaker, followed by body-array index;
- black holes absorb solid bodies regardless of the solid-dominance thresholds,
  defeat every non-black-hole absorber, and participate in ordinary absorber
  comparisons against other black holes;
- fragmentation: a body splits into a deterministic or seeded count between configurable minimum and maximum fragment counts (defaulting to 2–5), subject to a configurable per-step fragment cap; total mass approximately equals the original;
- conservation of mass and approximate conservation of momentum.

Fragmentation geometry must be dimension-correct: 2D fragments use balanced
planar directions, while 3D fragments use balanced XYZ directions. Fragment
position and velocity offsets must preserve the parent center of mass and
linear momentum within numerical tolerance.

Body creation/removal must occur in a deferred mutation phase. A solver or collision loop must never invalidate its own iteration by directly resizing body storage.

### 9.5 Explicit material properties and presets

Bodies should expose their material-response values directly rather than deriving all collision behavior from density and size. Density remains a physical body property linked to mass and radius by the spherical-body relationship, while the following material values are independently editable:

~~~cpp
enum class MaterialPreset {
    Custom,
    Rocky,
    Metallic,
    Icy,
    Gas
};

struct MaterialProperties {
    MaterialPreset preset;
    double density;                  // kg/m^3
    double compressive_strength;     // Pa
    double tensile_strength;         // Pa
    double brittleness;              // normalized 0..1
    double energy_absorption;        // normalized 0..1
    double restitution;              // normalized 0..1
    double damage_threshold;         // J/kg
    double fragmentation_threshold;  // J/kg
};
~~~

The initial presets should provide editable starting values for Rocky, Metallic, Icy, and Gas bodies. Applying a preset copies its values into the body; subsequent edits make the body Custom or otherwise visibly modified. Presets are convenience defaults, not hidden constraints.

Hard-body collisions should use the material restitution together with the global collision cap. Heuristic damage and fragmentation should compare pre-impulse specific impact energy against each body's damage and fragmentation thresholds, while considering compressive/tensile strength, brittleness, energy absorption, density, and gravitational binding. The model is intentionally approximate but all inputs must remain inspectable and serializable.

Material values must be preserved through snapshots, deterministic replay, solver switching, body duplication, and deferred collision mutations. A collision classifier should be separate from contact resolution so an ordinary hard-body bounce still evaluates impact severity without necessarily causing damage or fragmentation.

## 10. Boundary system

Boundaries are optional external world constraints, not simulated bodies.

### 10.1 Boundary settings

~~~cpp
struct BoundarySettings {
    bool enabled;
    BoundaryShape shape;        // box initially; sphere/planes later
    BoundaryResponse response;  // reflection or repulsive potential
    double restitution;
    double repulsion_strength;
    double repulsion_falloff;
};
~~~

The first UI should expose a simple axis-aligned box. The architecture should leave room for:

- a sphere;
- individual coordinate-plane thresholds;
- arbitrary planes;
- a user-defined angled plane later.

### 10.2 Reflective boundaries

This is the recommended first boundary response.

When a body reaches a boundary:

- clamp it to the valid side;
- reflect the normal velocity component;
- apply restitution;
- preserve tangential velocity unless another policy changes it.

For a collision normal n, a basic response is conceptually:

~~~text
v_new = v - (1 + restitution) * dot(v, n) * n
~~~

### 10.3 Repulsive boundaries

An impossibly strong gravitational push is allowed as a deliberately non-physical visualization mode.

It must be clearly labeled because it can:

- inject energy;
- inject momentum;
- trap bodies near boundaries;
- change the long-term evolution in nonphysical ways.

It should be excluded from scientific benchmarks or marked explicitly in benchmark metadata.

### 10.4 Unbounded simulations

With boundaries disabled, the numerical workspace is conceptually unbounded, limited by:

- numeric representation;
- integration stability;
- memory;
- renderer precision;
- practical runtime.

The UI should display warnings when bodies leave the visible region or approach numeric limits.

## 11. Units, scale, and large distances

The application should not assume one fixed unit system.

### 11.1 Unit system

~~~cpp
struct UnitSystem {
    double length_scale;
    double mass_scale;
    double time_scale;
    std::string length_name;
    std::string mass_name;
    std::string time_name;
};
~~~

Candidate presets:

- SI metres/kilograms/seconds;
- kilometres;
- astronomical units and years;
- custom normalized units.

Unit conversion and formatting must be governed by one presentation-facing
unit policy. It owns time display selectors, automatic km/AU/light-year
distance selection, precision rules, labels, and simulation/snapshot scale
metadata. Widgets must not carry independent conversion constants.

The equations should operate consistently in the selected units. Internally normalized values may be used to keep magnitudes near one.

### 11.2 Precision

The reference state should use double for positions, velocities, masses, and diagnostic accumulators. GPU storage may later use float where the precision tradeoff is understood and tested.

### 11.3 Origin rebasing

The application should support shifting the numerical origin when coordinates become inconveniently large:

~~~text
select reference position
→ subtract it from all positions
→ preserve relative positions and velocities
→ update the coordinate-frame metadata
~~~

This must be a coordinate transformation, not a physical impulse.

The selected global reference point may eventually be used as the rebasing target, but display reference and numerical rebasing should remain conceptually distinct.

### 11.4 Solar-System and Alpha-Centauri scale

Solar-System scenarios should be a realistic early target. A Sun, planets, and Alpha Centauri-scale separation is a desirable advanced target.

The main difficulty is not merely storing the distance. It is resolving:

- very small planetary orbital scales;
- very large interstellar separation;
- widely different dynamical timescales;
- a useful timestep for both.

The first implementation may use one global timestep with diagnostics. Later options include:

- adaptive timesteps;
- hierarchical timesteps;
- subsystem/barycentric integration;
- multiple coordinate frames;
- specialized distant-object approximations.

The project should not claim accurate Alpha-Centauri-scale dynamics until those timestep and error issues are measured.

## 12. Rotation, spin, and axial tilt

Spin and axial tilt are first-class body properties.

### 12.1 Rotational state

Rough draft:

~~~cpp
struct RotationState {
    Quaternion orientation;
    Vec3 angular_velocity;
    bool rotation_enabled;
};
~~~

The first implementation need not calculate torques or collision-induced spin changes, but it must preserve, display, edit, and animate orientation and spin.

Later physical features may add:

- moment of inertia;
- angular momentum;
- torque;
- tidal locking;
- spin transfer during collisions;
- non-spherical moments of inertia.

### 12.2 Axial tilt

Axial tilt should be represented by the body orientation/spin axis relative to the selected world/reference frame. It should not be stored as an isolated angle if a full orientation is already available.

The UI may present the orientation as:

- axial tilt;
- azimuth;
- spin period or angular speed;
- editable quaternion/vector values in an advanced panel.

### 12.3 Early visual verification

The renderer should expose a rotation debug mode from the beginning:

- spin-axis arrow;
- equatorial ring;
- north/south pole markers;
- a prime-meridian marker;
- optional angular-velocity vector.

The body-local debug surface should use the three configured colors in a clearly identifiable pattern, such as:

- latitude bands;
- an equatorial stripe;
- a prime-meridian stripe;
- contrasting polar caps.

The pattern must be attached to body-local coordinates. It must rotate with the body and reveal axial tilt, spin direction, and spin rate.

## 13. Reference points and body details

### 13.1 Global reference

The user can choose a reference point by:

- selecting a body;
- clicking a body;
- specifying an arbitrary coordinate;
- reverting to the mathematical origin.

Rough model:

~~~cpp
struct ReferenceFrame {
    enum class Kind { Origin, Body, Coordinate } kind;
    BodyId body;
    Vec3 coordinate;
};
~~~

The default reference is (0, 0, 0) with zero reference velocity.

The reference affects displayed values only. Internal force calculations proceed in the simulation’s own coordinate system.

### 13.2 Relative values

For a body and reference with positions r and R:

~~~text
relative_position = r - R
relative_velocity = v - V
distance = |relative_position|
~~~

For a body pair A/B, the details card should show:

- relative position vector;
- relative velocity vector;
- distance;
- gravitational force;
- optionally acceleration and potential energy.

### 13.3 Angular velocity terminology

The UI must distinguish:

- intrinsic spin/angular velocity of the body;
- instantaneous orbital angular velocity around the selected reference.

For orbital motion, a useful derived value is conceptually:

~~~text
omega_orbital = (r × v) / |r|²
~~~

This is a derived display quantity, not automatically body spin.

### 13.4 Body-to-body inspection

The body details window should allow choosing a comparison body through:

- a searchable/listed body selector;
- clicking another body in the simulator;
- possibly a temporary comparison-selection mode.

Changing the comparison body should not change the globally selected reference unless explicitly requested.

### 13.5 Recent trajectory trails

The renderer should optionally display a recent trajectory trail for each body. A trail is a visual history of the body's movement, not additional physical state and not a replacement for saved simulation snapshots.

The feature should provide:

- a global enable/disable setting, with an optional per-body visibility override;
- a configurable history duration or sample count;
- a configurable sampling/decimation policy so large simulations do not allocate unbounded history;
- a line or ribbon following the recent path;
- a fade toward the oldest end of the trail, ending transparently;
- sensible handling of body creation, deletion, teleportation, origin rebasing, and solver switching;
- a clear indication when the trail has been invalidated or restarted after a discontinuity.

Trajectory history belongs to presentation/diagnostic state. It should be maintained in a bounded ring buffer keyed by stable `BodyId` and should not affect force calculations, integration, or benchmark correctness.

### 13.6 Stable and periodic orbit visualization

The application should optionally assess whether a body's recent trajectory is sufficiently bounded, stable, and periodic to justify displaying an estimated orbit. This is an analysis feature, not a claim that the underlying motion is an exact Keplerian orbit.

For a selected body and reference, the analysis may use a rolling observation window and evaluate:

- whether the body remains bounded relative to the reference;
- whether orbital energy, angular momentum, and orbital-plane orientation remain sufficiently consistent;
- whether the trajectory returns near a prior position/velocity phase after a plausible period;
- whether the fitted orbital parameters remain stable across multiple windows;
- whether perturbations from other bodies make the estimate too uncertain.

When confidence is high enough, the renderer may show a predicted or fitted orbit, such as an ellipse in the estimated orbital plane, alongside the measured recent trail. The display should distinguish measured path from estimated continuation and should show no orbit, or mark it as uncertain, when the evidence is insufficient.

The analysis must tolerate non-Keplerian but periodic motion where practical, while clearly labeling approximations. It should expose thresholds, confidence, observation-window details, and invalidation conditions for diagnostics and reproducible comparisons. Orbit assessment must be bounded in cost and must not run inside the hot force-calculation loop.

## 14. Body editing and linked properties

### 14.1 Body inspector and presentation modes

Selecting a body in the body list displays a static inspector in the right
panel. Visual selection in the simulation may instead display an undocked
inspector anchored near the selected body and following it as it moves. Both
presentations use the same read-only view model, validation, and edit
commands. The overlay must be clamped to the viewport and may be pinned,
dismissed, or returned to the right panel.

Double-clicking a body or selecting its list entry opens an inspector containing:

- ID/name, later if names are added;
- position;
- velocity;
- mass;
- radius;
- density;
- static flag;
- material preset;
- compressive strength;
- tensile strength;
- brittleness;
- energy absorption;
- material restitution;
- damage threshold;
- fragmentation threshold;
- surface colors 1–3;
- luminosity strength;
- luminosity/light color;
- material preset;
- orientation/axial tilt;
- spin/angular velocity;
- optional black-hole properties;
- relative values to the global reference;
- optional relative comparison to another body.

### 14.2 Linked mass, radius, and density

For a spherical body:

~~~text
volume = 4/3 π r³
density = mass / volume
~~~

By default, mass, radius, and density are linked. Editing one should immediately update the others according to the active editing rule.

The UI should allow locking individual quantities with a lock icon.

If a user edits one property and that edit would require changing a locked property:

1. reject the edit;
2. restore the edited field to its previous value;
3. temporarily highlight the locked field that caused the rejection;
4. show a short explanatory notification if useful.

This should be implemented through a validation/calculation layer, not UI-specific arithmetic.

An eventual editing mode may let the user choose which two of the three values are independent. The default should remain the fully linked behavior described above.

### 14.3 Commands and undoability

Edits should be represented as commands:

~~~cpp
struct EditCommand {
    BodyId body;
    PropertyId property;
    Value old_value;
    Value new_value;
};
~~~

This enables future undo/redo, validation, event recording, and consistent updates from both text fields and camera tools.

## 15. Body creation and initial velocities

### 15.1 Text creation dialog

The text-based creation dialog should support:

- position x/y/z;
- initial velocity x/y/z;
- mass;
- radius;
- density;
- static flag;
- three surface colors;
- luminosity and light color;
- material preset;
- compressive strength, tensile strength, brittleness, and energy absorption;
- material restitution, damage threshold, and fragmentation threshold;
- orientation and axial tilt;
- spin/angular velocity;
- body kind, eventually including black hole.

In 2D mode, z-related controls should be hidden or disabled.

### 15.2 Camera-based placement

The camera-based tool should use a placement state machine:

~~~text
Choose world position
    → choose initial velocity vector
    → confirm body properties
    → create body
~~~

In 2D, a screen click maps directly to a world-plane position.

In 3D:

1. camera temporarily snaps to a top-down XY view;
2. user selects x/y position;
3. camera changes to an angled placement view;
4. mouse movement selects z offset;
5. a velocity vector is drawn from the body surface toward the cursor;
6. the vector length is converted from screen interaction to world velocity using camera scale/perspective;
7. user confirms or cancels.

The text creation dialog must remain available as an exact-input fallback.

### 15.3 Body generators and snapshots

The core persistence concept is an exact simulation snapshot, not a separate
scenario domain object. A snapshot contains the complete physical state and
its governing configuration:

~~~cpp
struct SimulationSnapshot {
    std::uint32_t format_version;
    SimulationState state;
    SnapshotMetadata metadata;
};
~~~

Snapshots can be loaded, edited in Startup or normal Edit Mode, and saved
again. They are useful both as checkpoints for resuming a run and as exact
inputs for tests and benchmarks. The canonical snapshot is physical state plus
the parameters that govern its evolution; presentation state remains separate
and is not required to make a simulation reproducible. A future optional
presentation/layout sidecar must not change the meaning of the physical
snapshot.

Body generators are Edit Mode tools that write into the current simulation
state. They
may provide configurable recipes and deterministic seeds, including:

- random clouds and distributions;
- spirals and galaxy-like arrangements;
- two-body and orbit setups;
- a solar-system-like arrangement with scale, epoch/phase, and static-Sun
  options;
- future custom or imported arrangements.

The generated body state is authoritative. Generator name, parameters, and
seed may be retained as optional metadata or command history for reproducible
editing, but they are not required to reconstruct a loaded snapshot.

The first implementation uses a versioned human-readable tagged format. The
serializer validates dimensions, finite numerical values, solver
configuration, world validity, and format version before a loaded state is
accepted. `SaveSnapshot` and `LoadSnapshot` commands are the application-facing
entry points; file dialogs, recent-file UI, and user-facing error notifications
are presentation work rather than new persistence concepts.

Each active session should retain an in-memory t=0 snapshot. Entering the
simulation creates or replaces this checkpoint after Startup Edit Mode. A
restart/reset command restores it exactly, while later edits in Edit Mode may
explicitly replace the checkpoint. This gives users a low-cost recovery path
without confusing the checkpoint with the current evolving state. The
checkpoint uses the same state representation as a serialized snapshot, but it
does not need filesystem I/O.

## 16. Application flow and modes

### 16.1 Startup flow

The main menu should offer either a new empty simulation or loading an exact
snapshot before entering Startup Edit Mode. Loading a snapshot does not make
it read-only: it becomes the current editable simulation state.

The required high-level flow is:

~~~text
Main Menu
    → Start Simulation
    → Startup Edit Mode
    → End Startup Edit Mode
    → Simulation Running
~~~

Startup Edit Mode behaves like normal Edit Mode except that it allows:

- choosing 2D or 3D through the top-bar dimension switch, with 2D as the
  default;
- selecting the initial solver/backend;
- setting initial physics and numerical parameters;
- placing and editing bodies;
- invoking body generators, including a configurable solar-system-like tool;
- loading, editing, and saving exact snapshots;
- configuring boundaries, lighting, grids, and presentation.

Once startup ends and the simulation begins, dimension is immutable for that run.

The dimension switch must not silently convert, preserve, or partially migrate
the current setup. After confirmation, the world, bodies, simulation time,
checkpoint, and setup-specific state are reset to an empty Initial Edit Mode
state in the selected dimension. Normal Edit Mode deliberately has no
dimension-switch control. Only the total-reset command can return to Initial
Edit Mode and re-enable it.

The frontend should not require a dedicated Solar-System scenario type. Solar
systems, debug arrangements, and test setups are ordinary generator tools or
snapshot files presented through the same workflow.

### 16.2 Normal Edit Mode

Normal Edit Mode is entered from a running or paused simulation.

It allows:

- body selection and editing;
- body creation/removal;
- physics/numerical setting edits;
- solver switching at safe frame boundaries;
- collision/boundary settings;
- reference-point changes;
- lighting and visualization changes;
- benchmark/telemetry controls.

### 16.3 Entering Edit Mode

If the simulation is running, entering Edit Mode causes the display timescale to decrease linearly from its current value to zero over one second.

If the simulation is already fully paused in Non-Edit mode, Edit Mode is entered immediately without the slowdown animation.

Edit Mode itself should always have simulation timescale zero. The transition should be wall-clock driven and should preserve the timescale that was active before entering Edit Mode.

### 16.4 Leaving Edit Mode

Attempting to unpause in Edit Mode must ask for confirmation:

> Leaving Edit Mode will end editing and resume the simulation. Continue?

If confirmed:

- Edit Mode ends;
- if the simulation was running before Edit Mode, the display timescale ramps linearly from zero to the pre-edit value over one second;
- if the simulation was paused before Edit Mode, the transition is immediate and the simulation remains paused;
- any edits are committed before the transition completes.

The UI must make the distinction between resume simulation and leave Edit Mode while staying paused clear if both actions are offered.

### 16.5 State ownership and transitions

~~~cpp
enum class AppMode {
    MainMenu,
    Simulation,
    Benchmark
};

enum class SimulationMode {
    StartupEdit,
    Running,
    Paused,
    Edit,
    EnteringEdit,
    LeavingEdit
};
~~~

`AppMode` answers which major application workspace is active. `SimulationMode`
answers what the simulation session is doing inside the Simulation workspace.
Panel folding, hover dwell, and transition animation progress belong to
`PresentationState`; they must not become additional global application modes.
The implementation already has these two levels, with the transition modes
reserved for the smooth Edit Mode and panel transitions still to be wired into
the UI.

## 17. HUD and quality-of-life features

The rudimentary HUD should be available during simulation and optionally
hideable. The full HUD is composed of foldable edge elements rather than one
monolithic window. Top and bottom elements are expanded by default; left and
right navigation/inspector panels are folded by default.

When folded, each element leaves a small edge indicator. Hovering it for a
configurable dwell time, initially about one second, expands the element with
an eased animation. The panel layout computes the remaining viewport bounds
continuously during the transition. Folding, unfolding, pinning, and hover
dwell are UI state and must be independent of simulation time.

Initial HUD contents:

- timescale/simulation speed control, likely a slider;
- simulated elapsed time with a nearby governed unit selector (seconds,
  days, months, or years);
- grid-distance indicator with automatic km/AU/light-year selection based on
  zoom level, plus an explicit override where useful;
- render FPS;
- simulation step duration;
- body count;
- selected dimension;
- force model;
- compute backend;
- physics model;
- collision mode;
- boundary status;
- lighting status;
- button to enter Edit Mode;
- benchmark/telemetry access.

The HUD should display the actual active solver, not merely the requested setting.
Unit conversion and automatic distance thresholds must come from one shared
unit-governance policy, not from formatting conditionals scattered through
individual widgets.

## 18. Grids, reference planes, and orrery presentation

### 18.1 Initial grids

The user can toggle a locating grid.

In 2D:

- one grid corresponding to the simulation plane.

In 3D:

- XY grid;
- XZ grid;
- YZ grid;
- independently toggleable visibility if practical.

### 18.2 Reference-plane indicators

The user can choose a base plane. Initially this can be any coordinate-axis plane.

Bodies may show perpendicular indicators from the plane to their positions, producing an orrery/mechanical-model presentation.

### 18.3 Arbitrary and dynamically calculated planes

Later features may support:

- user-entered plane orientation;
- direct angle input;
- an arbitrary plane defined by a point and normal;
- an average orbital plane calculated from all bodies;
- an average orbital plane calculated from selected bodies.

The orbital-plane candidate can be derived from aggregate angular momentum:

~~~text
L = Σ (r × p)
~~~

The UI should describe this as an approximation when the system does not have a single stable orbital plane.

### 18.4 Trajectory and orbit overlays

Trajectory trails and orbit overlays should be independently toggleable from the grid and reference-plane presentation. A trail shows where the body has recently been; an orbit overlay shows an assessed, fitted, or predicted path and must not be presented as measured data.

The overlay system should support:

- recent-path trails with age-based fading;
- selected-body and all-body display modes;
- estimated orbital ellipses or other supported periodic-orbit curves;
- visual distinction between measured trail, fitted orbit, and uncertain/incomplete analysis;
- reference-plane and orbital-plane alignment;
- reset/restart behavior after discontinuities or insufficient history.

Orbit fitting and stability assessment should use the diagnostics/analysis layer, publish read-only results to rendering, and remain independent of the selected force backend.

## 19. Runtime solver architecture

### 19.1 Public solver interface

The exact API is open, but conceptually:

~~~cpp
enum class ComputeBackend { Scalar, SIMD, GPU };
enum class ForceModel { Full, BarnesHut };
enum class GpuApi { Vulkan };

struct SolverInfo {
    ComputeBackend backend;
    ForceModel force_model;
    int dimensions;
    bool available;
    std::string display_name;
};

class ISolver {
public:
    virtual ~ISolver() = default;
    virtual void step(WorldState&, const SimulationParameters&) = 0;
    virtual SolverInfo info() const = 0;
};
~~~

### 19.2 Generic implementation family

Avoid six manually unrelated implementations. A generic family can combine policies:

~~~cpp
template<int Dimensions, ForceModel Model, ComputeBackend Backend>
class SolverImplementation;
~~~

The common code should cover:

- body iteration;
- integration;
- force-softening policy;
- static-body masks;
- result validation;
- backend metadata.

Specialized code should cover:

- scalar loops;
- AVX/AVX2 packing and reduction;
- Vulkan GPU kernels and device memory;
- quadtree/octree construction and traversal.

The runtime factory maps a SolverConfig to the concrete implementation. Runtime polymorphism or a std::variant can be used internally; the choice should be made for maintainability rather than ideology.

The factory and interface must not force hot-loop code through virtual calls, per-body heap allocations, shared-pointer ownership, or pointer-rich object graphs. Dynamic dispatch should happen once when selecting or initializing a backend; the actual step should execute a concrete, specialized implementation over dense spans/views.

The intended shape is:

~~~text
runtime configuration
    → one-time backend selection
    → concrete solver object
    → dense data-oriented step kernels
~~~

The solver object is an orchestration boundary, not an object-per-body abstraction.

### 19.3 Solver switching

Solver switching is allowed at a safe frame boundary:

~~~text
pause or enter Edit Mode
    → complete current step
    → serialize/repack current state
    → destroy old backend resources
    → initialize new backend
    → validate state equivalence
    → optionally resume
~~~

The runtime orchestration layer should own this boundary rather than making
the application coordinate raw engine pointers. A physics session owns the
selected engine, applies the selected solver configuration at each step,
publishes renderer frames, and performs candidate-engine construction and
validation before atomically replacing the active engine. The numerical
world remains the state-transfer object; switching backends must not require
the renderer or UI to understand backend-specific storage.

The dimension is not switchable after startup. A backend change must not alter physical state merely because the storage layout changes.

### 19.4 Vulkan GPU availability

The Vulkan GPU backend is compiled into the native executable when Vulkan support is enabled in the build.

At runtime:

- detect whether a usable Vulkan-capable GPU exists;
- expose GPU options only when available;
- show a clear unavailable status otherwise;
- keep scalar/SIMD CPU paths usable.

Vulkan device allocations should be owned through RAII wrappers. Every Vulkan API call should eventually have an error-checking policy.

## 20. Barnes–Hut implementation direction

The CPU and Vulkan GPU trees should be designed as separate representations sharing conceptual behavior.

### 20.1 CPU tree

Use an ownership-safe representation:

- indexed flat node pool, or
- std::unique_ptr children.

Avoid raw owning pointers and unbounded recursion.

The tree must define:

- bounds convention;
- leaf capacity;
- maximum depth;
- empty-node behavior;
- coincident-body behavior;
- center-of-mass calculation;
- opening criterion/tolerance;
- handling of the test body’s own containing node.

### 20.2 Vulkan GPU tree

Prefer a flat device-friendly node representation with explicit indices. Tree construction, body sorting, force traversal, and pointer ownership should be separated.

The Vulkan GPU representation must define:

- node count formula;
- child indexing;
- body ranges after sorting;
- initialization of every field;
- synchronization boundaries;
- maximum depth and allocation sizing;
- behavior for empty and tiny worlds.

The Vulkan GPU tree is not required to duplicate the CPU tree’s memory structure, only its mathematical force approximation.

## 21. Benchmarking and data collection

Benchmarking and telemetry are native parts of the application.

### 21.1 Passive telemetry

When enabled, collect:

- render FPS;
- simulation step time;
- force-calculation time;
- collision time;
- boundary time;
- rendering time;
- Vulkan GPU kernel time;
- host/device transfer time;
- bodies per second;
- simulated time per wall-clock second;
- memory statistics where practical.

### 21.2 Active benchmark sessions

An active benchmark should allow configuration of:

- solver/backend;
- dimension;
- force model;
- body count;
- snapshot input or deterministic body-generator command;
- random seed;
- number of warm-up steps;
- measured steps;
- timestep;
- whether rendering is included;
- output metrics.

The benchmark should ideally run the simulation workload independently of interactive rendering, while the UI displays progress and results.

### 21.3 Fair comparison rules

Comparisons require identical:

- initial body state;
- seed;
- parameters;
- timestep;
- collision/softening policy;
- boundary policy;
- number of steps;
- precision policy, where possible.

Results should distinguish:

- raw compute performance;
- transfer overhead;
- full application frame performance;
- numerical agreement.

### 21.4 Recorded data

Native output should support at least a simple machine-readable format initially, with a path open for binary snapshots later.

Record:

- exact snapshot metadata;
- snapshot format version and unit-system metadata;
- solver metadata;
- hardware/toolchain metadata;
- parameters;
- parameter-change events;
- timing samples;
- diagnostic samples;
- optional body snapshots.

Benchmarks should prefer exact snapshots as their primary input so that runs
are reproducible after interactive editing. A generator command may be stored
as provenance, but it must not replace the captured initial state used for the
measurement.

Python remains a companion for offline plotting, parameter sweeps, and exploratory analysis, not the primary live renderer.

## 22. Diagnostics and correctness criteria

The application should expose diagnostics in both tests and optionally the UI.

### 22.1 Invariants and derived quantities

Potential diagnostics:

- total mass;
- center of mass;
- total linear momentum;
- kinetic energy;
- potential energy;
- total energy;
- angular momentum;
- maximum coordinate magnitude;
- maximum velocity;
- NaN/Infinity detection;
- body count and ID validity.

Transparent softening and artificial boundaries alter ideal conservation behavior, so diagnostics must record the active policies.

### 22.2 Reference comparisons

For small deterministic scenarios:

- compare SIMD against scalar;
- compare Vulkan GPU against scalar;
- compare Barnes–Hut against full pairwise;
- compare 2D and 3D with z=0 where meaningful;
- compare before/after solver switching.

Barnes–Hut is expected to differ within tolerance. The tolerance must be measured and reported rather than hidden.

### 22.3 Minimal test scenarios

Required early cases:

- zero bodies;
- one body;
- two bodies at rest;
- two bodies with known symmetric motion;
- coincident bodies;
- unequal masses;
- static central body;
- hard-body collision;
- transparent near-collision;
- reflective boundary contact;
- 2D state with z constrained to zero;
- rotation-only visual test;
- locked mass/radius/density edit rejection;
- body creation and deletion;
- solver switching;
- deterministic serialization/replay.

## 23. Post-Newtonian and black-hole direction

### 23.1 Why it is a separate physics model

Post-Newtonian gravity is not merely a renderer toggle or a small correction hidden inside the Newtonian force function. It changes the acceleration equations and their validity range.

The first post-Newtonian model can remain body-centric and use:

- position;
- velocity;
- mass;
- optionally spin;
- additional body parameters as needed by the chosen approximation.

It can provide relativistic corrections in weak-field, slow-velocity regimes.

### 23.2 What it can realistically provide

A carefully scoped post-Newtonian model can support convincing effects such as:

- perihelion precession;
- relativistic orbital corrections;
- approximate compact-object behavior;
- spin-related corrections if implemented;
- black-hole absorption rules;
- approximate lensing in the renderer.

It will not be physically reliable arbitrarily close to an event horizon or during the final phase of a black-hole merger. The application should label the model and its limitations.

### 23.3 Black-hole body

Rough draft:

~~~cpp
struct BlackHoleProperties {
    double mass;
    Vec3 spin;
    double effective_horizon_radius;
};
~~~

The initial black-hole feature may provide:

- specialized visual representation;
- effective horizon radius;
- absorption of bodies crossing the effective horizon;
- Newtonian or post-Newtonian gravity;
- an accretion-disk visual effect later;
- approximate lensing later.

### 23.4 Full General Relativity is out of initial scope

Full numerical relativity would require evolving spacetime fields, not only body positions and velocities. A 3+1 formulation may involve:

- spatial metric;
- extrinsic curvature;
- lapse;
- shift;
- matter/energy fields;
- constraint equations;
- gauge/coordinate conditions.

Coordinates in General Relativity are labels whose physical interpretation depends on the chosen gauge. Coordinate distance and coordinate time are not automatically direct observables.

This is a separate numerical research engine and should not be an implicit requirement of the initial application. The project’s intended relativistic scope is post-Newtonian gravity plus approximate black-hole visuals unless future work justifies going further.

## 24. Build and dependency plan

### 24.1 Required build tools

- MSVC Build Tools;
- Windows SDK;
- CMake;
- Ninja;
- Vulkan SDK for Vulkan GPU builds;
- chosen graphics/UI dependencies.

### 24.2 CMake structure

The final CMake project should use:

- a simulation library target;
- a rendering/application library target;
- the main executable;
- a benchmark executable;
- test targets;
- an optional Vulkan GPU feature controlled by a CMake option.

Example conceptual targets:

~~~text
nbody_simulation
nbody_rendering
nbody_app
nbody_benchmark
nbody_tests
~~~

Use CMakePresets.json for:

- debug CPU;
- release CPU;
- release SIMD;
- release Vulkan GPU;
- tests;
- benchmarks.

Presets should not depend on Visual Studio IDE project files or stale absolute build directories.

### 24.3 Vulkan GPU build policy

Vulkan GPU support should be enabled before Vulkan sources and shaders are added to a target. Shader sources and host-side Vulkan wrappers should be built through the selected Vulkan toolchain.

The Vulkan GPU target architecture should eventually be explicit or configurable rather than always using the local machine’s native architecture, especially for distribution and benchmark reproducibility.

## 25. Implementation sequence

The following order is intended to minimize architectural rework.

### Phase 0 — repository and build reset

Status: foundational build and shell work is in place; renderer/UI
reconstruction remains ongoing.

Goals:

- preserve the old implementation for reference only;
- establish a clean source layout;
- create CMake presets;
- confirm MSVC/Ninja command-line builds;
- decide and integrate the first renderer/UI dependencies;
- define the application-shell, command, panel-layout, and presentation-state
  boundaries;
- define the graphics-ready render-scene/material boundary separately from
  the temporary debug body renderer;
- define hot/warm/cold data ownership and the domain-context boundaries;
- add a small data-layout/profiling harness before optimizing solver code;
- add warning levels and basic CI-like local commands.

Completed so far: command-line MSVC/CMake/Ninja builds, test and benchmark
targets, the application-state shell, and the first command boundary. The
remaining renderer/UI dependency and render-scene work continues in later
phases.

### Phase 1 — core types and deterministic state

Status: core state, session checkpointing, and physical snapshot persistence
are implemented. The data-oriented storage and richer visual-state portions
remain to be completed.

Implement:

- Vec3 and math primitives;
- stable BodyId;
- body physical state;
- appearance state;
- rotation state;
- 2D/3D dimension enum;
- unit system;
- dense storage and stable-ID mapping;
- explicit hot/warm/cold data separation;
- read-only views and mutation-phase boundaries;
- versioned serialization of a complete simulation snapshot; **completed**;
- an in-memory t=0 checkpoint and exact restore operation; **completed**;
- deterministic random initialization with explicit seed.

Completed deliverable: a headless program can create, serialize, load, and
inspect deterministic worlds. Snapshot loading replaces the session and enters
Startup Edit Mode, while saving is available through the command boundary.
Future work adds presentation metadata only if it can remain separate from the
canonical physical snapshot.

### Phase 2 — scalar full Newtonian reference

Status: implemented as the numerical reference. Conformance coverage,
diagnostics, and future physics additions still build on this baseline.

Implement:

- Newtonian force calculation;
- explicit transparent softening;
- chosen integration method;
- static bodies;
- 2D and 3D scalar kernels;
- edge cases for zero and one body;
- basic diagnostics.

Deliverable: headless deterministic simulation with tests for symmetry, finite values, and conservation behavior.

### Phase 3 — collision and boundary policies

Status: the principal transparent, hard-body, restitution, reflective-boundary,
and deferred-mutation paths are implemented. The phase remains open for
additional lifecycle and conservation coverage.

Implement:

- transparent collision mode;
- hard-body collision mode;
- restitution;
- reflective boundaries;
- boundary-disabled/unbounded mode;
- deferred body mutation mechanism, even if fragmentation is not yet implemented.

Deliverable: collision and boundary behavior is testable independently of rendering and solver backend.

### Phase 4 — command-line benchmark foundation

Status: benchmark target exists; the reproducible snapshot-driven benchmark
workflow is partially complete and should be formalized next to the frontend
workflow.

Implement:

- benchmark executable;
- deterministic snapshot inputs and body-generator fixtures;
- warm-up and measurement phases;
- wall-clock and simulation timing;
- telemetry records;
- snapshot and result output;
- scalar reference comparison.

Deliverable: repeatable command-line performance and correctness measurements.

### Phase 5 — dimension-agnostic native application shell

Status: application state, presentation state, simulation session, and command
dispatcher foundations are implemented. The visible shell is still
prototype-quality and is the next major frontend slice. The shell, panel
layout, command routing, camera contract, selection model, and HUD must work
for both dimensions even if the first renderer implementation is 2D.

Implement:

- main menu;
- startup Edit Mode for an empty state or a loaded snapshot;
- dimension choice as a Startup Edit Mode setting, with the choice locked once
  the run begins;
- foldable top, bottom, left, and right panel shell with animated viewport bounds;
- dimension-independent camera and viewport interfaces, with an initial 2D
  projection implementation;
- graphics-ready render-scene handoff with temporary debug body rendering;
- body-list selection and inspector routing through shared commands; viewport
  mouse selection is deferred to the dimension-specific presentation phases;
- run/pause state;
- HUD;
- simulated elapsed time;
- render FPS and solver information.

Deliverable: a usable application shell that can start either a 2D or 3D
session, backed initially by the tested scalar solver. 3D presentation may
still be a minimal/debug view at this point, but it must not require a second
UI architecture.

### Phase 6 — edit mode and body property workflows

Status: command-level lifecycle, selection, focus, body creation, checkpoint,
and snapshot load/save primitives are implemented. This phase now means
building the actual workflows and UI around those primitives.

Implement:

- running → Edit Mode slowdown;
- immediate entry when already paused in Non-Edit mode;
- Edit Mode unpause confirmation;
- reverse one-second ramp on confirmed resume;
- immediate behavior when the simulation was paused before Edit Mode;
- body list;
- body selection and comparison selection;
- static right-panel body inspector;
- world-anchored inspector overlay for viewport selection;
- universal text-based body creation/editing and validation;
- body-generator tools, including configurable solar-system-like generation;
- text-based body creation;
- initial position and velocity editing;
- locked mass/radius/density relationships;
- rejection/highlighting behavior;
- static-body and origin-anchor commands;
- top-bar Edit Mode entry and explicit t=0 reset/restore command;
- confirmed top-bar dimension switch while in Initial Edit Mode only;
- file-picker/file-path UI and notifications for the existing load/save snapshot
  commands;
- undo/redo-ready command representation.

Deliverable: a coherent interactive editor without yet requiring advanced 3D placement.

### Phase 7 — grid, references, and diagnostic visualization

Status: split deliberately into separate 2D and 3D presentation tracks. The
shared diagnostic data model can precede either renderer, but visual behavior
must be implemented and tested per dimension.

Implement:

- mathematical-origin reference and shared body/coordinate read models;
- relative position/velocity/distance display;
- body-to-body force details;
- 2D mouse selection, mathematical-origin grid lines, trajectory/orbit
  overlays, and 2D reference visuals;
- 3D reference/read-model groundwork, followed by separate 3D ray selection,
  depth-aware grid planes, trajectory/orbit-plane overlays, and 3D reference
  visuals;
- spin-axis arrow;
- equatorial ring;
- prime-meridian marker;
- three-color debug surface pattern.
- bounded recent-trajectory history with age-based fading;
- trajectory trail rendering for selected bodies and an all-body mode;
- initial orbit-analysis data model and diagnostics for bounded/periodic motion.

Deliverable: users can understand spatial, orbital, and rotational state visually, including recent movement and the confidence of any displayed orbit estimate.

### Phase 8 — SIMD full solver

Status: implemented, including AVX2 execution where available and a scalar
fallback path. Remaining work is conformance hardening, profiling, and
benchmark characterization rather than building the solver from scratch.

Implement:

- structure-of-arrays solver view;
- SIMD 2D full kernel;
- SIMD 3D full kernel;
- aligned, cache-friendly data views and reusable scratch buffers;
- measurement of layout, packing, and reorder costs separately from kernel time;
- deterministic comparison against scalar;
- remainder handling for non-multiple-of-vector-width body counts;
- benchmark integration.

Deliverable: SIMD is both faster where expected and numerically characterized.

### Phase 9 — CPU Barnes–Hut

Status: implemented for scalar full/approximated engine selection and the
existing Barnes–Hut tree. Remaining work is edge-case, accuracy, and scaling
validation.

Implement:

- bounded quadtree and octree/3D tree;
- safe node ownership or flat pool;
- empty/coincident-body handling;
- tolerance/opening criterion;
- scalar 2D and 3D traversal;
- convergence comparison against full pairwise.

Deliverable: a reliable CPU approximation backend with visible accuracy/performance tradeoffs.

### Phase 10 — SIMD Barnes–Hut

Status: implemented, including the SIMD Barnes–Hut kernel and fallback path.
Remaining work is parity validation and performance measurement.

Implement:

- SIMD processing of leaf interactions and/or selected node operations;
- comparison against scalar Barnes–Hut;
- comparison against full pairwise;
- benchmark metrics.

Deliverable: the full CPU implementation matrix is operational.

### Phase 11 — Vulkan GPU full solver

Status: not implemented. The current GPU selection is a fallback/placeholder;
Vulkan simulation compute remains future work.

Implement:

- Vulkan device state;
- GPU full pairwise computation;
- explicit transfer timing;
- state repacking;
- error checks;
- CPU/GPU numerical comparison;
- runtime device availability;
- safe solver switching.

Avoid a full N × N interaction matrix unless measurements demonstrate that it is appropriate. Prefer memory-efficient reductions or tiled force accumulation.

Deliverable: Vulkan GPU full pairwise is usable and benchmarkable.

### Phase 12 — full 3D renderer and interaction

Implement:

- production 3D camera implementation behind the dimension-independent camera
  interface;
- orbit/pan/zoom controls;
- 3D body rendering;
- point/mesh selection;
- XY/XZ/YZ grids;
- reference-plane indicators;
- 3D body property display;
- direct text placement with z position and velocity.

Deliverable: the 3D path is a first-class implementation of the same startup,
simulation, editing, selection, inspection, and command workflows already used
by 2D.

### Phase 13 — 3D camera placement and richer presentation

Implement:

- top-down XY placement;
- z-offset placement view;
- camera ray/plane intersection;
- camera-based initial velocity vector;
- surface-anchored arrows;
- camera-scaled world velocity mapping;
- three-color lighting/debug materials.

Deliverable: camera-based 3D creation is convenient while exact text entry remains available.

### Phase 14 — Vulkan GPU Barnes–Hut

Status: not implemented; depends on the Vulkan GPU execution and resource
boundaries from Phase 11.

Implement:

- flat device tree;
- deterministic or documented tree build strategy;
- body sorting and index ranges;
- force traversal;
- memory sizing and maximum depth;
- CPU/Vulkan GPU accuracy comparison;
- runtime backend switching.

Deliverable: all six required solver modes exist for 2D and 3D, subject to measured support and documented limitations.

### Phase 15 — lighting and materials

Implement:

- ambient/unlit mode;
- dynamic point lights;
- synthetic light objects;
- luminous bodies;
- automatic lighting activation;
- emissive rendering;
- three-color debug material;
- rock, gas, and star presets.

Deliverable: the application has a visually expressive but still diagnostically useful presentation.

### Phase 16 — advanced grids and orbital-plane tools

Implement:

- arbitrary coordinate-axis plane selection;
- perpendicular indicators;
- body-selected reference planes;
- text-defined plane orientation;
- aggregate and selected-body orbital-plane estimate;
- orrery visualization controls.
- bounded trajectory trails with configurable duration, sampling, and fading;
- rolling orbit-stability and periodicity assessment outside the solver hot loop;
- fitted/predicted elliptical orbit overlays when confidence thresholds are met;
- diagnostics explaining the observation window, fit, confidence, and invalidation state.

Deliverable: spatial relationships, recent trajectories, and sufficiently stable/periodic orbital geometry can be explored interactively without confusing estimates with measured paths.

### Phase 17 — heuristic collisions and body lifecycle

Implement:

- collision classification;
- absorption;
- mass transfer;
- fragmentation;
- stable ID creation/removal;
- conservation diagnostics;
- configurable minimum and maximum fragments per collision;
- configurable total fragment cap per mutation phase;
- runtime hard-body/transparent transition consistency checks with a bounded
  50-pass overlap untangling failure path and unresolved-body diagnostics;
- dimension-correct 2D and 3D fragmentation geometry, including conservation
  tests for mass, center of mass, and linear momentum;
- deferred mutation and replay support.

Deliverable: visually interesting collision outcomes without pretending to perform full material-disintegration simulation.

### Phase 18 — scale and large-system work

Implement and measure:

- unit presets;
- normalized internal units;
- origin rebasing;
- large-coordinate diagnostics;
- adaptive or hierarchical timestep experiments;
- Solar-System validation scenarios;
- Alpha-Centauri-scale prototype scenarios if practical.

Deliverable: scale limitations are explicit, measured, and manageable.

### Phase 19 — post-Newtonian model

Implement:

- separate post-Newtonian physics policy;
- validity warnings;
- benchmark/reference scenarios;
- spin-aware terms if justified;
- relativistic correction diagnostics;
- runtime model selection with clear labeling.

Deliverable: useful relativistic-looking behavior in the valid approximation regime, without claiming full General Relativity.

### Phase 20 — black holes and approximate lensing

Implement:

- black-hole body kind;
- effective horizon;
- absorption;
- black-hole material/visual preset;
- accretion-disk approximation;
- approximate gravitational lensing;
- comparison of Newtonian and post-Newtonian behavior.

Deliverable: visually convincing, explicitly approximate black-hole scenarios.

## 26. Explicit non-goals for the early rebuild

The following should not delay the foundational phases:

- full General Relativity;
- physically accurate planet fragmentation into debris fields;
- adaptive multi-scale integration from the first prototype;
- fully realistic planet textures;
- arbitrary mesh bodies;
- multiplayer or network simulation;
- mobile or cross-platform UI support;
- replacing the benchmark harness with interactive FPS alone.

These can remain future directions without being allowed to distort the initial architecture.

## 27. Decisions to revisit during implementation

These are design questions, not blockers for beginning the rebuild:

- raylib versus another lightweight renderer;
- Dear ImGui integration details;
- float versus double in each backend;
- body-oriented editable storage versus a fully normalized SoA model;
- exact integrator;
- discrete versus continuous hard-body collision detection;
- whether solver switching is allowed while actively running or only in Edit Mode;
- benchmark visualization format;
- snapshot file format;
- exact post-Newtonian approximation;
- whether an average orbital plane should use all bodies, selected bodies, or a mass threshold.
- trajectory-history storage format and sampling/decimation policy;
- orbit-stability and periodicity metrics, thresholds, and confidence presentation;
- whether fitted orbits should be restricted initially to two-body-like elliptical cases;
- behavior of trajectory and orbit overlays after origin rebasing or solver switching.

The implementation should document decisions as they are made rather than treating this draft as immutable.

## 28. Definition of a successful first release

The first meaningful release does not need black holes or every visual feature. It should provide:

- a clean command-line MSVC/CMake build;
- a native Windows executable;
- a headless deterministic simulation core;
- 2D scalar full pairwise simulation;
- transparent and hard-body collision settings;
- reflective optional boundaries;
- startup and normal Edit Modes;
- body creation with positions and initial velocities;
- body properties with linked mass/radius/density editing;
- reference-point-relative body details;
- a usable 2D renderer and HUD;
- benchmark and telemetry collection;
- tests that establish scalar correctness.

Everything beyond that should be added on top of this foundation rather than built into an untestable first prototype.
