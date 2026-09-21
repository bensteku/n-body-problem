#include "io/snapshot.hpp"

#include "simulation/solver_configuration.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace nbody::io {

namespace {

template <typename T>
bool read(std::istream& input, T& value) {
    return static_cast<bool>(input >> value);
}

bool readTag(std::istream& input, std::string_view expected) {
    std::string actual;
    return static_cast<bool>(input >> actual) && actual == expected;
}

template <typename Enum>
bool readEnum(std::istream& input, Enum& value) {
    int encoded = 0;
    if (!read(input, encoded)) return false;
    value = static_cast<Enum>(encoded);
    return true;
}

template <typename Enum>
void writeEnum(std::ostream& output, Enum value) {
    output << static_cast<int>(value);
}

bool finite(double value) {
    return std::isfinite(value);
}

bool validParameters(const SimulationParameters& parameters) {
    return parameters.dimension == Dimension::Two || parameters.dimension == Dimension::Three;
}

bool readMaterial(std::istream& input, MaterialProperties& material) {
    return readTag(input, "MATERIAL")
        && readEnum(input, material.preset)
        && read(input, material.density)
        && read(input, material.compressive_strength)
        && read(input, material.tensile_strength)
        && read(input, material.brittleness)
        && read(input, material.energy_absorption)
        && read(input, material.restitution)
        && read(input, material.damage_threshold)
        && read(input, material.fragmentation_threshold);
}

void writeMaterial(std::ostream& output, const MaterialProperties& material) {
    output << "MATERIAL ";
    writeEnum(output, material.preset);
    output << ' ' << std::setprecision(std::numeric_limits<double>::max_digits10)
           << material.density << ' ' << material.compressive_strength
           << ' ' << material.tensile_strength << ' ' << material.brittleness
           << ' ' << material.energy_absorption << ' ' << material.restitution
           << ' ' << material.damage_threshold << ' ' << material.fragmentation_threshold << '\n';
}

bool readBody(std::istream& input, BodyState& body) {
    int is_static = 0;
    const bool result = readTag(input, "BODY")
        && read(input, body.id.value)
        && read(input, body.position.x) && read(input, body.position.y) && read(input, body.position.z)
        && read(input, body.velocity.x) && read(input, body.velocity.y) && read(input, body.velocity.z)
        && read(input, body.mass) && read(input, body.radius)
        && read(input, is_static)
        && read(input, body.accumulated_damage)
        && readEnum(input, body.kind)
        && readMaterial(input, body.material);
    body.is_static = is_static != 0;
    return result;
}

void writeBody(std::ostream& output, const BodyState& body) {
    output << "BODY " << body.id.value << ' '
           << std::setprecision(std::numeric_limits<double>::max_digits10)
           << body.position.x << ' ' << body.position.y << ' ' << body.position.z << ' '
           << body.velocity.x << ' ' << body.velocity.y << ' ' << body.velocity.z << ' '
           << body.mass << ' ' << body.radius << ' ' << (body.is_static ? 1 : 0) << ' '
           << body.accumulated_damage << ' ';
    writeEnum(output, body.kind);
    output << '\n';
    writeMaterial(output, body.material);
}

void writeParameters(std::ostream& output, const SimulationParameters& parameters) {
    output << "PARAMETERS\nDIMENSION ";
    writeEnum(output, parameters.dimension);
    output << "\nGRAVITATIONAL_CONSTANT " << std::setprecision(std::numeric_limits<double>::max_digits10)
           << parameters.gravitational_constant << "\nTIMESTEP " << parameters.timestep
           << "\nSOFTENING " << parameters.softening_length << "\nINTEGRATOR ";
    writeEnum(output, parameters.integrator);
    output << "\nSEED " << parameters.seed << "\nSOLVER ";
    writeEnum(output, parameters.solver.backend);
    output << ' ';
    writeEnum(output, parameters.solver.force_model);
    output << ' ';
    writeEnum(output, parameters.solver.kind);
    output << ' ';
    writeEnum(output, parameters.solver.threading);
    output << ' ' << parameters.solver.worker_count << ' '
           << parameters.solver.barnes_hut.opening_angle << ' '
           << parameters.solver.barnes_hut.leaf_capacity << ' '
           << parameters.solver.barnes_hut.maximum_depth << '\n';
    output << "COLLISION ";
    writeEnum(output, parameters.collision.model);
    output << ' ' << parameters.collision.restitution << ' '
           << parameters.collision.maximum_consistency_passes << ' '
           << parameters.collision.absorber_mass_tolerance << ' '
           << parameters.collision.absorber_size_tolerance << ' '
           << parameters.collision.solid_absorption_mass_ratio << ' '
           << parameters.collision.solid_absorption_size_ratio << ' '
           << (parameters.collision.broad_phase.spatial_tree_enabled ? 1 : 0) << ' '
           << parameters.collision.broad_phase.leaf_capacity << ' '
           << parameters.collision.broad_phase.maximum_depth << ' '
           << parameters.collision.broad_phase.looseness << ' '
           << parameters.collision.minimum_fragments << ' '
           << parameters.collision.maximum_fragments << ' '
           << parameters.collision.maximum_fragment_count << '\n';
    const auto& classifier = parameters.collision.classifier;
    output << "CLASSIFIER " << classifier.damage_threshold_scale << ' '
           << classifier.fragmentation_threshold_scale << ' ' << classifier.strength_scale << ' '
           << classifier.binding_scale << ' ' << classifier.tangential_energy_scale << ' '
           << (classifier.fragmentation_enabled ? 1 : 0) << ' '
           << (classifier.absorption_enabled ? 1 : 0) << ' '
           << (classifier.force_damage ? 1 : 0) << ' '
           << (classifier.force_fragmentation ? 1 : 0) << '\n';
    output << "BOUNDARY " << (parameters.boundary.enabled ? 1 : 0) << ' ';
    writeEnum(output, parameters.boundary.response);
    output << ' ' << parameters.boundary.minimum.x << ' ' << parameters.boundary.minimum.y << ' '
           << parameters.boundary.minimum.z << ' ' << parameters.boundary.maximum.x << ' '
           << parameters.boundary.maximum.y << ' ' << parameters.boundary.maximum.z << ' '
           << ' ' << parameters.boundary.restitution << ' ' << parameters.boundary.repulsion_strength
           << ' ' << parameters.boundary.repulsion_falloff << "\nEND_PARAMETERS\n";
}

bool readParameters(std::istream& input, SimulationParameters& parameters) {
    int broad_phase_enabled = 0;
    int fragmentation_enabled = 0;
    int absorption_enabled = 0;
    int force_damage = 0;
    int force_fragmentation = 0;
    int boundary_enabled = 0;
    const bool result = readTag(input, "PARAMETERS")
        && readTag(input, "DIMENSION") && readEnum(input, parameters.dimension)
        && readTag(input, "GRAVITATIONAL_CONSTANT") && read(input, parameters.gravitational_constant)
        && readTag(input, "TIMESTEP") && read(input, parameters.timestep)
        && readTag(input, "SOFTENING") && read(input, parameters.softening_length)
        && readTag(input, "INTEGRATOR") && readEnum(input, parameters.integrator)
        && readTag(input, "SEED") && read(input, parameters.seed)
        && readTag(input, "SOLVER") && readEnum(input, parameters.solver.backend)
        && readEnum(input, parameters.solver.force_model) && readEnum(input, parameters.solver.kind)
        && readEnum(input, parameters.solver.threading) && read(input, parameters.solver.worker_count)
        && read(input, parameters.solver.barnes_hut.opening_angle)
        && read(input, parameters.solver.barnes_hut.leaf_capacity)
        && read(input, parameters.solver.barnes_hut.maximum_depth)
        && readTag(input, "COLLISION") && readEnum(input, parameters.collision.model)
        && read(input, parameters.collision.restitution)
        && read(input, parameters.collision.maximum_consistency_passes)
        && read(input, parameters.collision.absorber_mass_tolerance)
        && read(input, parameters.collision.absorber_size_tolerance)
        && read(input, parameters.collision.solid_absorption_mass_ratio)
        && read(input, parameters.collision.solid_absorption_size_ratio)
        && read(input, broad_phase_enabled)
        && read(input, parameters.collision.broad_phase.leaf_capacity)
        && read(input, parameters.collision.broad_phase.maximum_depth)
        && read(input, parameters.collision.broad_phase.looseness)
        && read(input, parameters.collision.minimum_fragments)
        && read(input, parameters.collision.maximum_fragments)
        && read(input, parameters.collision.maximum_fragment_count)
        && readTag(input, "CLASSIFIER")
        && read(input, parameters.collision.classifier.damage_threshold_scale)
        && read(input, parameters.collision.classifier.fragmentation_threshold_scale)
        && read(input, parameters.collision.classifier.strength_scale)
        && read(input, parameters.collision.classifier.binding_scale)
        && read(input, parameters.collision.classifier.tangential_energy_scale)
        && read(input, fragmentation_enabled) && read(input, absorption_enabled)
        && read(input, force_damage) && read(input, force_fragmentation)
        && readTag(input, "BOUNDARY") && read(input, boundary_enabled)
        && readEnum(input, parameters.boundary.response)
        && read(input, parameters.boundary.minimum.x) && read(input, parameters.boundary.minimum.y)
        && read(input, parameters.boundary.minimum.z) && read(input, parameters.boundary.maximum.x)
        && read(input, parameters.boundary.maximum.y) && read(input, parameters.boundary.maximum.z)
        && read(input, parameters.boundary.restitution)
        && read(input, parameters.boundary.repulsion_strength)
        && read(input, parameters.boundary.repulsion_falloff)
        && readTag(input, "END_PARAMETERS");
    parameters.collision.broad_phase.spatial_tree_enabled = broad_phase_enabled != 0;
    parameters.collision.classifier.fragmentation_enabled = fragmentation_enabled != 0;
    parameters.collision.classifier.absorption_enabled = absorption_enabled != 0;
    parameters.collision.classifier.force_damage = force_damage != 0;
    parameters.collision.classifier.force_fragmentation = force_fragmentation != 0;
    parameters.boundary.enabled = boundary_enabled != 0;
    return result;
}

bool validateSnapshot(const SimulationSnapshot& snapshot, std::string& message) {
    const app::SimulationState& state = snapshot.state;
    if (snapshot.format_version != current_snapshot_format_version) {
        message = "unsupported snapshot format version";
        return false;
    }
    if (!validParameters(state.parameters)
        || state.parameters.dimension != state.world.dimension()) {
        message = "snapshot dimension is inconsistent";
        return false;
    }
    if (!finite(state.world.time()) || !finite(state.parameters.gravitational_constant)
        || !finite(state.parameters.timestep) || !finite(state.parameters.softening_length)) {
        message = "snapshot contains non-finite simulation values";
        return false;
    }
    if (!validateSolverConfiguration(state.parameters.solver).valid || !state.world.isValid()) {
        message = "snapshot contains invalid simulation state";
        return false;
    }
    return true;
}

}

SnapshotResult SnapshotSerializer::save(const std::filesystem::path& path,
                                        const SimulationSnapshot& snapshot) {
    std::string validation_message;
    if (!validateSnapshot(snapshot, validation_message)) {
        return {SnapshotStatus::Failed, validation_message, {}};
    }
    std::ofstream output(path, std::ios::trunc);
    if (!output) return {SnapshotStatus::Failed, "could not open snapshot for writing", {}};

    output << "NBODY_SNAPSHOT " << snapshot.format_version << "\nNAME "
           << std::quoted(snapshot.metadata.name) << "\nWORLD\nDIMENSION ";
    writeEnum(output, snapshot.state.world.dimension());
    output << "\nTIME " << std::setprecision(std::numeric_limits<double>::max_digits10)
           << snapshot.state.world.time() << "\nACTIVE_COLLISION ";
    writeEnum(output, snapshot.state.world.activeCollisionModel());
    output << "\nBODY_COUNT " << snapshot.state.world.bodyCount() << '\n';
    for (const BodyState& body : snapshot.state.world.snapshotBodies()) writeBody(output, body);
    output << "END_WORLD\n";
    writeParameters(output, snapshot.state.parameters);
    output << "END_SNAPSHOT\n";
    if (!output) return {SnapshotStatus::Failed, "failed while writing snapshot", {}};
    return {SnapshotStatus::Succeeded, "snapshot saved", snapshot};
}

SnapshotResult SnapshotSerializer::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) return {SnapshotStatus::Failed, "could not open snapshot for reading", {}};

    SimulationSnapshot snapshot;
    std::string header;
    if (!(input >> header >> snapshot.format_version) || header != "NBODY_SNAPSHOT"
        || !readTag(input, "NAME") || !(input >> std::quoted(snapshot.metadata.name))
        || !readTag(input, "WORLD")) {
        return {SnapshotStatus::Failed, "invalid snapshot header", {}};
    }

    Dimension dimension{};
    double time = 0.0;
    CollisionModel active_collision = CollisionModel::Transparent;
    std::size_t body_count = 0;
    if (!readTag(input, "DIMENSION") || !readEnum(input, dimension)
        || !readTag(input, "TIME") || !read(input, time)
        || !readTag(input, "ACTIVE_COLLISION") || !readEnum(input, active_collision)
        || !readTag(input, "BODY_COUNT") || !read(input, body_count)) {
        return {SnapshotStatus::Failed, "invalid snapshot world header", {}};
    }

    snapshot.state.world = WorldState(dimension);
    std::vector<BodyState> bodies;
    bodies.reserve(body_count);
    for (std::size_t index = 0; index < body_count; ++index) {
        BodyState body;
        if (!readBody(input, body)) return {SnapshotStatus::Failed, "invalid snapshot body", {}};
        bodies.push_back(body);
    }
    if (!readTag(input, "END_WORLD") || !readParameters(input, snapshot.state.parameters)
        || !readTag(input, "END_SNAPSHOT")) {
        return {SnapshotStatus::Failed, "invalid snapshot body or parameter section", {}};
    }
    snapshot.state.world.replaceBodies(std::move(bodies));
    snapshot.state.world.setTime(time);
    snapshot.state.world.setActiveCollisionModel(active_collision);

    std::string validation_message;
    if (!validateSnapshot(snapshot, validation_message)) {
        return {SnapshotStatus::Failed, validation_message, {}};
    }
    return {SnapshotStatus::Succeeded, "snapshot loaded", std::move(snapshot)};
}

}
