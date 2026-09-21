#pragma once

#include "simulation/body_id.hpp"
#include "simulation/simulation_frame.hpp"
#include "simulation/world_state.hpp"

#include <algorithm>
#include <limits>
#include <span>
#include <vector>

namespace nbody::rendering {

struct TrajectoryPoint {
    BodyId body;
    Vec3 position;
    double simulation_time{};
};

struct TrajectoryView {
    std::span<const TrajectoryPoint> points;
};

struct TrajectoryHistoryPolicy {
    std::size_t max_samples_per_body{512};
    double sample_interval_seconds{0.05};
    double duration_seconds{60.0};
};

// Presentation-side history. It observes immutable published frames and
// interpolates between render observations when the simulation advances faster
// than the UI. The simulation and its backends remain unaware of this policy.
class TrajectoryHistory {
public:
    explicit TrajectoryHistory(TrajectoryHistoryPolicy policy = {}) : policy_(policy) {}

    void setDurationSeconds(double duration) {
        policy_.duration_seconds = std::max(duration, 0.0);
    }

    void setSamplingIntervalSeconds(double interval) {
        policy_.sample_interval_seconds = std::max(interval, 1.0e-12);
    }

    void clear() {
        tracks_.clear();
        flattened_.clear();
        last_observed_time_ = -std::numeric_limits<double>::infinity();
    }

    void observe(const FramePublication& publication) {
        if (!publication.cpu_snapshot) return;
        const SimulationFrame& frame = *publication.cpu_snapshot;
        if (frame.simulation_time < last_observed_time_) clear();
        last_observed_time_ = frame.simulation_time;
        if (policy_.max_samples_per_body == 0 || policy_.duration_seconds <= 0.0) return;

        for (const RenderBody& body : frame.bodies) {
            Track& track = findOrCreate(body.id);
            const TrajectoryPoint current{body.id, body.position, frame.simulation_time};
            if (track.points.empty()) {
                track.points.push_back(current);
                track.next_sample_time = frame.simulation_time + policy_.sample_interval_seconds;
                continue;
            }
            const TrajectoryPoint previous = track.points.back();
            const double interval = frame.simulation_time - previous.simulation_time;
            if (interval <= 0.0) continue;
            const double sample_interval = std::max(policy_.sample_interval_seconds, 1.0e-12);
            // If the simulation advances beyond the retained history window
            // in one frame (for example, the solar-system demo's 3600 s
            // timestep), there is no useful history to interpolate through.
            // Starting a new track avoids unbounded synthetic samples.
            if (interval >= policy_.duration_seconds) {
                track.points.clear();
                track.points.push_back(current);
                track.next_sample_time = frame.simulation_time + sample_interval;
                continue;
            }
            while (track.next_sample_time <= frame.simulation_time) {
                if (track.points.size() >= policy_.max_samples_per_body) break;
                const double alpha = std::clamp(
                    (track.next_sample_time - previous.simulation_time) / interval, 0.0, 1.0);
                TrajectoryPoint sample{body.id,
                    previous.position * (1.0 - alpha) + body.position * alpha,
                    track.next_sample_time};
                track.points.push_back(sample);
                track.next_sample_time += sample_interval;
            }
            if (track.points.size() >= policy_.max_samples_per_body) {
                track.next_sample_time = frame.simulation_time + sample_interval;
            }
            if (track.points.back().simulation_time < frame.simulation_time
                && frame.simulation_time - track.points.back().simulation_time
                    >= sample_interval * 0.5) {
                track.points.push_back(current);
                track.next_sample_time = frame.simulation_time + sample_interval;
            }
            trim(track, frame.simulation_time);
        }
        flattened_.clear();
        for (const Track& track : tracks_) {
            flattened_.insert(flattened_.end(), track.points.begin(), track.points.end());
        }
    }

    TrajectoryView view() const { return {flattened_}; }

private:
    struct Track {
        BodyId body;
        std::vector<TrajectoryPoint> points;
        double next_sample_time{};
    };

    Track& findOrCreate(BodyId body) {
        for (Track& track : tracks_) if (track.body == body) return track;
        tracks_.push_back({body, {}, 0.0});
        tracks_.back().points.reserve(policy_.max_samples_per_body);
        return tracks_.back();
    }

    void trim(Track& track, double current_time) const {
        const double oldest = current_time - policy_.duration_seconds;
        while (!track.points.empty()
            && (track.points.front().simulation_time < oldest
                || track.points.size() > policy_.max_samples_per_body)) {
            track.points.erase(track.points.begin());
        }
    }

    TrajectoryHistoryPolicy policy_;
    std::vector<Track> tracks_;
    std::vector<TrajectoryPoint> flattened_;
    double last_observed_time_{-std::numeric_limits<double>::infinity()};
};

}
