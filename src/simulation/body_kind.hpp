#pragma once

namespace nbody {

enum class BodyKind {
    Ordinary,
    Gas,
    Star,
    BlackHole
};

inline bool isAbsorber(BodyKind kind) {
    return kind == BodyKind::Gas || kind == BodyKind::Star || kind == BodyKind::BlackHole;
}

inline bool isBlackHole(BodyKind kind) {
    return kind == BodyKind::BlackHole;
}

}
