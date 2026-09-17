#pragma once

namespace nbody {

struct SimdCapabilities {
    bool avx{false};
    bool avx2{false};
    bool fma{false};
};

SimdCapabilities detectSimdCapabilities();

}
