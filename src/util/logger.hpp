#pragma once

#include <spdlog/spdlog.h>

namespace bytron::log {

    /// <summary>
    /// Initialize spdlog with Bytron's default pattern and info level.
    /// </summary>
    inline void init() {
        spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        spdlog::set_level(spdlog::level::info);
    }

    inline void set_debug() {
        spdlog::set_level(spdlog::level::debug);
    }
}
