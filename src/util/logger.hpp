/**
 * @file logger.hpp
 * @brief Global logging initialization using spdlog.
 */

#pragma once

#include <spdlog/spdlog.h>

namespace bytron::log {

    /**
     * @brief Initialize global logging with timestamped pattern.
     *
     * Sets pattern to "[HH:MM:SS.mmm] [LEVEL] message" and level to info.
     * Call once at program startup before any spdlog usage.
     *
     * @note Thread safety: Not thread-safe. Call from main thread only.
     */
    inline void init() {
        spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        spdlog::set_level(spdlog::level::info);
    }

    /** @brief Lower log level to debug for verbose output. */
    inline void set_debug() {
        spdlog::set_level(spdlog::level::debug);
    }

}
