/**
 * @file overlay_window.hpp
 * @brief Win32 layered window with per-pixel alpha for transparent overlays.
 */

#pragma once

#include <windows.h>
#include <cstddef>
#include <cstdint>

namespace bytron {

    /**
     * @brief Manages a transparent, click-through, always-on-top Win32 window.
     *
     * Uses WS_EX_LAYERED + WS_EX_TRANSPARENT + WS_EX_TOOLWINDOW with per-pixel
     * alpha via UpdateLayeredWindow. Excluded from desktop capture
     * (WDA_EXCLUDEFROMCAPTURE) to avoid feedback loops with DesktopCapture.
     *
     * Workflow: clear pixels(), draw into them, call present() which copies
     * to a DIB section and calls UpdateLayeredWindow.
     *
     * @note Thread safety: All methods must be called from the display thread.
     */
    class OverlayWindow {
    public:
        OverlayWindow(int width, int height);
        ~OverlayWindow();

        OverlayWindow(const OverlayWindow&) = delete;
        OverlayWindow& operator=(const OverlayWindow&) = delete;

        [[nodiscard]] std::uint8_t* pixels()      noexcept { return dib_bits_; }
        [[nodiscard]] std::size_t   pixel_bytes()  const noexcept { return static_cast<std::size_t>(width_) * height_ * 4; }
        [[nodiscard]] int           width()        const noexcept { return width_; }
        [[nodiscard]] int           height()       const noexcept { return height_; }

        void present();
        void poll_events();

        [[nodiscard]] bool should_close() const noexcept { return closed_; }

    private:
        static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);

        int width_;
        int height_;

        HWND    hwnd_    = nullptr;
        HDC     mem_dc_  = nullptr;
        HBITMAP dib_     = nullptr;
        HBITMAP old_bmp_ = nullptr;
        std::uint8_t* dib_bits_ = nullptr;

        bool closed_ = false;
    };
}
