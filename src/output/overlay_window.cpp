/**
 * @file overlay_window.cpp
 * @brief Win32 layered window creation, DIB blitting, and per-pixel alpha presentation.
 */

#include "output/overlay_window.hpp"

#include <spdlog/spdlog.h>
#include <cstring>
#include <stdexcept>

namespace bytron {

    namespace {
        const wchar_t* OVERLAY_CLASS_NAME = L"BytronOverlay";
        bool class_registered = false;
    }

    LRESULT CALLBACK OverlayWindow::wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
        if (msg == WM_CLOSE) {
            auto* self = reinterpret_cast<OverlayWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
            if (self) self->closed_ = true;
            return 0;
        }
        if (msg == WM_DESTROY) {
            return 0;
        }
        return DefWindowProcW(hwnd, msg, wp, lp);
    }

    OverlayWindow::OverlayWindow(int width, int height)
        : width_(width), height_(height)
    {
        if (!class_registered) {
            WNDCLASSEXW wc{};
            wc.cbSize        = sizeof(wc);
            wc.lpfnWndProc   = wnd_proc;
            wc.hInstance      = GetModuleHandleW(nullptr);
            wc.lpszClassName  = OVERLAY_CLASS_NAME;
            wc.hCursor        = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));

            if (!RegisterClassExW(&wc)) {
                throw std::runtime_error("Failed to register overlay window class");
            }
            class_registered = true;
        }

        hwnd_ = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
            OVERLAY_CLASS_NAME,
            L"Bytron Overlay",
            WS_POPUP,
            0, 0, width_, height_,
            nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);

        if (!hwnd_) {
            throw std::runtime_error("Failed to create overlay window");
        }

        SetWindowLongPtrW(hwnd_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
        SetWindowDisplayAffinity(hwnd_, WDA_EXCLUDEFROMCAPTURE);

        HDC screen_dc = GetDC(nullptr);
        mem_dc_ = CreateCompatibleDC(screen_dc);
        ReleaseDC(nullptr, screen_dc);

        // Top-down DIB: negative biHeight makes scanline 0 the top row,
        // matching GPU frame layout and avoiding a vertical flip on present().
        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth       = width_;
        bmi.bmiHeader.biHeight      = -height_;
        bmi.bmiHeader.biPlanes      = 1;
        bmi.bmiHeader.biBitCount    = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        dib_ = CreateDIBSection(mem_dc_, &bmi, DIB_RGB_COLORS,
                                reinterpret_cast<void**>(&dib_bits_), nullptr, 0);
        if (!dib_ || !dib_bits_) {
            throw std::runtime_error("Failed to create DIB section for overlay");
        }

        old_bmp_ = static_cast<HBITMAP>(SelectObject(mem_dc_, dib_));

        pixels_.resize(static_cast<std::size_t>(width_) * height_ * 4, 0);

        ShowWindow(hwnd_, SW_SHOWNOACTIVATE);

        spdlog::info("Overlay window: {}x{}, click-through, per-pixel alpha, "
                     "excluded from capture", width_, height_);
    }

    OverlayWindow::~OverlayWindow() {
        if (mem_dc_) {
            if (old_bmp_) SelectObject(mem_dc_, old_bmp_);
            DeleteDC(mem_dc_);
        }
        if (dib_)  DeleteObject(dib_);
        if (hwnd_) DestroyWindow(hwnd_);
    }

    void OverlayWindow::present() {
        if (closed_) return;

        std::memcpy(dib_bits_, pixels_.data(), pixels_.size());

        POINT pt_pos = { 0, 0 };
        SIZE  sz     = { static_cast<LONG>(width_), static_cast<LONG>(height_) };
        POINT pt_src = { 0, 0 };

        BLENDFUNCTION blend{};
        blend.BlendOp             = AC_SRC_OVER;
        blend.SourceConstantAlpha = 255;
        blend.AlphaFormat         = AC_SRC_ALPHA;

        UpdateLayeredWindow(hwnd_, nullptr, &pt_pos, &sz,
                            mem_dc_, &pt_src, 0, &blend, ULW_ALPHA);

        poll_events();
    }

    void OverlayWindow::poll_events() {
        MSG msg;
        while (PeekMessageW(&msg, hwnd_, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}
