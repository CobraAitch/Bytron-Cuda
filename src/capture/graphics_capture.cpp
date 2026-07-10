/**
 * @file graphics_capture.cpp
 * @brief Windows.Graphics.Capture implementation with D3D11-CUDA interop.
 */

#include "capture/graphics_capture.hpp"
#include "util/cuda_check.hpp"

// C++/WinRT projections + interop shims for Windows.Graphics.Capture.
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include <spdlog/spdlog.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>

#pragma comment(lib, "windowsapp.lib")

namespace bytron {

    namespace wgc  = winrt::Windows::Graphics::Capture;
    namespace wgd  = winrt::Windows::Graphics::DirectX;
    namespace wgd3 = winrt::Windows::Graphics::DirectX::Direct3D11;
    template <typename T> using ComPtr = Microsoft::WRL::ComPtr<T>;

    struct GraphicsCapture::Impl {
        explicit Impl(int monitor_index) : monitor_index_(monitor_index) {
            // Frames arrive on a threadpool thread, so we need the multithreaded
            // apartment. init_apartment is idempotent for repeated MTA joins.
            winrt::init_apartment(winrt::apartment_type::multi_threaded);

            if (!wgc::GraphicsCaptureSession::IsSupported())
                throw std::runtime_error("Windows.Graphics.Capture is not supported on this system");

            create_device();
            build_capture();
            open_ = true;

            spdlog::info("Windows.Graphics.Capture: monitor {} ({}x{}), borderless, no cursor",
                         monitor_index_, width_, height_);
        }

        ~Impl() {
            teardown_capture();
            // D3D + CUDA context resources released by ComPtr / explicit unregister.
        }

        // ── Device ────────────────────────────────────────────────────────────
        void create_device() {
            UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
            D3D_FEATURE_LEVEL feature_level{};
            HRESULT hr = D3D11CreateDevice(
                nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
                nullptr, 0, D3D11_SDK_VERSION,
                d3d_device_.GetAddressOf(), &feature_level,
                d3d_context_.GetAddressOf());
            if (FAILED(hr))
                throw std::runtime_error("Failed to create D3D11 device for WGC (HRESULT: "
                                         + std::to_string(hr) + ")");

            ComPtr<IDXGIDevice> dxgi_device;
            if (FAILED(d3d_device_.As(&dxgi_device)))
                throw std::runtime_error("Failed to get IDXGIDevice for WGC");

            // Wrap the D3D11 device as a WinRT IDirect3DDevice for the frame pool.
            winrt::com_ptr<::IInspectable> inspectable;
            winrt::check_hresult(
                CreateDirect3D11DeviceFromDXGIDevice(dxgi_device.Get(), inspectable.put()));
            rt_device_ = inspectable.as<wgd3::IDirect3DDevice>();
        }

        // ── Monitor lookup ────────────────────────────────────────────────────
        // Map monitor_index_ to an HMONITOR + size via DXGI so the index matches
        // the Desktop Duplication backend's output ordering.
        HMONITOR resolve_monitor() {
            ComPtr<IDXGIDevice> dxgi_device;
            if (FAILED(d3d_device_.As(&dxgi_device)))
                throw std::runtime_error("WGC: failed to get IDXGIDevice");

            ComPtr<IDXGIAdapter> adapter;
            if (FAILED(dxgi_device->GetAdapter(adapter.GetAddressOf())))
                throw std::runtime_error("WGC: failed to get DXGI adapter");

            ComPtr<IDXGIOutput> output;
            if (FAILED(adapter->EnumOutputs(static_cast<UINT>(monitor_index_), output.GetAddressOf())))
                throw std::runtime_error("WGC: failed to enumerate output "
                                         + std::to_string(monitor_index_));

            DXGI_OUTPUT_DESC desc{};
            output->GetDesc(&desc);
            width_  = desc.DesktopCoordinates.right  - desc.DesktopCoordinates.left;
            height_ = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;
            return desc.Monitor;
        }

        // ── Capture graph ─────────────────────────────────────────────────────
        void build_capture() {
            HMONITOR hmon = resolve_monitor();

            auto interop = winrt::get_activation_factory<
                wgc::GraphicsCaptureItem, ::IGraphicsCaptureItemInterop>();
            wgc::GraphicsCaptureItem item{ nullptr };
            winrt::check_hresult(interop->CreateForMonitor(
                hmon, winrt::guid_of<wgc::GraphicsCaptureItem>(), winrt::put_abi(item)));
            item_ = item;

            create_staging_and_register();

            frame_pool_ = wgc::Direct3D11CaptureFramePool::CreateFreeThreaded(
                rt_device_, wgd::DirectXPixelFormat::B8G8R8A8UIntNormalized,
                kFramePoolBuffers, winrt::Windows::Graphics::SizeInt32{ width_, height_ });

            session_ = frame_pool_.CreateCaptureSession(item_);

            // Documented quality options: keep the recording border and the mouse
            // cursor out of captured pixels. Guarded — older runtimes may lack them.
            try { session_.IsCursorCaptureEnabled(false); }
            catch (...) { spdlog::debug("WGC: IsCursorCaptureEnabled unsupported"); }
            try { session_.IsBorderRequired(false); }
            catch (...) { spdlog::debug("WGC: IsBorderRequired unsupported"); }

            frame_arrived_token_ = frame_pool_.FrameArrived(
                [this](auto&&, auto&&) {
                    { std::lock_guard lk(mtx_); frame_ready_ = true; }
                    cv_.notify_one();
                });
            closed_token_ = item_.Closed(
                [this](auto&&, auto&&) {
                    { std::lock_guard lk(mtx_); item_closed_ = true; }
                    cv_.notify_one();
                });

            session_.StartCapture();
        }

        void create_staging_and_register() {
            D3D11_TEXTURE2D_DESC desc{};
            desc.Width            = static_cast<UINT>(width_);
            desc.Height           = static_cast<UINT>(height_);
            desc.MipLevels        = 1;
            desc.ArraySize        = 1;
            desc.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
            desc.SampleDesc.Count = 1;
            desc.Usage            = D3D11_USAGE_DEFAULT;
            desc.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
            desc.MiscFlags        = D3D11_RESOURCE_MISC_SHARED;

            staging_tex_.Reset();
            if (FAILED(d3d_device_->CreateTexture2D(&desc, nullptr, staging_tex_.GetAddressOf())))
                throw std::runtime_error("WGC: failed to create staging texture");

            CUDA_CHECK(cudaGraphicsD3D11RegisterResource(
                &cuda_resource_, staging_tex_.Get(), cudaGraphicsRegisterFlagsNone));
        }

        void teardown_capture() noexcept {
            if (frame_pool_) frame_pool_.FrameArrived(frame_arrived_token_);
            if (item_)       item_.Closed(closed_token_);
            frame_arrived_token_ = {};
            closed_token_        = {};

            try { if (session_) session_.Close(); } catch (...) {}
            try { if (frame_pool_) frame_pool_.Close(); } catch (...) {}
            session_    = nullptr;
            frame_pool_ = nullptr;
            item_       = nullptr;

            if (cuda_resource_) {
                cudaGraphicsUnregisterResource(cuda_resource_);
                cuda_resource_ = nullptr;
            }
            staging_tex_.Reset();
        }

        // ── Acquire ───────────────────────────────────────────────────────────
        AcquireResult acquire_frame(CudaDeviceBuffer& dst, cudaStream_t stream, unsigned timeout_ms) {
            if (!open_) return AcquireResult::kError;

            {
                std::unique_lock lk(mtx_);
                if (!frame_ready_ && !item_closed_) {
                    cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                                 [&] { return frame_ready_ || item_closed_; });
                }
                if (item_closed_) { open_ = false; return AcquireResult::kAccessLost; }
                if (!frame_ready_) return AcquireResult::kTimeout;
                frame_ready_ = false;
            }

            // Drain the pool to the freshest frame (latest-wins); recycle the rest.
            wgc::Direct3D11CaptureFrame frame{ nullptr };
            while (auto next = frame_pool_.TryGetNextFrame()) {
                if (frame) frame.Close();
                frame = next;
            }
            if (!frame) return AcquireResult::kTimeout;

            auto size = frame.ContentSize();
            if (size.Width != width_ || size.Height != height_) {
                handle_resize(size.Width, size.Height, frame);
                return AcquireResult::kTimeout;  // fetch a correctly-sized frame next call
            }

            // WinRT surface -> ID3D11Texture2D via the DXGI interop shim.
            ComPtr<ID3D11Texture2D> frame_tex;
            {
                auto access = frame.Surface().as<
                    ::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
                winrt::check_hresult(access->GetInterface(
                    __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(frame_tex.GetAddressOf())));
            }

            d3d_context_->CopyResource(staging_tex_.Get(), frame_tex.Get());
            // Flush so the copy is submitted before CUDA maps the shared texture.
            d3d_context_->Flush();
            frame.Close();

            CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_resource_, stream));
            struct MapGuard {
                cudaGraphicsResource_t res; cudaStream_t s;
                ~MapGuard() { cudaGraphicsUnmapResources(1, &res, s); }
            } map_guard{ cuda_resource_, stream };

            cudaArray_t cuda_array = nullptr;
            CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0));

            CUDA_CHECK(cudaMemcpy2DFromArrayAsync(
                dst.ptr(), std::size_t(width_) * 4,
                cuda_array, 0, 0,
                std::size_t(width_) * 4, std::size_t(height_),
                cudaMemcpyDeviceToDevice, stream));

            return AcquireResult::kNewFrame;
        }

        void handle_resize(int new_w, int new_h, wgc::Direct3D11CaptureFrame& frame) {
            frame.Close();
            if (cuda_resource_) {
                cudaGraphicsUnregisterResource(cuda_resource_);
                cuda_resource_ = nullptr;
            }
            width_  = new_w;
            height_ = new_h;
            create_staging_and_register();
            frame_pool_.Recreate(rt_device_, wgd::DirectXPixelFormat::B8G8R8A8UIntNormalized,
                                 kFramePoolBuffers, winrt::Windows::Graphics::SizeInt32{ new_w, new_h });
            spdlog::info("WGC resolution change -> {}x{}", new_w, new_h);
        }

        bool reacquire() {
            teardown_capture();
            try {
                build_capture();
                {
                    std::lock_guard lk(mtx_);
                    item_closed_ = false;
                    frame_ready_ = false;
                }
                open_ = true;
                spdlog::info("WGC capture re-acquired");
                return true;
            } catch (const std::exception& e) {
                spdlog::warn("WGC re-acquire failed: {}", e.what());
                return false;
            }
        }

        static constexpr int kFramePoolBuffers = 2;

        int monitor_index_ = 0;
        int width_  = 0;
        int height_ = 0;
        std::atomic<bool> open_{ false };

        ComPtr<ID3D11Device>        d3d_device_;
        ComPtr<ID3D11DeviceContext> d3d_context_;
        ComPtr<ID3D11Texture2D>     staging_tex_;
        cudaGraphicsResource_t      cuda_resource_ = nullptr;

        wgd3::IDirect3DDevice               rt_device_{ nullptr };
        wgc::GraphicsCaptureItem            item_{ nullptr };
        wgc::Direct3D11CaptureFramePool     frame_pool_{ nullptr };
        wgc::GraphicsCaptureSession         session_{ nullptr };
        winrt::event_token frame_arrived_token_{};
        winrt::event_token closed_token_{};

        std::mutex              mtx_;
        std::condition_variable cv_;
        bool frame_ready_ = false;
        bool item_closed_ = false;
    };

    GraphicsCapture::GraphicsCapture(int monitor_index)
        : impl_(std::make_unique<Impl>(monitor_index)) {}

    GraphicsCapture::~GraphicsCapture() = default;

    AcquireResult GraphicsCapture::acquire_frame(CudaDeviceBuffer& dst, cudaStream_t stream,
                                                 unsigned timeout_ms) {
        return impl_->acquire_frame(dst, stream, timeout_ms);
    }

    bool GraphicsCapture::reacquire()            { return impl_->reacquire(); }
    bool GraphicsCapture::is_open() const noexcept { return impl_->open_.load(); }
    int  GraphicsCapture::width()   const noexcept { return impl_->width_; }
    int  GraphicsCapture::height()  const noexcept { return impl_->height_; }
    int  GraphicsCapture::pitch()   const noexcept { return impl_->width_ * 4; }

    bool GraphicsCapture::is_supported() noexcept {
        try {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            return wgc::GraphicsCaptureSession::IsSupported();
        } catch (...) {
            return false;
        }
    }

}
