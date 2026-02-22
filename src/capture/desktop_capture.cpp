/**
 * @file desktop_capture.cpp
 * @brief DXGI Desktop Duplication and D3D11-CUDA interop implementation.
 */

#include "capture/desktop_capture.hpp"
#include "util/cuda_check.hpp"

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

namespace bytron {

    DesktopCapture::DesktopCapture(int monitor_index)
        : monitor_index_(monitor_index)
    {
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        D3D_FEATURE_LEVEL feature_level{};

        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
            nullptr, 0, D3D11_SDK_VERSION,
            d3d_device_.GetAddressOf(), &feature_level,
            d3d_context_.GetAddressOf());

        if (FAILED(hr))
            throw std::runtime_error("Failed to create D3D11 device (HRESULT: "
                                     + std::to_string(hr) + ")");

        spdlog::info("D3D11 device created, feature level: {:#x}",
                     static_cast<unsigned>(feature_level));

        ComPtr<IDXGIDevice> dxgi_device;
        hr = d3d_device_.As(&dxgi_device);
        if (FAILED(hr)) throw std::runtime_error("Failed to get IDXGIDevice");

        ComPtr<IDXGIAdapter> adapter;
        hr = dxgi_device->GetAdapter(adapter.GetAddressOf());
        if (FAILED(hr)) throw std::runtime_error("Failed to get DXGI adapter");

        ComPtr<IDXGIOutput> output;
        hr = adapter->EnumOutputs(static_cast<UINT>(monitor_index), output.GetAddressOf());
        if (FAILED(hr))
            throw std::runtime_error("Failed to enumerate DXGI output "
                                     + std::to_string(monitor_index));

        DXGI_OUTPUT_DESC output_desc{};
        output->GetDesc(&output_desc);
        width_  = output_desc.DesktopCoordinates.right  - output_desc.DesktopCoordinates.left;
        height_ = output_desc.DesktopCoordinates.bottom - output_desc.DesktopCoordinates.top;

        spdlog::info("Desktop capture target: monitor {} ({}x{})",
                     monitor_index, width_, height_);

        ComPtr<IDXGIOutput1> output1;
        hr = output.As(&output1);
        if (FAILED(hr))
            throw std::runtime_error("Failed to get IDXGIOutput1 (requires Windows 8+)");

        hr = output1->DuplicateOutput(d3d_device_.Get(), duplication_.GetAddressOf());
        if (FAILED(hr))
            throw std::runtime_error("DuplicateOutput failed (HRESULT: "
                                     + std::to_string(hr)
                                     + "). Ensure no other app holds the duplication.");

        D3D11_TEXTURE2D_DESC staging_desc{};
        staging_desc.Width            = static_cast<UINT>(width_);
        staging_desc.Height           = static_cast<UINT>(height_);
        staging_desc.MipLevels        = 1;
        staging_desc.ArraySize        = 1;
        staging_desc.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
        staging_desc.SampleDesc.Count = 1;
        staging_desc.Usage            = D3D11_USAGE_DEFAULT;
        staging_desc.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
        staging_desc.MiscFlags        = D3D11_RESOURCE_MISC_SHARED;

        hr = d3d_device_->CreateTexture2D(&staging_desc, nullptr, staging_tex_.GetAddressOf());
        if (FAILED(hr)) throw std::runtime_error("Failed to create staging texture");

        CUDA_CHECK(cudaGraphicsD3D11RegisterResource(
            &cuda_resource_, staging_tex_.Get(), cudaGraphicsRegisterFlagsNone));

        open_ = true;
        spdlog::info("DXGI Desktop Duplication initialized: {}x{}, BGRA, pitch={}",
                     width_, height_, pitch());
    }

    DesktopCapture::~DesktopCapture() {
        if (cuda_resource_) cudaGraphicsUnregisterResource(cuda_resource_);
        if (duplication_)   duplication_->ReleaseFrame();
    }

    bool DesktopCapture::reacquire() {
        if (duplication_) {
            duplication_->ReleaseFrame();
            duplication_.Reset();
        }

        ComPtr<IDXGIDevice> dxgi_device;
        HRESULT hr = d3d_device_.As(&dxgi_device);
        if (FAILED(hr)) return false;

        ComPtr<IDXGIAdapter> adapter;
        hr = dxgi_device->GetAdapter(adapter.GetAddressOf());
        if (FAILED(hr)) return false;

        ComPtr<IDXGIOutput> output;
        hr = adapter->EnumOutputs(static_cast<UINT>(monitor_index_), output.GetAddressOf());
        if (FAILED(hr)) return false;

        ComPtr<IDXGIOutput1> output1;
        hr = output.As(&output1);
        if (FAILED(hr)) return false;

        hr = output1->DuplicateOutput(d3d_device_.Get(), duplication_.GetAddressOf());
        if (FAILED(hr)) {
            spdlog::warn("DuplicateOutput re-acquire failed (HRESULT: {})", hr);
            return false;
        }

        open_ = true;
        spdlog::info("DXGI Desktop Duplication re-acquired successfully");
        return true;
    }

    AcquireResult DesktopCapture::acquire_frame_blocking(
        CudaDeviceBuffer& dst, cudaStream_t stream, UINT timeout_ms
    ) {
        if (!open_) return AcquireResult::kError;

        ComPtr<IDXGIResource> desktop_resource;
        DXGI_OUTDUPL_FRAME_INFO frame_info{};
        HRESULT hr = duplication_->AcquireNextFrame(
            timeout_ms, &frame_info, desktop_resource.GetAddressOf());

        if (hr == DXGI_ERROR_WAIT_TIMEOUT)
            return AcquireResult::kTimeout;

        if (FAILED(hr)) {
            // Access lost: display mode change, resolution switch, or UAC dialog.
            if (hr == DXGI_ERROR_ACCESS_LOST) {
                spdlog::warn("Desktop duplication access lost (display mode change?)");
                open_ = false;
                return AcquireResult::kAccessLost;
            }
            spdlog::error("AcquireNextFrame failed (HRESULT: {})", hr);
            return AcquireResult::kError;
        }

        ComPtr<ID3D11Texture2D> desktop_tex;
        hr = desktop_resource.As(&desktop_tex);
        if (SUCCEEDED(hr)) {
            d3d_context_->CopyResource(staging_tex_.Get(), desktop_tex.Get());
            // Flush ensures the CopyResource command is submitted to the GPU
            // before CUDA maps the shared texture. Without this, the D3D11
            // command may still be in the deferred command buffer.
            d3d_context_->Flush();
        }

        duplication_->ReleaseFrame();
        if (FAILED(hr)) return AcquireResult::kError;

        // Map D3D11 shared texture into CUDA, copy BGRA pixels to dst, then unmap.
        // RAII guard ensures unmap even if cudaMemcpy2DFromArrayAsync fails.
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_resource_, stream));
        struct MapGuard {
            cudaGraphicsResource_t res; cudaStream_t s;
            ~MapGuard() { cudaGraphicsUnmapResources(1, &res, s); }
        } map_guard{cuda_resource_, stream};

        cudaArray_t cuda_array = nullptr;
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0));

        CUDA_CHECK(cudaMemcpy2DFromArrayAsync(
            dst.ptr(),
            std::size_t(width_) * 4,
            cuda_array, 0, 0,
            std::size_t(width_) * 4,
            std::size_t(height_),
            cudaMemcpyDeviceToDevice, stream));

        return AcquireResult::kNewFrame;
    }

}
