/**
 * @file vulkan_display.hpp
 * @brief Vulkan swapchain display with CUDA interop and transparent overlay modes.
 */

#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <vulkan/vulkan.h>

#include <cuda_runtime.h>
#include <cstdint>

namespace bytron {

    /**
     * @brief Vulkan-based frame presenter with two operating modes.
     *
     * **Full-frame mode** (`transparent=false`): Allocates a Vulkan staging buffer
     * with VK_EXTERNAL_MEMORY and imports it into CUDA via
     * cudaImportExternalMemory(). The inference pipeline writes BGRA frames
     * directly into this shared buffer with cudaMemcpyAsync (device-to-device),
     * then present() submits a buffer-to-image copy to the swapchain.
     *
     * **Overlay mode** (`transparent=true`): Allocates a HOST_VISIBLE+HOST_COHERENT
     * staging buffer with a persistent vkMapMemory mapping. The CPU writes detection
     * boxes into staging_ptr(), then present_overlay() copies to the swapchain.
     * The GLFW window is configured as click-through, always-on-top, with
     * colorkey transparency (RGB(1,0,1)) and excluded from desktop capture.
     *
     * Both modes use MAILBOX present mode (falls back to FIFO) with 2 frames
     * in flight for non-blocking presentation.
     *
     * @note Thread safety: All methods must be called from the display thread.
     *       CUDA interop calls in present() require the CUDA context to be current.
     */
    class VulkanDisplay {
    public:
        /**
         * @brief Create Vulkan instance, device, swapchain, and staging buffer.
         *
         * @param [in] width       Display width in pixels.
         * @param [in] height      Display height in pixels.
         * @param [in] title       GLFW window title.
         * @param [in] transparent Enable overlay mode with colorkey transparency.
         * @throws std::runtime_error on GLFW, Vulkan, or CUDA interop failure.
         */
        VulkanDisplay(int width, int height, const char* title,
                      bool transparent = false);
        ~VulkanDisplay();

        VulkanDisplay(const VulkanDisplay&) = delete;
        VulkanDisplay& operator=(const VulkanDisplay&) = delete;

        /**
         * @brief Copy a CUDA device BGRA frame to the swapchain and present.
         *
         * Waits for the current frame's fence, enqueues async copy of device_bgra
         * into the CUDA-imported staging buffer, signals an external semaphore from
         * CUDA, then submits a Vulkan buffer-to-image copy that waits on the
         * semaphore (GPU-side sync only, no CPU blocking).
         *
         * @param [in] device_bgra  Device pointer to BGRA frame (width*height*4 bytes).
         * @param [in] width        Frame width in pixels.
         * @param [in] height       Frame height in pixels.
         * @param [in] cuda_stream  CUDA stream for the async memcpy.
         * @pre Full-frame mode (transparent=false). device_bgra must be valid device memory.
         */
        void present(void* device_bgra, int width, int height,
                     cudaStream_t cuda_stream = nullptr);

        /**
         * @brief Present the host-mapped staging buffer contents to the swapchain.
         *
         * Caller writes BGRA pixels into staging_ptr() before calling this method.
         *
         * @pre Overlay mode (transparent=true). staging_ptr() contents are ready.
         */
        void present_overlay();

        [[nodiscard]] bool should_close() const;

        [[nodiscard]] void*       staging_ptr()    const noexcept { return staging_mapped_ptr_; }  ///< Host-mapped staging buffer (overlay mode only).
        [[nodiscard]] std::size_t staging_size()   const noexcept { return static_cast<std::size_t>(staging_size_); }  ///< Staging buffer size in bytes.
        [[nodiscard]] int         display_width()  const noexcept { return capture_w_; }   ///< Display width in pixels.
        [[nodiscard]] int         display_height() const noexcept { return capture_h_; }   ///< Display height in pixels.

    private:
        static constexpr int FRAMES_IN_FLIGHT = 2;               ///< Double-buffered frame submission.
        static constexpr std::uint32_t MAX_SWAPCHAIN_IMAGES = 4; ///< Upper bound on swapchain image count.

        void create_instance();
        void create_surface();
        void pick_physical_device();
        void create_device();
        void create_swapchain();
        void create_command_pool();
        void create_staging_buffer();
        void create_sync_objects();
        void recreate_swapchain();
        void submit_and_present();

        bool transparent_ = false;  ///< True = overlay mode (host-mapped), false = full-frame (CUDA interop).

        GLFWwindow* window_ = nullptr;

        VkInstance       instance_        = VK_NULL_HANDLE;
        VkSurfaceKHR     surface_         = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
        VkDevice         device_          = VK_NULL_HANDLE;
        VkQueue          graphics_queue_  = VK_NULL_HANDLE;
        VkQueue          present_queue_   = VK_NULL_HANDLE;
        std::uint32_t    graphics_family_ = 0;
        std::uint32_t    present_family_  = 0;

        VkSwapchainKHR swapchain_        = VK_NULL_HANDLE;
        VkFormat       swapchain_format_ = VK_FORMAT_B8G8R8A8_UNORM;  ///< Matches BGRA frame data from capture/CUDA.
        VkExtent2D     swapchain_extent_{};
        std::uint32_t  swapchain_image_count_ = 0;
        VkImage        swapchain_images_[MAX_SWAPCHAIN_IMAGES]{};

        VkCommandPool command_pool_ = VK_NULL_HANDLE;

        /** @brief Per-frame synchronization and command buffer resources. */
        struct FrameResources {
            VkCommandBuffer cmd             = VK_NULL_HANDLE;  ///< Primary command buffer for copy + present.
            VkFence         fence           = VK_NULL_HANDLE;  ///< Signals when GPU finishes this frame's work.
            VkSemaphore     image_available = VK_NULL_HANDLE;  ///< Signals when swapchain image is acquired.
            VkSemaphore     render_finished = VK_NULL_HANDLE;  ///< Signals when copy command completes.
            VkSemaphore     cuda_ready      = VK_NULL_HANDLE;  ///< Signaled by CUDA after staging write (full-frame mode).
        };
        FrameResources frames_[FRAMES_IN_FLIGHT]{};
        int current_frame_ = 0;  ///< Alternates 0/1 for double-buffered submission.
        bool pending_cuda_signal_ = false;  ///< True when CUDA has signaled cuda_ready for current frame.

        VkBuffer       staging_buf_ = VK_NULL_HANDLE;   ///< Staging buffer for swapchain copy source.
        VkDeviceMemory staging_mem_ = VK_NULL_HANDLE;    ///< Backing memory (device-local or host-visible).
        VkDeviceSize   staging_size_ = 0;                ///< Buffer size in bytes (width * height * 4).

        cudaExternalMemory_t    cuda_ext_mem_     = nullptr;  ///< CUDA handle to imported Vulkan memory (full-frame mode).
        void*                   cuda_staging_ptr_ = nullptr;  ///< CUDA device pointer mapped from ext_mem (full-frame mode).
        void*                   staging_mapped_ptr_ = nullptr;  ///< Host pointer from vkMapMemory (overlay mode).
        cudaExternalSemaphore_t cuda_ext_sem_[FRAMES_IN_FLIGHT]{};  ///< CUDA handles for imported Vulkan semaphores.

        int capture_w_;  ///< Capture / display width in pixels.
        int capture_h_;  ///< Capture / display height in pixels.
    };
}
