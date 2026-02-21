/**
 * @file vulkan_display.cpp
 * @brief Vulkan swapchain setup, CUDA external memory interop, and frame presentation.
 */

#include "output/vulkan_display.hpp"
#include "util/cuda_check.hpp"

#include <spdlog/spdlog.h>
#include <format>
#include <stdexcept>
#include <algorithm>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#include <cuda.h>
#endif

namespace bytron {

    namespace {
        void check_vk(VkResult result, const char* call) {
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::format("Vulkan error {} in {}", static_cast<int>(result), call));
        }
    }

#define VK_CHECK(call) check_vk(call, #call)

    VulkanDisplay::VulkanDisplay(int capture_w, int capture_h, const char* title,
                                 bool transparent)
        : transparent_(transparent), capture_w_(capture_w), capture_h_(capture_h)
    {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        if (transparent_) {
            glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
            glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
        }

        window_ = glfwCreateWindow(capture_w, capture_h, title, nullptr, nullptr);
        if (!window_) {
            throw std::runtime_error("Failed to create GLFW window");
        }

        if (transparent_) {
            HWND hwnd = glfwGetWin32Window(window_);

            LONG ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE);
            ex_style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW;
            SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style);

            // Colorkey: RGB(1,0,1) treated as transparent — near-black magenta is
            // imperceptible and unlikely to appear in real content.
            SetLayeredWindowAttributes(hwnd, RGB(1, 0, 1), 0, LWA_COLORKEY);
            SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);

            glfwSetWindowPos(window_, 0, 0);

            spdlog::info("Overlay window: {}x{}, click-through, colorkey transparency, "
                         "excluded from capture", capture_w, capture_h);
        }

        create_instance();
        create_surface();
        pick_physical_device();
        create_device();
        create_swapchain();
        create_command_pool();
        create_staging_buffer();
        create_sync_objects();

        spdlog::info("VulkanDisplay initialized: {}x{}, {} frames in flight, mode: {}",
                     capture_w, capture_h, FRAMES_IN_FLIGHT,
                     transparent_ ? "transparent overlay" : "full-frame");
    }

    VulkanDisplay::~VulkanDisplay() {
        if (device_) vkDeviceWaitIdle(device_);

        if (transparent_) {
            if (staging_mapped_ptr_ && staging_mem_) {
                vkUnmapMemory(device_, staging_mem_);
                staging_mapped_ptr_ = nullptr;
            }
        } else {
            for (auto& sem : cuda_ext_sem_)
                if (sem) cudaDestroyExternalSemaphore(sem);
            if (cuda_staging_ptr_) cudaFree(cuda_staging_ptr_);
            if (cuda_ext_mem_)     cudaDestroyExternalMemory(cuda_ext_mem_);
        }

        for (auto& f : frames_) {
            if (f.fence)           vkDestroyFence(device_, f.fence, nullptr);
            if (f.cuda_ready)      vkDestroySemaphore(device_, f.cuda_ready, nullptr);
            if (f.render_finished) vkDestroySemaphore(device_, f.render_finished, nullptr);
            if (f.image_available) vkDestroySemaphore(device_, f.image_available, nullptr);
        }

        if (staging_mem_)  vkFreeMemory(device_, staging_mem_, nullptr);
        if (staging_buf_)  vkDestroyBuffer(device_, staging_buf_, nullptr);
        if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);

        if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        if (device_)    vkDestroyDevice(device_, nullptr);
        if (surface_)   vkDestroySurfaceKHR(instance_, surface_, nullptr);
        if (instance_)  vkDestroyInstance(instance_, nullptr);

        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

    void VulkanDisplay::create_instance() {
        VkApplicationInfo app_info{};
        app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName   = "Bytron";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName        = "Bytron";
        app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion         = VK_API_VERSION_1_1;

        std::uint32_t glfw_ext_count = 0;
        const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

        std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_ext_count);
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

        if (!transparent_) {
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
            extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        }

        VkInstanceCreateInfo create_info{};
        create_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo        = &app_info;
        create_info.enabledExtensionCount   = static_cast<std::uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();

        VK_CHECK(vkCreateInstance(&create_info, nullptr, &instance_));
    }

    void VulkanDisplay::create_surface() {
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
    }

    void VulkanDisplay::pick_physical_device() {
        std::uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (count == 0) {
            throw std::runtime_error("No Vulkan physical devices found");
        }

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        physical_device_ = devices[0];
        for (auto& dev : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical_device_ = dev;
                spdlog::info("Vulkan device: {}", props.deviceName);
                break;
            }
        }

        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, families.data());

        bool found_graphics = false, found_present = false;
        for (std::uint32_t i = 0; i < family_count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphics_family_ = i;
                found_graphics = true;
            }
            VkBool32 present_support = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, i, surface_, &present_support);
            if (present_support) {
                present_family_ = i;
                found_present = true;
            }
            if (found_graphics && found_present) break;
        }

        if (!found_graphics || !found_present) {
            throw std::runtime_error("Failed to find suitable queue families");
        }
    }

    void VulkanDisplay::create_device() {
        float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queue_infos;

        VkDeviceQueueCreateInfo gfx_queue_info{};
        gfx_queue_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        gfx_queue_info.queueFamilyIndex = graphics_family_;
        gfx_queue_info.queueCount       = 1;
        gfx_queue_info.pQueuePriorities = &priority;
        queue_infos.push_back(gfx_queue_info);

        if (present_family_ != graphics_family_) {
            VkDeviceQueueCreateInfo present_queue_info = gfx_queue_info;
            present_queue_info.queueFamilyIndex = present_family_;
            queue_infos.push_back(present_queue_info);
        }

        std::vector<const char*> device_extensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };

        if (!transparent_) {
            device_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
            device_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
            device_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
            device_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
        }

        VkPhysicalDeviceFeatures features{};

        VkDeviceCreateInfo create_info{};
        create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount    = static_cast<std::uint32_t>(queue_infos.size());
        create_info.pQueueCreateInfos       = queue_infos.data();
        create_info.enabledExtensionCount   = static_cast<std::uint32_t>(device_extensions.size());
        create_info.ppEnabledExtensionNames = device_extensions.data();
        create_info.pEnabledFeatures        = &features;

        VK_CHECK(vkCreateDevice(physical_device_, &create_info, nullptr, &device_));

        vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
        vkGetDeviceQueue(device_, present_family_, 0, &present_queue_);
    }

    void VulkanDisplay::create_swapchain() {
        VkSurfaceCapabilitiesKHR caps;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &caps);

        // Determine swapchain extent: use the compositor's current extent if available,
        // otherwise fall back to capture dimensions clamped to device limits.
        if (caps.currentExtent.width != UINT32_MAX) {
            swapchain_extent_ = caps.currentExtent;
        } else {
            swapchain_extent_.width  = std::clamp(static_cast<std::uint32_t>(capture_w_),
                                                   caps.minImageExtent.width,
                                                   caps.maxImageExtent.width);
            swapchain_extent_.height = std::clamp(static_cast<std::uint32_t>(capture_h_),
                                                   caps.minImageExtent.height,
                                                   caps.maxImageExtent.height);
        }

        // Query supported surface formats; prefer B8G8R8A8_UNORM (matches BGRA frame data).
        std::uint32_t format_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, formats.data());

        swapchain_format_ = formats[0].format;
        VkColorSpaceKHR color_space = formats[0].colorSpace;
        for (const auto& fmt : formats) {
            if (fmt.format == VK_FORMAT_B8G8R8A8_UNORM &&
                fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                swapchain_format_ = fmt.format;
                color_space = fmt.colorSpace;
                break;
            }
        }

        if (swapchain_format_ != VK_FORMAT_B8G8R8A8_UNORM)
            spdlog::warn("B8G8R8A8_UNORM not available; using format {}", static_cast<int>(swapchain_format_));

        std::uint32_t present_mode_count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_,
                                                   &present_mode_count, nullptr);
        std::vector<VkPresentModeKHR> present_modes(present_mode_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_,
                                                   &present_mode_count, present_modes.data());

        VkPresentModeKHR chosen_mode = VK_PRESENT_MODE_FIFO_KHR;
        for (auto mode : present_modes) {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                chosen_mode = mode;
                break;
            }
        }

        std::uint32_t image_count = caps.minImageCount + 1;
        if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
            image_count = caps.maxImageCount;
        }
        image_count = std::min(image_count, MAX_SWAPCHAIN_IMAGES);

        VkCompositeAlphaFlagBitsKHR composite_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        VkSwapchainCreateInfoKHR sc_info{};
        sc_info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        sc_info.surface          = surface_;
        sc_info.minImageCount    = image_count;
        sc_info.imageFormat      = swapchain_format_;
        sc_info.imageColorSpace  = color_space;
        sc_info.imageExtent      = swapchain_extent_;
        sc_info.imageArrayLayers = 1;
        sc_info.imageUsage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        sc_info.preTransform     = caps.currentTransform;
        sc_info.compositeAlpha   = composite_alpha;
        sc_info.presentMode      = chosen_mode;
        sc_info.clipped          = VK_TRUE;

        std::uint32_t queue_families[] = { graphics_family_, present_family_ };
        if (graphics_family_ != present_family_) {
            sc_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            sc_info.queueFamilyIndexCount = 2;
            sc_info.pQueueFamilyIndices   = queue_families;
        } else {
            sc_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        VK_CHECK(vkCreateSwapchainKHR(device_, &sc_info, nullptr, &swapchain_));

        vkGetSwapchainImagesKHR(device_, swapchain_, &swapchain_image_count_, nullptr);
        swapchain_image_count_ = std::min(swapchain_image_count_, MAX_SWAPCHAIN_IMAGES);
        vkGetSwapchainImagesKHR(device_, swapchain_, &swapchain_image_count_, swapchain_images_);

        const char* alpha_str = "OPAQUE";
        if (composite_alpha == VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
            alpha_str = "PRE_MULTIPLIED";
        else if (composite_alpha == VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
            alpha_str = "POST_MULTIPLIED";
        else if (composite_alpha == VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
            alpha_str = "INHERIT";

        spdlog::info("Vulkan swapchain: {}x{}, {} images, mode {}, alpha {}",
                     swapchain_extent_.width, swapchain_extent_.height,
                     swapchain_image_count_,
                     chosen_mode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : "FIFO",
                     alpha_str);
    }

    void VulkanDisplay::create_command_pool() {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = graphics_family_;

        VK_CHECK(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_));

        VkCommandBuffer cmds[FRAMES_IN_FLIGHT];
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool        = command_pool_;
        alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = FRAMES_IN_FLIGHT;

        VK_CHECK(vkAllocateCommandBuffers(device_, &alloc_info, cmds));

        for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
            frames_[i].cmd = cmds[i];
        }
    }

    void VulkanDisplay::create_staging_buffer() {
        staging_size_ = static_cast<VkDeviceSize>(capture_w_) * capture_h_ * 4;

        if (transparent_) {
            VkBufferCreateInfo buf_info{};
            buf_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buf_info.size        = staging_size_;
            buf_info.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VK_CHECK(vkCreateBuffer(device_, &buf_info, nullptr, &staging_buf_));

            VkMemoryRequirements mem_reqs;
            vkGetBufferMemoryRequirements(device_, staging_buf_, &mem_reqs);

            VkPhysicalDeviceMemoryProperties mem_props;
            vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

            VkMemoryPropertyFlags required =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            std::uint32_t mem_type = UINT32_MAX;
            for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
                if ((mem_reqs.memoryTypeBits & (1 << i)) &&
                    (mem_props.memoryTypes[i].propertyFlags & required) == required) {
                    mem_type = i;
                    break;
                }
            }
            if (mem_type == UINT32_MAX) {
                throw std::runtime_error("No HOST_VISIBLE+HOST_COHERENT Vulkan memory type");
            }

            VkMemoryAllocateInfo alloc_info{};
            alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize  = mem_reqs.size;
            alloc_info.memoryTypeIndex = mem_type;

            VK_CHECK(vkAllocateMemory(device_, &alloc_info, nullptr, &staging_mem_));
            VK_CHECK(vkBindBufferMemory(device_, staging_buf_, staging_mem_, 0));
            VK_CHECK(vkMapMemory(device_, staging_mem_, 0, staging_size_, 0, &staging_mapped_ptr_));

            spdlog::info("Vulkan staging buffer: {}x{} BGRA ({} bytes), host-mapped (overlay mode)",
                         capture_w_, capture_h_, staging_size_);
        } else {
            // External memory interop: Vulkan allocates device-local memory with an
            // exportable Win32 handle, which CUDA imports to get a device pointer.
            // This enables zero-copy CUDA→Vulkan frame transfer.
            VkExternalMemoryBufferCreateInfo ext_info{};
            ext_info.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
            ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

            VkBufferCreateInfo buf_info{};
            buf_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buf_info.pNext       = &ext_info;
            buf_info.size        = staging_size_;
            buf_info.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VK_CHECK(vkCreateBuffer(device_, &buf_info, nullptr, &staging_buf_));

            VkMemoryRequirements mem_reqs;
            vkGetBufferMemoryRequirements(device_, staging_buf_, &mem_reqs);

            VkPhysicalDeviceMemoryProperties mem_props;
            vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

            std::uint32_t mem_type = UINT32_MAX;
            for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
                if ((mem_reqs.memoryTypeBits & (1 << i)) &&
                    (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                    mem_type = i;
                    break;
                }
            }
            if (mem_type == UINT32_MAX) {
                throw std::runtime_error("No suitable Vulkan memory type for staging buffer");
            }

            VkExportMemoryAllocateInfo export_info{};
            export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
            export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

            VkMemoryAllocateInfo alloc_info{};
            alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.pNext           = &export_info;
            alloc_info.allocationSize  = mem_reqs.size;
            alloc_info.memoryTypeIndex = mem_type;

            VK_CHECK(vkAllocateMemory(device_, &alloc_info, nullptr, &staging_mem_));
            VK_CHECK(vkBindBufferMemory(device_, staging_buf_, staging_mem_, 0));

            VkMemoryGetWin32HandleInfoKHR handle_info{};
            handle_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
            handle_info.memory     = staging_mem_;
            handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

            auto vkGetMemoryWin32HandleKHR_fn = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
                vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR"));
            if (!vkGetMemoryWin32HandleKHR_fn) {
                throw std::runtime_error("vkGetMemoryWin32HandleKHR not available");
            }

            HANDLE win32_handle = nullptr;
            VK_CHECK(vkGetMemoryWin32HandleKHR_fn(device_, &handle_info, &win32_handle));

            cudaExternalMemoryHandleDesc cuda_mem_desc{};
            cuda_mem_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
            cuda_mem_desc.handle.win32.handle = win32_handle;
            cuda_mem_desc.size                = mem_reqs.size;

            CUDA_CHECK(cudaImportExternalMemory(&cuda_ext_mem_, &cuda_mem_desc));

            CloseHandle(win32_handle);

            cudaExternalMemoryBufferDesc buf_desc{};
            buf_desc.offset = 0;
            buf_desc.size   = mem_reqs.size;
            buf_desc.flags  = 0;

            CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&cuda_staging_ptr_, cuda_ext_mem_, &buf_desc));

            spdlog::info("Vulkan staging buffer: {}x{} BGRA ({} bytes), CUDA interop OK",
                         capture_w_, capture_h_, staging_size_);
        }
    }

    void VulkanDisplay::create_sync_objects() {
        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        // For full-frame mode: exportable semaphore that CUDA signals after staging write.
        VkExportSemaphoreCreateInfo export_sem_info{};
        export_sem_info.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        export_sem_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        VkSemaphoreCreateInfo cuda_sem_info{};
        cuda_sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        cuda_sem_info.pNext = &export_sem_info;

        auto vkGetSemaphoreWin32HandleKHR_fn = !transparent_
            ? reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
                  vkGetDeviceProcAddr(device_, "vkGetSemaphoreWin32HandleKHR"))
            : nullptr;

        if (!transparent_ && !vkGetSemaphoreWin32HandleKHR_fn)
            throw std::runtime_error("vkGetSemaphoreWin32HandleKHR not available");

        for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
            auto& f = frames_[i];
            VK_CHECK(vkCreateSemaphore(device_, &sem_info, nullptr, &f.image_available));
            VK_CHECK(vkCreateSemaphore(device_, &sem_info, nullptr, &f.render_finished));
            VK_CHECK(vkCreateFence(device_, &fence_info, nullptr, &f.fence));

            if (!transparent_) {
                VK_CHECK(vkCreateSemaphore(device_, &cuda_sem_info, nullptr, &f.cuda_ready));

                VkSemaphoreGetWin32HandleInfoKHR handle_info{};
                handle_info.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
                handle_info.semaphore  = f.cuda_ready;
                handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

                HANDLE win32_handle = nullptr;
                VK_CHECK(vkGetSemaphoreWin32HandleKHR_fn(device_, &handle_info, &win32_handle));

                cudaExternalSemaphoreHandleDesc sem_desc{};
                sem_desc.type                = cudaExternalSemaphoreHandleTypeOpaqueWin32;
                sem_desc.handle.win32.handle = win32_handle;

                CUDA_CHECK(cudaImportExternalSemaphore(&cuda_ext_sem_[i], &sem_desc));
                CloseHandle(win32_handle);
            }
        }
    }

    void VulkanDisplay::recreate_swapchain() {
        vkDeviceWaitIdle(device_);

        VkSwapchainKHR old = swapchain_;
        swapchain_ = VK_NULL_HANDLE;
        vkDestroySwapchainKHR(device_, old, nullptr);

        create_swapchain();
    }

    void VulkanDisplay::submit_and_present() {
        auto& fr = frames_[current_frame_];

        std::uint32_t image_index = 0;
        VkResult result = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                                 fr.image_available, VK_NULL_HANDLE,
                                                 &image_index);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreate_swapchain();
            return;
        }

        vkResetCommandBuffer(fr.cmd, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(fr.cmd, &begin_info);

        VkImageMemoryBarrier sc_barrier{};
        sc_barrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        sc_barrier.srcAccessMask    = 0;
        sc_barrier.dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
        sc_barrier.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        sc_barrier.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        sc_barrier.image            = swapchain_images_[image_index];
        sc_barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        vkCmdPipelineBarrier(fr.cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &sc_barrier);

        VkBufferImageCopy copy_region{};
        copy_region.bufferOffset      = 0;
        copy_region.bufferRowLength   = static_cast<std::uint32_t>(capture_w_);
        copy_region.bufferImageHeight = static_cast<std::uint32_t>(capture_h_);
        copy_region.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy_region.imageOffset       = { 0, 0, 0 };
        copy_region.imageExtent       = { swapchain_extent_.width, swapchain_extent_.height, 1 };

        vkCmdCopyBufferToImage(fr.cmd, staging_buf_,
                               swapchain_images_[image_index],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &copy_region);

        sc_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sc_barrier.dstAccessMask = 0;
        sc_barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        sc_barrier.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        vkCmdPipelineBarrier(fr.cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &sc_barrier);

        vkEndCommandBuffer(fr.cmd);

        VkSemaphore          wait_sems[2]   = { fr.image_available, fr.cuda_ready };
        VkPipelineStageFlags wait_stages[2] = { VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT };
        // Wait on CUDA semaphore only when present() signaled it this frame.
        std::uint32_t wait_count = pending_cuda_signal_ ? 2u : 1u;
        pending_cuda_signal_ = false;

        VkSubmitInfo submit_info{};
        submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount   = wait_count;
        submit_info.pWaitSemaphores      = wait_sems;
        submit_info.pWaitDstStageMask    = wait_stages;
        submit_info.commandBufferCount   = 1;
        submit_info.pCommandBuffers      = &fr.cmd;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores    = &fr.render_finished;

        VK_CHECK(vkQueueSubmit(graphics_queue_, 1, &submit_info, fr.fence));

        VkPresentInfoKHR present_info{};
        present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores    = &fr.render_finished;
        present_info.swapchainCount     = 1;
        present_info.pSwapchains        = &swapchain_;
        present_info.pImageIndices      = &image_index;

        result = vkQueuePresentKHR(present_queue_, &present_info);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            recreate_swapchain();
        }

        current_frame_ = (current_frame_ + 1) % FRAMES_IN_FLIGHT;
    }

    void VulkanDisplay::present(void* device_bgra, int width, int height,
                               cudaStream_t cuda_stream)
    {
        glfwPollEvents();
        if (should_close()) return;
        if (!device_bgra) return;

        auto& fr = frames_[current_frame_];

        vkWaitForFences(device_, 1, &fr.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &fr.fence);

        std::size_t frame_bytes = static_cast<std::size_t>(width) * 4 * height;
        CUDA_CHECK(cudaMemcpyAsync(cuda_staging_ptr_, device_bgra,
                                    frame_bytes, cudaMemcpyDeviceToDevice, cuda_stream));

        // Signal the external semaphore from CUDA after the staging write completes.
        // Vulkan will wait on this semaphore GPU-side before reading the staging buffer,
        // eliminating the need for cudaStreamSynchronize (no CPU blocking).
        cudaExternalSemaphoreSignalParams signal_params{};
        CUDA_CHECK(cudaSignalExternalSemaphoresAsync(
            &cuda_ext_sem_[current_frame_], &signal_params, 1, cuda_stream));
        pending_cuda_signal_ = true;

        submit_and_present();
    }

    void VulkanDisplay::present_overlay() {
        glfwPollEvents();
        if (should_close()) return;

        auto& fr = frames_[current_frame_];

        vkWaitForFences(device_, 1, &fr.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &fr.fence);

        submit_and_present();
    }

    bool VulkanDisplay::should_close() const {
        return glfwWindowShouldClose(window_);
    }
}
