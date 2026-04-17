/*
 * llm-lite :: Native GUI
 * Vulkan + Dear ImGui  |  Hacker-aesthetic, zero-bloat
 * Talks to the Python Flask backend via HTTP/SSE (localhost:5000)
 *
 * Layout:
 *   +──────────────────────────────+──────────────+
 *   │  TERMINAL LOG (scroll)       │  CTRL PANEL  │
 *   │                              │  sliders     │
 *   │  > user: ...                 │  stats       │
 *   │  > model: ...                │  [SEND/RST]  │
 *   +──────────[ INPUT ]───────────+──────────────+
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <functional>

/* POSIX sockets (Linux / macOS) */
#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   using ssize_t = int;
#  define SOCK_CLOSE closesocket
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <unistd.h>
#  include <fcntl.h>
#  define SOCK_CLOSE close
#  define SOCKET int
#  define INVALID_SOCKET (-1)
#endif

/* ─── config ─────────────────────────────────────────────── */
#define APP_VERSION  "0.1.0"
#define BACKEND_HOST "127.0.0.1"

static constexpr int   WIN_W        = 1200;
static constexpr int   WIN_H        = 720;
static constexpr int   LOG_MAX      = 800;
static constexpr int   BACKEND_PORT = 5000;

/* ─── log entry ──────────────────────────────────────────── */
enum LogRole { LOG_SYS, LOG_USER, LOG_MODEL, LOG_ERR };

struct LogLine {
    LogRole     role;
    std::string text;
};

/* ─── shared app state ───────────────────────────────────── */
struct AppState {
    std::mutex              log_mtx;
    std::deque<LogLine>     log;

    std::atomic<bool>       generating{false};
    std::atomic<int>        token_count{0};
    std::atomic<int>        ctx_pos{0};
    std::atomic<bool>       backend_ok{false};

    /* generation params (written only from main thread) */
    float   temperature = 0.65f;
    float   top_p       = 0.90f;
    int     max_tokens  = 512;

    char    input_buf[2048]{};
    bool    scroll_to_bottom = false;
} g_state;

/* ─── helpers ────────────────────────────────────────────── */
static void push_log(LogRole r, const std::string& s)
{
    std::lock_guard<std::mutex> lk(g_state.log_mtx);
    if (g_state.log.size() >= LOG_MAX)
        g_state.log.pop_front();
    g_state.log.push_back({r, s});
    g_state.scroll_to_bottom = true;
}

/* ─── tiny HTTP/SSE client ───────────────────────────────── */
namespace http {

static SOCKET connect_to_backend()
{
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(BACKEND_PORT);
    addr.sin_addr.s_addr = inet_addr(BACKEND_HOST);

    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return INVALID_SOCKET;

    if (::connect(s, (sockaddr*)&addr, sizeof(addr)) < 0) {
        SOCK_CLOSE(s);
        return INVALID_SOCKET;
    }
    return s;
}

/* Returns response body, or "" on error.  Sets *ok=false on failure. */
static std::string get(const std::string& path, bool* ok)
{
    SOCKET s = connect_to_backend();
    if (s == INVALID_SOCKET) { *ok = false; return ""; }

    std::string req = "GET " + path + " HTTP/1.0\r\nHost: " +
                      BACKEND_HOST + "\r\nConnection: close\r\n\r\n";
    send(s, req.c_str(), (int)req.size(), 0);

    std::string resp;
    char buf[4096];
    ssize_t n;
    while ((n = recv(s, buf, sizeof(buf)-1, 0)) > 0) {
        buf[n] = '\0';
        resp += buf;
    }
    SOCK_CLOSE(s);

    auto pos = resp.find("\r\n\r\n");
    *ok = (pos != std::string::npos);
    return (pos != std::string::npos) ? resp.substr(pos + 4) : "";
}

/*
 * POST JSON, stream SSE lines back.
 * on_token(text) called for each "data: {...}" with "token" key.
 * on_done(total_tok, ctx)  called on final event.
 */
static void post_chat_sse(
    const std::string& prompt,
    float temperature, float top_p, int max_tokens,
    std::function<void(const std::string&, int)> on_token,
    std::function<void(int,int)> on_done,
    std::function<void(const std::string&)> on_error)
{
    SOCKET s = connect_to_backend();
    if (s == INVALID_SOCKET) { on_error("Cannot connect to backend"); return; }

    /* build JSON body manually (no deps) */
    char body[1024];
    snprintf(body, sizeof(body),
        "{\"prompt\":\"%s\",\"temperature\":%.2f,\"top_p\":%.2f,\"max_tokens\":%d}",
        prompt.c_str(), temperature, top_p, max_tokens);

    char req[2048];
    snprintf(req, sizeof(req),
        "POST /api/chat HTTP/1.0\r\n"
        "Host: %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n\r\n%s",
        BACKEND_HOST, (int)strlen(body), body);

    send(s, req, (int)strlen(req), 0);

    /* skip HTTP headers */
    std::string buf_str;
    char tmp[1];
    int header_end = 0;
    while (header_end < 4) {
        if (recv(s, tmp, 1, 0) <= 0) break;
        buf_str += tmp[0];
        if ((header_end == 0 || header_end == 2) && tmp[0] == '\r') header_end++;
        else if ((header_end == 1 || header_end == 3) && tmp[0] == '\n') header_end++;
        else header_end = 0;
    }

    /* stream SSE */
    std::string line;
    char c[1];
    while (true) {
        ssize_t n = recv(s, c, 1, 0);
        if (n <= 0) break;
        if (c[0] == '\n') {
            if (line.rfind("data: ", 0) == 0) {
                std::string payload = line.substr(6);
                /* minimal JSON parse: look for "token":"...", "done":true */
                auto find_str = [&](const std::string& key) -> std::string {
                    std::string search = "\"" + key + "\":\"";
                    auto p = payload.find(search);
                    if (p == std::string::npos) return "";
                    p += search.size();
                    auto e = payload.find('"', p);
                    return (e == std::string::npos) ? "" : payload.substr(p, e - p);
                };
                auto find_int = [&](const std::string& key) -> int {
                    std::string search = "\"" + key + "\":";
                    auto p = payload.find(search);
                    if (p == std::string::npos) return 0;
                    return std::stoi(payload.substr(p + search.size()));
                };

                std::string tok = find_str("token");
                if (!tok.empty()) {
                    int cnt = find_int("count");
                    on_token(tok, cnt);
                }
                if (payload.find("\"done\":true") != std::string::npos) {
                    int total = find_int("total_tokens");
                    int ctx   = find_int("context_pos");
                    on_done(total, ctx);
                    break;
                }
                if (payload.find("\"error\"") != std::string::npos) {
                    on_error(find_str("error"));
                    break;
                }
            }
            line.clear();
        } else if (c[0] != '\r') {
            line += c[0];
        }
    }
    SOCK_CLOSE(s);
}

static void reset_context()
{
    SOCKET s = connect_to_backend();
    if (s == INVALID_SOCKET) return;
    const char* req =
        "POST /api/reset HTTP/1.0\r\n"
        "Host: " BACKEND_HOST "\r\n"
        "Content-Length: 0\r\n"
        "Connection: close\r\n\r\n";
    send(s, req, (int)strlen(req), 0);
    char buf[512]; recv(s, buf, sizeof(buf), 0);
    SOCK_CLOSE(s);
}

} // namespace http

/* ─── inference thread ───────────────────────────────────── */
static void run_inference(std::string prompt,
                          float temperature, float top_p, int max_tokens)
{
    g_state.generating = true;
    g_state.token_count = 0;

    push_log(LOG_USER, "> " + prompt);

    std::string model_buf;
    push_log(LOG_MODEL, "");   /* placeholder line we'll update */

    http::post_chat_sse(
        prompt, temperature, top_p, max_tokens,
        /* on_token */
        [&](const std::string& tok, int cnt) {
            model_buf += tok;
            g_state.token_count = cnt;
            {
                std::lock_guard<std::mutex> lk(g_state.log_mtx);
                if (!g_state.log.empty())
                    g_state.log.back().text = "  " + model_buf;
                g_state.scroll_to_bottom = true;
            }
        },
        /* on_done */
        [&](int total, int ctx) {
            g_state.ctx_pos = ctx;
            push_log(LOG_SYS,
                     std::string("  [generated ") + std::to_string(total) +
                     " tokens  ctx=" + std::to_string(ctx) + "]");
        },
        /* on_error */
        [&](const std::string& err) {
            push_log(LOG_ERR, "  [ERR] " + err);
        }
    );

    g_state.generating = false;
}

/* ─── ImGui style: sharp-edge hacker dark ───────────────── */
static void apply_hacker_style()
{
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding    = 0.0f;
    s.ChildRounding     = 0.0f;
    s.FrameRounding     = 2.0f;
    s.GrabRounding      = 0.0f;
    s.TabRounding       = 0.0f;
    s.ScrollbarRounding = 0.0f;
    s.WindowBorderSize  = 1.0f;
    s.FrameBorderSize   = 1.0f;
    s.ItemSpacing       = {6, 4};
    s.FramePadding      = {8, 5};

    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg]          = {0.05f, 0.05f, 0.05f, 1.f};
    c[ImGuiCol_ChildBg]           = {0.07f, 0.07f, 0.07f, 1.f};
    c[ImGuiCol_PopupBg]           = {0.08f, 0.08f, 0.08f, 1.f};
    c[ImGuiCol_Border]            = {0.22f, 0.22f, 0.22f, 1.f};
    c[ImGuiCol_FrameBg]           = {0.10f, 0.10f, 0.10f, 1.f};
    c[ImGuiCol_FrameBgHovered]    = {0.15f, 0.15f, 0.15f, 1.f};
    c[ImGuiCol_FrameBgActive]     = {0.20f, 0.20f, 0.20f, 1.f};
    c[ImGuiCol_TitleBg]           = {0.08f, 0.08f, 0.08f, 1.f};
    c[ImGuiCol_TitleBgActive]     = {0.10f, 0.10f, 0.10f, 1.f};
    c[ImGuiCol_TitleBgCollapsed]  = {0.05f, 0.05f, 0.05f, 1.f};
    c[ImGuiCol_MenuBarBg]         = {0.08f, 0.08f, 0.08f, 1.f};
    c[ImGuiCol_ScrollbarBg]       = {0.05f, 0.05f, 0.05f, 1.f};
    c[ImGuiCol_ScrollbarGrab]     = {0.25f, 0.25f, 0.25f, 1.f};
    c[ImGuiCol_ScrollbarGrabHovered]= {0.35f, 0.35f, 0.35f, 1.f};
    c[ImGuiCol_ScrollbarGrabActive] = {0.45f, 0.45f, 0.45f, 1.f};
    c[ImGuiCol_CheckMark]         = {0.30f, 0.80f, 0.50f, 1.f};
    c[ImGuiCol_SliderGrab]        = {0.20f, 0.70f, 0.90f, 1.f};
    c[ImGuiCol_SliderGrabActive]  = {0.30f, 0.85f, 1.00f, 1.f};
    c[ImGuiCol_Button]            = {0.15f, 0.15f, 0.15f, 1.f};
    c[ImGuiCol_ButtonHovered]     = {0.22f, 0.22f, 0.22f, 1.f};
    c[ImGuiCol_ButtonActive]      = {0.30f, 0.30f, 0.30f, 1.f};
    c[ImGuiCol_Header]            = {0.15f, 0.15f, 0.15f, 1.f};
    c[ImGuiCol_HeaderHovered]     = {0.20f, 0.20f, 0.20f, 1.f};
    c[ImGuiCol_HeaderActive]      = {0.28f, 0.28f, 0.28f, 1.f};
    c[ImGuiCol_Separator]         = {0.20f, 0.20f, 0.20f, 1.f};
    c[ImGuiCol_Tab]               = {0.08f, 0.08f, 0.08f, 1.f};
    c[ImGuiCol_TabHovered]        = {0.15f, 0.15f, 0.15f, 1.f};
    c[ImGuiCol_TabActive]         = {0.14f, 0.40f, 0.55f, 1.f};
    c[ImGuiCol_Text]              = {0.83f, 0.83f, 0.83f, 1.f};
    c[ImGuiCol_TextDisabled]      = {0.40f, 0.40f, 0.40f, 1.f};
    c[ImGuiCol_PlotLines]         = {0.30f, 0.80f, 0.50f, 1.f};
    c[ImGuiCol_PlotHistogram]     = {0.20f, 0.65f, 0.85f, 1.f};
}

/* ────────────────────────────────────────────────────────────
 * Vulkan boilerplate
 * ──────────────────────────────────────────────────────────── */
#ifndef NDEBUG
static constexpr bool VK_VALIDATION = true;
#else
static constexpr bool VK_VALIDATION = false;
#endif

struct VkCtx {
    VkInstance               instance        = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    VkSurfaceKHR             surface         = VK_NULL_HANDLE;
    VkPhysicalDevice         gpu             = VK_NULL_HANDLE;
    VkDevice                 device          = VK_NULL_HANDLE;
    VkQueue                  gfx_queue       = VK_NULL_HANDLE;
    uint32_t                 gfx_family      = UINT32_MAX;

    VkSwapchainKHR           swapchain       = VK_NULL_HANDLE;
    VkFormat                 sc_format       = VK_FORMAT_UNDEFINED;
    VkExtent2D               sc_extent{};
    std::vector<VkImage>     sc_images;
    std::vector<VkImageView> sc_views;

    VkRenderPass             render_pass     = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    VkCommandPool            cmd_pool        = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> cmd_bufs;

    VkDescriptorPool         imgui_pool      = VK_NULL_HANDLE;

    static constexpr uint32_t FRAMES_IN_FLIGHT = 2;
    VkSemaphore img_available[FRAMES_IN_FLIGHT]{};
    VkSemaphore render_done[FRAMES_IN_FLIGHT]{};
    VkFence     in_flight[FRAMES_IN_FLIGHT]{};
    uint32_t    frame_idx = 0;
} g_vk;

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* d, void*)
{
    fprintf(stderr, "[VK] %s\n", d->pMessage);
    return VK_FALSE;
}

static void vk_check(VkResult r, const char* ctx)
{
    if (r != VK_SUCCESS) {
        fprintf(stderr, "Vulkan error %d in %s\n", r, ctx);
        exit(1);
    }
}
#define VK_CHECK(x) vk_check((x), #x)

static void create_instance()
{
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName   = "llm-lite-native";
    ai.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    ai.apiVersion         = VK_API_VERSION_1_2;

    uint32_t glfw_ext_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    std::vector<const char*> exts(glfw_exts, glfw_exts + glfw_ext_count);
    std::vector<const char*> layers;

    if (VK_VALIDATION) {
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo        = &ai;
    ci.enabledExtensionCount   = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    ci.enabledLayerCount       = (uint32_t)layers.size();
    ci.ppEnabledLayerNames     = layers.data();
    VK_CHECK(vkCreateInstance(&ci, nullptr, &g_vk.instance));

    if (VK_VALIDATION) {
        auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(g_vk.instance, "vkCreateDebugUtilsMessengerEXT");
        if (fn) {
            VkDebugUtilsMessengerCreateInfoEXT dci{
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            dci.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = debug_callback;
            fn(g_vk.instance, &dci, nullptr, &g_vk.debug_messenger);
        }
    }
}

static void pick_gpu()
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g_vk.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(g_vk.instance, &count, devs.data());

    /* prefer discrete, fallback to first */
    for (auto d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g_vk.gpu = d;
            fprintf(stdout, "[GPU] discrete: %s\n", p.deviceName);
            return;
        }
    }
    g_vk.gpu = devs[0];
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(g_vk.gpu, &p);
    fprintf(stdout, "[GPU] integrated: %s\n", p.deviceName);
}

static void create_device()
{
    uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.gpu, &qfc, nullptr);
    std::vector<VkQueueFamilyProperties> qfps(qfc);
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.gpu, &qfc, qfps.data());

    for (uint32_t i = 0; i < qfc; i++) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(g_vk.gpu, i, g_vk.surface, &present);
        if ((qfps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
            g_vk.gfx_family = i;
            break;
        }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_vk.gfx_family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    const char* dev_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = 1;
    dci.ppEnabledExtensionNames = dev_exts;
    VK_CHECK(vkCreateDevice(g_vk.gpu, &dci, nullptr, &g_vk.device));
    vkGetDeviceQueue(g_vk.device, g_vk.gfx_family, 0, &g_vk.gfx_queue);
}

static void create_swapchain(int w, int h)
{
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_vk.gpu, g_vk.surface, &caps);

    uint32_t fmt_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.gpu, g_vk.surface, &fmt_count, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.gpu, g_vk.surface, &fmt_count, fmts.data());

    VkSurfaceFormatKHR chosen = fmts[0];
    for (auto& f : fmts)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            chosen = f;

    g_vk.sc_format = chosen.format;
    g_vk.sc_extent = {(uint32_t)w, (uint32_t)h};

    uint32_t img_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) img_count = std::min(img_count, caps.maxImageCount);

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface          = g_vk.surface;
    sci.minImageCount    = img_count;
    sci.imageFormat      = chosen.format;
    sci.imageColorSpace  = chosen.colorSpace;
    sci.imageExtent      = g_vk.sc_extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped          = VK_TRUE;
    VK_CHECK(vkCreateSwapchainKHR(g_vk.device, &sci, nullptr, &g_vk.swapchain));

    uint32_t ic = 0;
    vkGetSwapchainImagesKHR(g_vk.device, g_vk.swapchain, &ic, nullptr);
    g_vk.sc_images.resize(ic);
    vkGetSwapchainImagesKHR(g_vk.device, g_vk.swapchain, &ic, g_vk.sc_images.data());

    g_vk.sc_views.resize(ic);
    for (uint32_t i = 0; i < ic; i++) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image                       = g_vk.sc_images[i];
        vci.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
        vci.format                      = g_vk.sc_format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(g_vk.device, &vci, nullptr, &g_vk.sc_views[i]));
    }
}

static void create_render_pass()
{
    VkAttachmentDescription att{};
    att.format         = g_vk.sc_format;
    att.samples        = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription  sp{};
    sp.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1;
    sp.pColorAttachments    = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rci.attachmentCount = 1;  rci.pAttachments = &att;
    rci.subpassCount    = 1;  rci.pSubpasses   = &sp;
    rci.dependencyCount = 1;  rci.pDependencies= &dep;
    VK_CHECK(vkCreateRenderPass(g_vk.device, &rci, nullptr, &g_vk.render_pass));
}

static void create_framebuffers()
{
    g_vk.framebuffers.resize(g_vk.sc_views.size());
    for (size_t i = 0; i < g_vk.sc_views.size(); i++) {
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass      = g_vk.render_pass;
        fci.attachmentCount = 1;
        fci.pAttachments    = &g_vk.sc_views[i];
        fci.width           = g_vk.sc_extent.width;
        fci.height          = g_vk.sc_extent.height;
        fci.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(g_vk.device, &fci, nullptr, &g_vk.framebuffers[i]));
    }
}

static void create_cmd_pool_and_buffers()
{
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = g_vk.gfx_family;
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(g_vk.device, &pci, nullptr, &g_vk.cmd_pool));

    g_vk.cmd_bufs.resize(g_vk.framebuffers.size());
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = g_vk.cmd_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = (uint32_t)g_vk.cmd_bufs.size();
    VK_CHECK(vkAllocateCommandBuffers(g_vk.device, &ai, g_vk.cmd_bufs.data()));
}

static void create_sync_objects()
{
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo     fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < VkCtx::FRAMES_IN_FLIGHT; i++) {
        VK_CHECK(vkCreateSemaphore(g_vk.device, &si, nullptr, &g_vk.img_available[i]));
        VK_CHECK(vkCreateSemaphore(g_vk.device, &si, nullptr, &g_vk.render_done[i]));
        VK_CHECK(vkCreateFence    (g_vk.device, &fi, nullptr, &g_vk.in_flight[i]));
    }
}

static void create_imgui_descriptor_pool()
{
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    };
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets       = 1;
    pi.poolSizeCount = 1;
    pi.pPoolSizes    = pool_sizes;
    VK_CHECK(vkCreateDescriptorPool(g_vk.device, &pi, nullptr, &g_vk.imgui_pool));
}

static void cleanup_swapchain()
{
    vkDeviceWaitIdle(g_vk.device);
    for (auto fb : g_vk.framebuffers)
        vkDestroyFramebuffer(g_vk.device, fb, nullptr);
    g_vk.framebuffers.clear();
    for (auto v : g_vk.sc_views)
        vkDestroyImageView(g_vk.device, v, nullptr);
    g_vk.sc_views.clear();
    if (g_vk.swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(g_vk.device, g_vk.swapchain, nullptr);
        g_vk.swapchain = VK_NULL_HANDLE;
    }
}

static void rebuild_swapchain(GLFWwindow* win)
{
    int w, h;
    while (true) {
        glfwGetFramebufferSize(win, &w, &h);
        if (w > 0 && h > 0) break;
        glfwWaitEvents();
    }
    cleanup_swapchain();
    create_swapchain(w, h);
    create_framebuffers();
    ImGui_ImplVulkan_SetMinImageCount((uint32_t)g_vk.sc_images.size());
}

/* ─── UI drawing ─────────────────────────────────────────── */
static void draw_ui()
{
    ImGuiIO& io = ImGui::GetIO();
    float    sw = io.DisplaySize.x;
    float    sh = io.DisplaySize.y;

    static const float CTRL_W    = 280.0f;
    static const float INPUT_H   = 54.0f;
    static const float HEADER_H  = 28.0f;
    float log_w = sw - CTRL_W - 1.0f;
    float log_h = sh - INPUT_H - HEADER_H;

    /* ── header bar ── */
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({sw, HEADER_H});
    ImGui::Begin("##header", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar  | ImGuiWindowFlags_NoSavedSettings);
    ImGui::TextColored({0.30f, 0.80f, 0.50f, 1.f}, "llm-lite");
    ImGui::SameLine();
    ImGui::TextDisabled(":: Gemma 3N E4B  |  v" APP_VERSION "  |  INT4 iGPU");
    ImGui::SameLine(sw - 220.0f);
    bool ok = g_state.backend_ok.load();
    ImGui::TextColored(ok ? ImVec4{0.3f,0.8f,0.3f,1.f} : ImVec4{0.8f,0.3f,0.3f,1.f},
                       ok ? "[backend: OK]" : "[backend: OFFLINE]");
    ImGui::End();

    /* ── terminal log ── */
    ImGui::SetNextWindowPos({0, HEADER_H});
    ImGui::SetNextWindowSize({log_w, log_h});
    ImGui::Begin("##log", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_HorizontalScrollbar);

    {
        std::lock_guard<std::mutex> lk(g_state.log_mtx);
        for (auto& l : g_state.log) {
            switch (l.role) {
            case LOG_SYS:
                ImGui::TextColored({0.45f,0.45f,0.45f,1.f}, "%s", l.text.c_str()); break;
            case LOG_USER:
                ImGui::TextColored({0.30f,0.75f,1.00f,1.f}, "%s", l.text.c_str()); break;
            case LOG_MODEL:
                ImGui::TextColored({0.83f,0.83f,0.83f,1.f}, "%s", l.text.c_str()); break;
            case LOG_ERR:
                ImGui::TextColored({1.00f,0.35f,0.35f,1.f}, "%s", l.text.c_str()); break;
            }
        }
    }

    if (g_state.scroll_to_bottom) {
        ImGui::SetScrollHereY(1.0f);
        g_state.scroll_to_bottom = false;
    }
    ImGui::End();

    /* ── input bar ── */
    ImGui::SetNextWindowPos({0, HEADER_H + log_h});
    ImGui::SetNextWindowSize({log_w, INPUT_H});
    ImGui::Begin("##input", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings);

    bool send_pressed = false;
    ImGui::SetNextItemWidth(log_w - 80.0f);
    if (ImGui::InputText("##prompt", g_state.input_buf, sizeof(g_state.input_buf),
                         ImGuiInputTextFlags_EnterReturnsTrue))
        send_pressed = true;
    ImGui::SameLine();
    bool busy = g_state.generating.load();
    if (busy) ImGui::BeginDisabled();
    if (ImGui::Button("SEND", {60, 0})) send_pressed = true;
    if (busy) ImGui::EndDisabled();

    if (send_pressed && !busy && g_state.input_buf[0]) {
        std::string p = g_state.input_buf;
        g_state.input_buf[0] = '\0';
        float t = g_state.temperature;
        float tp = g_state.top_p;
        int mx = g_state.max_tokens;
        std::thread(run_inference, p, t, tp, mx).detach();
    }
    ImGui::End();

    /* ── control panel ── */
    ImGui::SetNextWindowPos({log_w + 1.0f, HEADER_H});
    ImGui::SetNextWindowSize({CTRL_W, sh - HEADER_H});
    ImGui::Begin("##ctrl", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings);

    ImGui::TextColored({0.45f,0.45f,0.45f,1.f}, "-- PARAMETERS ------------------");
    ImGui::Spacing();

    ImGui::Text("Temperature"); ImGui::SameLine(110);
    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("##temp", &g_state.temperature, 0.1f, 1.5f, "%.2f");

    ImGui::Text("Top-p");       ImGui::SameLine(110);
    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("##topp", &g_state.top_p, 0.5f, 1.0f, "%.2f");

    ImGui::Text("Max tokens");  ImGui::SameLine(110);
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("##maxt", &g_state.max_tokens, 64, 2048, "%d");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored({0.45f,0.45f,0.45f,1.f}, "-- ACTIONS ---------------------");
    ImGui::Spacing();

    if (busy) ImGui::BeginDisabled();
    if (ImGui::Button("NEW CONVERSATION", {CTRL_W-24, 0})) {
        std::thread([](){
            http::reset_context();
            g_state.ctx_pos    = 0;
            g_state.token_count= 0;
            std::lock_guard<std::mutex> lk(g_state.log_mtx);
            g_state.log.clear();
            g_state.log.push_back({LOG_SYS, "  [context reset]"});
        }).detach();
    }
    if (busy) ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored({0.45f,0.45f,0.45f,1.f}, "-- STATS -----------------------");
    ImGui::Spacing();

    ImGui::Text("ctx pos  "); ImGui::SameLine();
    ImGui::TextColored({0.30f,0.80f,0.50f,1.f}, "%d", (int)g_state.ctx_pos);

    ImGui::Text("gen tok  "); ImGui::SameLine();
    ImGui::TextColored({0.30f,0.80f,0.50f,1.f}, "%d", (int)g_state.token_count);

    ImGui::Text("status   "); ImGui::SameLine();
    if (busy)
        ImGui::TextColored({1.0f,0.8f,0.0f,1.f}, "GENERATING");
    else
        ImGui::TextColored({0.30f,0.80f,0.50f,1.f}, "IDLE");

    ImGui::Text("backend  "); ImGui::SameLine();
    ImGui::TextColored(g_state.backend_ok
        ? ImVec4{0.3f,0.8f,0.3f,1.f}
        : ImVec4{0.8f,0.3f,0.3f,1.f},
        "%s", BACKEND_HOST);

    ImGui::End();
}

/* ─── backend health probe thread ───────────────────────── */
static void health_probe_loop()
{
    while (true) {
        bool ok = false;
        http::get("/api/health", &ok);
        g_state.backend_ok = ok;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
}

/* ─── main ───────────────────────────────────────────────── */
int main()
{
#ifdef _WIN32
    WSADATA wsd; WSAStartup(MAKEWORD(2,2), &wsd);
#endif

    /* GLFW */
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H,
        "llm-lite :: Gemma 3N E4B", nullptr, nullptr);

    /* Vulkan init */
    create_instance();
    VK_CHECK(glfwCreateWindowSurface(g_vk.instance, window, nullptr, &g_vk.surface));
    pick_gpu();
    create_device();

    int fw, fh;
    glfwGetFramebufferSize(window, &fw, &fh);
    create_swapchain(fw, fh);
    create_render_pass();
    create_framebuffers();
    create_cmd_pool_and_buffers();
    create_sync_objects();
    create_imgui_descriptor_pool();

    /* ImGui */
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;   /* no imgui.ini */
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    apply_hacker_style();

    /* glyph ranges: Basic Latin + Box Drawing + Hangul */
    static const ImWchar kr_ranges[] = {
        0x0020, 0x00FF,  /* Basic Latin + Latin Supplement */
        0x2500, 0x257F,  /* Box Drawing (─ │ ┌ etc.) */
        0x25A0, 0x25FF,  /* Geometric Shapes */
        0x3130, 0x318F,  /* Hangul Compatibility Jamo */
        0xAC00, 0xD7A3,  /* Hangul Syllables */
        0,
    };
    /* Korean fonts (prefer NanumGothic → NotoSansCJK fallback) */
    const char* ko_font_paths[] = {
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/baekmuk/gulim.ttf",
        "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
        nullptr,
    };
    bool font_loaded = false;
    for (int i = 0; ko_font_paths[i]; ++i) {
        if (FILE* f = fopen(ko_font_paths[i], "rb")) {
            fclose(f);
            io.Fonts->AddFontFromFileTTF(ko_font_paths[i], 15.0f, nullptr, kr_ranges);
            font_loaded = true;
            break;
        }
    }
    if (!font_loaded) {
        /* Latin-only fallback */
        const char* lat_paths[] = {
            "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
            "/usr/share/fonts/jetbrains-mono/JetBrainsMono-Regular.ttf",
            nullptr,
        };
        for (int i = 0; lat_paths[i]; ++i) {
            if (FILE* f = fopen(lat_paths[i], "rb")) {
                fclose(f);
                io.Fonts->AddFontFromFileTTF(lat_paths[i], 14.0f);
                font_loaded = true;
                break;
            }
        }
        if (!font_loaded) io.Fonts->AddFontDefault();
    }

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo vii{};
    vii.Instance        = g_vk.instance;
    vii.PhysicalDevice  = g_vk.gpu;
    vii.Device          = g_vk.device;
    vii.QueueFamily     = g_vk.gfx_family;
    vii.Queue           = g_vk.gfx_queue;
    vii.DescriptorPool  = g_vk.imgui_pool;
    vii.RenderPass      = g_vk.render_pass;
    vii.MinImageCount   = (uint32_t)g_vk.sc_images.size();
    vii.ImageCount      = (uint32_t)g_vk.sc_images.size();
    vii.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&vii);

    push_log(LOG_SYS, "  llm-lite native gui :: v" APP_VERSION);
    push_log(LOG_SYS, "  backend: http://" BACKEND_HOST ":" + std::to_string(BACKEND_PORT));
    push_log(LOG_SYS, "  type a message and press ENTER or click SEND");

    /* background health probe */
    std::thread(health_probe_loop).detach();

    bool swapchain_dirty = false;

    /* ── main loop ── */
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (swapchain_dirty) {
            rebuild_swapchain(window);
            swapchain_dirty = false;
            continue;
        }

        uint32_t fi = g_vk.frame_idx % VkCtx::FRAMES_IN_FLIGHT;
        vkWaitForFences(g_vk.device, 1, &g_vk.in_flight[fi], VK_TRUE, UINT64_MAX);

        uint32_t img_idx;
        VkResult acq = vkAcquireNextImageKHR(
            g_vk.device, g_vk.swapchain, UINT64_MAX,
            g_vk.img_available[fi], VK_NULL_HANDLE, &img_idx);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
            swapchain_dirty = true; continue;
        }

        vkResetFences(g_vk.device, 1, &g_vk.in_flight[fi]);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        draw_ui();
        ImGui::Render();

        VkCommandBuffer cb = g_vk.cmd_bufs[img_idx];
        vkResetCommandBuffer(cb, 0);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);

        VkClearValue clear{{0.05f, 0.05f, 0.05f, 1.0f}};
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass        = g_vk.render_pass;
        rbi.framebuffer       = g_vk.framebuffers[img_idx];
        rbi.renderArea.extent = g_vk.sc_extent;
        rbi.clearValueCount   = 1;
        rbi.pClearValues      = &clear;
        vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
        vkCmdEndRenderPass(cb);
        vkEndCommandBuffer(cb);

        VkPipelineStageFlags wait_stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount   = 1;
        si.pWaitSemaphores      = &g_vk.img_available[fi];
        si.pWaitDstStageMask    = &wait_stages;
        si.commandBufferCount   = 1;
        si.pCommandBuffers      = &cb;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores    = &g_vk.render_done[fi];
        vkQueueSubmit(g_vk.gfx_queue, 1, &si, g_vk.in_flight[fi]);

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores    = &g_vk.render_done[fi];
        pi.swapchainCount     = 1;
        pi.pSwapchains        = &g_vk.swapchain;
        pi.pImageIndices      = &img_idx;
        VkResult pr = vkQueuePresentKHR(g_vk.gfx_queue, &pi);
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR)
            swapchain_dirty = true;

        g_vk.frame_idx++;
    }

    /* ── cleanup ── */
    vkDeviceWaitIdle(g_vk.device);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    for (uint32_t i = 0; i < VkCtx::FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(g_vk.device, g_vk.img_available[i], nullptr);
        vkDestroySemaphore(g_vk.device, g_vk.render_done[i],   nullptr);
        vkDestroyFence    (g_vk.device, g_vk.in_flight[i],     nullptr);
    }
    vkDestroyDescriptorPool(g_vk.device, g_vk.imgui_pool, nullptr);
    vkFreeCommandBuffers(g_vk.device, g_vk.cmd_pool,
        (uint32_t)g_vk.cmd_bufs.size(), g_vk.cmd_bufs.data());
    vkDestroyCommandPool(g_vk.device, g_vk.cmd_pool, nullptr);
    cleanup_swapchain();
    vkDestroyRenderPass(g_vk.device, g_vk.render_pass, nullptr);
    vkDestroyDevice(g_vk.device, nullptr);
    vkDestroySurfaceKHR(g_vk.instance, g_vk.surface, nullptr);
    if (VK_VALIDATION && g_vk.debug_messenger) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(g_vk.instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(g_vk.instance, g_vk.debug_messenger, nullptr);
    }
    vkDestroyInstance(g_vk.instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
