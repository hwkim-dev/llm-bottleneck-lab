/*
 * llm-lite :: Native GUI
 * Vulkan + Dear ImGui  |  Modern slim dark, zero-bloat
 * Talks to the Python Flask backend via HTTP/SSE (localhost:5000)
 *
 * Layout:
 *   +──────────────────────────────+──────────────+
 *   │  TERMINAL LOG (scroll)       │  CTRL PANEL  │
 *   │                              │  sliders     │
 *   │  > user: ...                 │  stats       │
 *   │  > model: ...                │  [SEND/RST]  │
 *   +──────────[ INPUT ]───────────+──────────────+
 *
 * For edge / headless devices (KV260) this replaces the browser GUI
 * entirely.  Later plans: move model-manager + speculative-decode toggle
 * into this process once they stabilize in the web GUI.
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
#include <cmath>

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

/* ─── url opener ─────────────────────────────────────────── */
static void open_url(const char* url) {
#ifdef _WIN32
    std::string cmd = std::string("start \"\" \"") + url + "\"";
    (void)!system(cmd.c_str());
#elif __APPLE__
    std::string cmd = std::string("open '") + url + "'";
    (void)!system(cmd.c_str());
#else
    std::string cmd = std::string("xdg-open '") + url + "' >/dev/null 2>&1 &";
    (void)!system(cmd.c_str());
#endif
}

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
    int     kv_cache    = 512;
    int     last_kv_sent = 512;

    /* backend GPU info (populated by the startup probe) */
    std::string  gpu_device;
    std::atomic<bool> gpu_fp16_faster{false};

    /* weight mode reported by /api/status */
    std::string  weight_mode = "INT4";

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

/* Fire-and-forget POST with a JSON body; response body is discarded. */
static void post_json(const std::string& path, const std::string& body)
{
    SOCKET s = connect_to_backend();
    if (s == INVALID_SOCKET) return;
    char req[2048];
    snprintf(req, sizeof(req),
        "POST %s HTTP/1.0\r\n"
        "Host: %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n\r\n%s",
        path.c_str(), BACKEND_HOST, (int)body.size(), body.c_str());
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

/* ─── ImGui style: modern slim dark (purple accent) ─────── */
static void apply_hacker_style()
{
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding    = 8.0f;
    s.ChildRounding     = 6.0f;
    s.FrameRounding     = 6.0f;
    s.GrabRounding      = 6.0f;
    s.TabRounding       = 5.0f;
    s.PopupRounding     = 6.0f;
    s.ScrollbarRounding = 8.0f;
    s.WindowBorderSize  = 1.0f;
    s.FrameBorderSize   = 0.0f;
    s.ItemSpacing       = {8, 6};
    s.FramePadding      = {10, 6};

    ImVec4* c = s.Colors;
    /* purple primary (#8b5cf6 = 0.545,0.361,0.965), cyan-ish accent2 (#06b6d4) */
    c[ImGuiCol_WindowBg]          = {0.059f, 0.059f, 0.090f, 1.f};   // #0f0f17
    c[ImGuiCol_ChildBg]           = {0.086f, 0.086f, 0.120f, 1.f};   // #16161f
    c[ImGuiCol_PopupBg]           = {0.118f, 0.118f, 0.165f, 1.f};   // #1e1e2a
    c[ImGuiCol_Border]            = {1.00f, 1.00f, 1.00f, 0.07f};
    c[ImGuiCol_FrameBg]           = {1.00f, 1.00f, 1.00f, 0.04f};
    c[ImGuiCol_FrameBgHovered]    = {0.545f, 0.361f, 0.965f, 0.15f};
    c[ImGuiCol_FrameBgActive]     = {0.545f, 0.361f, 0.965f, 0.30f};
    c[ImGuiCol_TitleBg]           = {0.086f, 0.086f, 0.120f, 1.f};
    c[ImGuiCol_TitleBgActive]     = {0.118f, 0.118f, 0.165f, 1.f};
    c[ImGuiCol_TitleBgCollapsed]  = {0.059f, 0.059f, 0.090f, 1.f};
    c[ImGuiCol_MenuBarBg]         = {0.086f, 0.086f, 0.120f, 1.f};
    c[ImGuiCol_ScrollbarBg]       = {0.0f, 0.0f, 0.0f, 0.0f};
    c[ImGuiCol_ScrollbarGrab]     = {0.145f, 0.145f, 0.196f, 1.f};
    c[ImGuiCol_ScrollbarGrabHovered]= {0.22f, 0.22f, 0.30f, 1.f};
    c[ImGuiCol_ScrollbarGrabActive] = {0.545f, 0.361f, 0.965f, 0.6f};
    c[ImGuiCol_CheckMark]         = {0.545f, 0.361f, 0.965f, 1.f};
    c[ImGuiCol_SliderGrab]        = {0.545f, 0.361f, 0.965f, 1.f};
    c[ImGuiCol_SliderGrabActive]  = {0.655f, 0.471f, 1.000f, 1.f};
    c[ImGuiCol_Button]            = {0.545f, 0.361f, 0.965f, 0.14f};
    c[ImGuiCol_ButtonHovered]     = {0.545f, 0.361f, 0.965f, 0.30f};
    c[ImGuiCol_ButtonActive]      = {0.545f, 0.361f, 0.965f, 0.55f};
    c[ImGuiCol_Header]            = {0.545f, 0.361f, 0.965f, 0.12f};
    c[ImGuiCol_HeaderHovered]     = {0.545f, 0.361f, 0.965f, 0.25f};
    c[ImGuiCol_HeaderActive]      = {0.545f, 0.361f, 0.965f, 0.45f};
    c[ImGuiCol_Separator]         = {1.00f, 1.00f, 1.00f, 0.08f};
    c[ImGuiCol_Tab]               = {0.086f, 0.086f, 0.120f, 1.f};
    c[ImGuiCol_TabHovered]        = {0.545f, 0.361f, 0.965f, 0.30f};
    c[ImGuiCol_TabActive]         = {0.545f, 0.361f, 0.965f, 0.60f};
    c[ImGuiCol_Text]              = {0.886f, 0.886f, 0.941f, 1.f};   // #e2e2f0
    c[ImGuiCol_TextDisabled]      = {0.545f, 0.545f, 0.627f, 1.f};   // #8b8ba0
    c[ImGuiCol_PlotLines]         = {0.545f, 0.361f, 0.965f, 1.f};
    c[ImGuiCol_PlotHistogram]     = {0.024f, 0.714f, 0.831f, 1.f};   // #06b6d4
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

/* ─── UI drawing (modern dark theme, web-parity) ─────────── */

/* Shared palette */
static const ImVec4 COL_ACCENT        = {0.545f, 0.361f, 0.965f, 1.00f};  // #8b5cf6
static const ImVec4 COL_ACCENT_SOFT   = {0.545f, 0.361f, 0.965f, 0.16f};
static const ImVec4 COL_ACCENT2       = {0.024f, 0.714f, 0.831f, 1.00f};  // #06b6d4
static const ImVec4 COL_TEXT          = {0.886f, 0.886f, 0.941f, 1.00f};  // #e2e2f0
static const ImVec4 COL_TEXT_MUTED    = {0.545f, 0.545f, 0.627f, 1.00f};  // #8b8ba0
static const ImVec4 COL_CARD_USER     = {0.545f, 0.361f, 0.965f, 0.10f};
static const ImVec4 COL_CARD_MODEL    = {0.118f, 0.118f, 0.165f, 1.00f};  // #1e1e2a
static const ImVec4 COL_OK            = {0.133f, 0.773f, 0.369f, 1.00f};  // #22c55e
static const ImVec4 COL_WARN          = {0.984f, 0.667f, 0.282f, 1.00f};  // #fbbf24
static const ImVec4 COL_ERR           = {0.937f, 0.267f, 0.267f, 1.00f};  // #ef4444

enum AppTab { TAB_CHAT = 0, TAB_SETTINGS = 1, TAB_ABOUT = 2 };
static int g_tab = TAB_CHAT;

static void section_header(const char* label)
{
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT_MUTED);
    ImGui::Text("%s", label);
    ImGui::PopStyleColor();
    ImGui::Separator();
    ImGui::Spacing();
}

static void draw_msg_card(const LogLine& l, int idx, float width)
{
    if (l.role == LOG_SYS) {
        ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT_MUTED);
        ImGui::TextWrapped(" ·  %s", l.text.c_str());
        ImGui::PopStyleColor();
        return;
    }
    if (l.role == LOG_ERR) {
        ImGui::PushStyleColor(ImGuiCol_Text, COL_ERR);
        ImGui::TextWrapped(" !  %s", l.text.c_str());
        ImGui::PopStyleColor();
        return;
    }

    const bool is_user = (l.role == LOG_USER);
    const ImVec4 bg        = is_user ? COL_CARD_USER  : COL_CARD_MODEL;
    const ImVec4 avatar_bg = is_user ? COL_ACCENT     : COL_ACCENT2;
    const ImVec4 name_c    = is_user ? ImVec4{0.655f, 0.471f, 1.000f, 1.f}
                                     : ImVec4{0.400f, 0.914f, 0.976f, 1.f};
    const char* name       = is_user ? "You" : "Gemma 3N";
    const char* avatar     = is_user ? "U"   : "G";

    char label[32];
    snprintf(label, sizeof(label), "##msg%d", idx);

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {14, 10});
    ImGui::PushStyleColor(ImGuiCol_ChildBg, bg);
    ImGui::PushStyleColor(ImGuiCol_Border, {1.0f, 1.0f, 1.0f, 0.05f});

    ImGui::BeginChild(label, {width, 0},
        ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Borders,
        ImGuiWindowFlags_NoScrollbar);

    /* Avatar circle (fake — we use a small colored box + text) */
    ImGui::PushStyleColor(ImGuiCol_Button,        avatar_bg);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, avatar_bg);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  avatar_bg);
    ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4{1,1,1,1});
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7.0f);
    ImGui::Button(avatar, {26, 26});
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(4);

    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::Dummy({0, 2});
    ImGui::PushStyleColor(ImGuiCol_Text, name_c);
    ImGui::Text("%s", name);
    ImGui::PopStyleColor();
    ImGui::EndGroup();

    ImGui::Dummy({0, 2});

    /* message body wrapped at card width */
    ImGui::PushTextWrapPos(width - 28);
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT);
    ImGui::TextUnformatted(l.text.c_str());
    ImGui::PopStyleColor();
    ImGui::PopTextWrapPos();

    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);

    ImGui::Dummy({0, 6});
}

static void draw_header(float sw, float h)
{
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({sw, h});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {20, 10});
    ImGui::Begin("##header", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar  | ImGuiWindowFlags_NoSavedSettings);

    /* logo */
    ImGui::PushStyleColor(ImGuiCol_Text, COL_ACCENT);
    ImGui::Text("\xe2\x9c\xa6  llm-lite");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT_MUTED);
    ImGui::Text("  Gemma 3N E4B");
    ImGui::PopStyleColor();

    /* tabs (centered-ish) */
    ImGui::SameLine(180);
    const char* labels[] = {"Chat", "Settings", "About"};
    for (int i = 0; i < 3; ++i) {
        if (i > 0) ImGui::SameLine();
        bool active = (g_tab == i);
        ImGui::PushStyleColor(ImGuiCol_Button,        active ? COL_ACCENT_SOFT : ImVec4{0,0,0,0});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, COL_ACCENT_SOFT);
        ImGui::PushStyleColor(ImGuiCol_Text,          active ? COL_ACCENT : COL_TEXT_MUTED);
        if (ImGui::Button(labels[i], {100, 0})) g_tab = i;
        ImGui::PopStyleColor(3);
    }

    /* right: weight mode + backend pill */
    ImGui::SameLine(sw - 260);
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT_MUTED);
    ImGui::Text("W:%s", g_state.weight_mode.c_str());
    ImGui::PopStyleColor();
    ImGui::SameLine();

    bool ok = g_state.backend_ok.load();
    ImGui::PushStyleColor(ImGuiCol_Text, ok ? COL_OK : COL_ERR);
    ImGui::Text(ok ? "\xe2\x97\x8f  backend" : "\xe2\x97\x8b  backend");
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();
}

static void draw_chat_tab(float sw, float sh, float header_h)
{
    static const float INPUT_H = 60.0f;
    static const float STATS_H = 34.0f;
    float body_h = sh - header_h - INPUT_H - STATS_H;

    /* GPU warning strip (above chat) */
    float warn_h = 0;
    if (g_state.gpu_fp16_faster.load()) {
        warn_h = 34;
        body_h -= warn_h;
        ImGui::SetNextWindowPos({0, header_h});
        ImGui::SetNextWindowSize({sw, warn_h});
        ImGui::PushStyleColor(ImGuiCol_WindowBg, {0.984f, 0.667f, 0.282f, 0.10f});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {20, 8});
        ImGui::Begin("##warn", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar  | ImGuiWindowFlags_NoSavedSettings);
        ImGui::PushStyleColor(ImGuiCol_Text, COL_WARN);
        ImGui::Text("\xe2\x9a\xa0  %s detected \xe2\x80\x94 FP16 may outperform INT on this iGPU.",
                    g_state.gpu_device.c_str());
        ImGui::PopStyleColor();
        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    }

    /* chat log */
    ImGui::SetNextWindowPos({0, header_h + warn_h});
    ImGui::SetNextWindowSize({sw, body_h});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {28, 16});
    ImGui::Begin("##log", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings);

    float card_w = ImGui::GetContentRegionAvail().x;
    {
        std::lock_guard<std::mutex> lk(g_state.log_mtx);
        int idx = 0;
        for (auto& l : g_state.log) {
            draw_msg_card(l, idx++, card_w);
        }
    }
    if (g_state.scroll_to_bottom) {
        ImGui::SetScrollHereY(1.0f);
        g_state.scroll_to_bottom = false;
    }
    ImGui::End();
    ImGui::PopStyleVar();

    /* input bar */
    bool busy = g_state.generating.load();
    ImGui::SetNextWindowPos({0, header_h + warn_h + body_h});
    ImGui::SetNextWindowSize({sw, INPUT_H});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {28, 12});
    ImGui::Begin("##input", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings);
    bool send_pressed = false;
    ImGui::SetNextItemWidth(sw - 180);
    if (ImGui::InputTextWithHint("##prompt", "Message Gemma 3N \xe2\x80\x94 Enter to send",
            g_state.input_buf, sizeof(g_state.input_buf),
            ImGuiInputTextFlags_EnterReturnsTrue))
        send_pressed = true;
    ImGui::SameLine();
    if (busy) ImGui::BeginDisabled();
    ImGui::PushStyleColor(ImGuiCol_Button,        COL_ACCENT);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.655f, 0.471f, 1.f, 1});
    ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4{1,1,1,1});
    if (ImGui::Button(busy ? "..." : "Send", {80, 0})) send_pressed = true;
    ImGui::PopStyleColor(3);
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4{1,1,1,0.04f});
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{1,1,1,0.10f});
    if (ImGui::Button("New", {50, 0})) {
        std::thread([](){
            http::reset_context();
            g_state.ctx_pos    = 0;
            g_state.token_count= 0;
            std::lock_guard<std::mutex> lk(g_state.log_mtx);
            g_state.log.clear();
            g_state.log.push_back({LOG_SYS, "context reset"});
        }).detach();
    }
    ImGui::PopStyleColor(2);
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
    ImGui::PopStyleVar();

    /* stats footer */
    ImGui::SetNextWindowPos({0, sh - STATS_H});
    ImGui::SetNextWindowSize({sw, STATS_H});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {28, 8});
    ImGui::Begin("##stats", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar  | ImGuiWindowFlags_NoSavedSettings);
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT_MUTED);
    ImGui::Text("ctx: ");
    ImGui::SameLine(); ImGui::PushStyleColor(ImGuiCol_Text, COL_ACCENT);
    ImGui::Text("%d", (int)g_state.ctx_pos.load()); ImGui::PopStyleColor();
    ImGui::SameLine(); ImGui::Text("    tokens: ");
    ImGui::SameLine(); ImGui::PushStyleColor(ImGuiCol_Text, COL_ACCENT);
    ImGui::Text("%d", (int)g_state.token_count.load()); ImGui::PopStyleColor();
    ImGui::SameLine(); ImGui::Text("    status: ");
    ImGui::SameLine();
    if (busy) { ImGui::PushStyleColor(ImGuiCol_Text, COL_WARN); ImGui::Text("generating \xe2\x80\xa6"); }
    else      { ImGui::PushStyleColor(ImGuiCol_Text, COL_OK);   ImGui::Text("idle"); }
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::End();
    ImGui::PopStyleVar();
}

static void draw_settings_tab(float sw, float sh, float header_h)
{
    ImGui::SetNextWindowPos({0, header_h});
    ImGui::SetNextWindowSize({sw, sh - header_h});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {32, 24});
    ImGui::Begin("##settings", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings);

    section_header("GENERATION");

    ImGui::Text("Temperature"); ImGui::SameLine(160);
    ImGui::SetNextItemWidth(240);
    ImGui::SliderFloat("##temp", &g_state.temperature, 0.1f, 1.5f, "%.2f");

    ImGui::Text("Top-p");       ImGui::SameLine(160);
    ImGui::SetNextItemWidth(240);
    ImGui::SliderFloat("##topp", &g_state.top_p, 0.5f, 1.0f, "%.2f");

    ImGui::Text("Max new tokens"); ImGui::SameLine(160);
    ImGui::SetNextItemWidth(240);
    ImGui::SliderInt("##maxt", &g_state.max_tokens, 64, 2048, "%d");

    ImGui::Text("KV cache size"); ImGui::SameLine(160);
    ImGui::SetNextItemWidth(240);
    ImGui::SliderInt("##kv", &g_state.kv_cache, 128, 2048, "%d");
    ImGui::Dummy({160, 0}); ImGui::SameLine();
    {
        float pages = roundf((g_state.kv_cache / 400.0f) * 10.0f) / 10.0f;
        ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT_MUTED);
        ImGui::Text("\xec\x95\xbd A4 %.1f\xec\x9e\xa5 \xeb\xb6\x84\xeb\x9f\x89\xec\x9e\x85\xeb\x8b\x88\xeb\x8b\xa4", pages);
        ImGui::PopStyleColor();
    }
    if (g_state.kv_cache != g_state.last_kv_sent) {
        g_state.last_kv_sent = g_state.kv_cache;
        int kv = g_state.kv_cache;
        std::thread([kv]{
            std::string body = "{\"max_tokens\":" + std::to_string(kv) + "}";
            http::post_json("/api/config/kv_cache", body);
        }).detach();
    }

    section_header("MODEL");
    ImGui::Text("Weight mode  "); ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, COL_ACCENT);
    ImGui::Text("%s", g_state.weight_mode.c_str());
    ImGui::PopStyleColor();

    ImGui::Text("GPU          "); ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT);
    ImGui::Text("%s", g_state.gpu_device.empty() ? "(probing...)" : g_state.gpu_device.c_str());
    ImGui::PopStyleColor();

    if (g_state.gpu_fp16_faster.load()) {
        ImGui::PushStyleColor(ImGuiCol_Text, COL_WARN);
        ImGui::Dummy({160, 0}); ImGui::SameLine();
        ImGui::TextWrapped("\xe2\x9a\xa0  Older iGPU \xe2\x80\x94 FP16 may outperform INT.");
        ImGui::PopStyleColor();
    }

    ImGui::Dummy({0, 8});
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT_MUTED);
    ImGui::TextWrapped("Model manager (download / quantize / delete) is available in the web GUI at http://127.0.0.1:5000 \xe2\x80\x94 Settings \xe2\x86\x92 Models.");
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();
}

static void about_link_card(const char* title, const char* subtitle, const char* url,
                            ImVec4 icon_bg, const char* icon_label)
{
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, {1, 1, 1, 0.03f});
    ImGui::PushStyleColor(ImGuiCol_Border, {1, 1, 1, 0.08f});
    char cid[32]; snprintf(cid, sizeof(cid), "##aboutcard_%s", title);
    ImGui::BeginChild(cid, {380, 70},
        ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar);
    ImGui::PushStyleColor(ImGuiCol_Button,        icon_bg);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, icon_bg);
    ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4{1,1,1,1});
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 9.0f);
    ImGui::Dummy({4, 4});
    ImGui::SameLine();
    ImGui::Button(icon_label, {40, 40});
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);

    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT_MUTED);
    ImGui::Text("%s", subtitle);
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT);
    ImGui::Text("%s", title);
    ImGui::PopStyleColor();
    ImGui::EndGroup();

    /* click target covers the whole child */
    ImGui::SetCursorPos({0, 0});
    if (ImGui::InvisibleButton(title, {380, 70})) {
        open_url(url);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
}

static void draw_about_tab(float sw, float sh, float header_h)
{
    ImGui::SetNextWindowPos({0, header_h});
    ImGui::SetNextWindowSize({sw, sh - header_h});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {40, 32});
    ImGui::Begin("##about", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings);

    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT);
    ImGui::Text("llm-lite \xe2\x80\x94 Gemma 3N E4B \xeb\xa1\x9c\xec\xbb\xac \xec\xb6\x94\xeb\xa1\xa0 \xec\x97\x94\xec\xa7\x84");
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT_MUTED);
    ImGui::TextWrapped("Multi-backend inference. No cloud dependency. Runs on Ryzen APU, Apple Silicon, RPi, KV260.");
    ImGui::PopStyleColor();

    ImGui::Dummy({0, 18});

    about_link_card("hkimw / llm-bottleneck-lab",  "Visit on GitHub",
                    "https://github.com/hkimw/llm-bottleneck-lab",
                    COL_ACCENT, "GH");
    ImGui::SameLine();
    about_link_card("llm-lite tag",          "Read the blog",
                    "https://hkimw.github.io/hkimw/",
                    COL_ACCENT2, "B");

    ImGui::Dummy({0, 24});
    ImGui::PushStyleColor(ImGuiCol_Text, COL_TEXT_MUTED);
    ImGui::TextWrapped("\xc2\xa9 2026 hkimw \xe2\x80\x94 Licensed under the MIT License.");
    ImGui::TextWrapped("Built on NumPy \xc2\xb7 Flask \xc2\xb7 Dear ImGui \xc2\xb7 Vulkan \xc2\xb7 huggingface_hub.");
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();
}

static void draw_ui()
{
    ImGuiIO& io = ImGui::GetIO();
    float sw = io.DisplaySize.x;
    float sh = io.DisplaySize.y;
    static const float HEADER_H = 52.0f;

    draw_header(sw, HEADER_H);

    switch (g_tab) {
        case TAB_CHAT:     draw_chat_tab(sw, sh, HEADER_H);     break;
        case TAB_SETTINGS: draw_settings_tab(sw, sh, HEADER_H); break;
        case TAB_ABOUT:    draw_about_tab(sw, sh, HEADER_H);    break;
    }
}

/* ─── backend health probe thread ───────────────────────── */
static std::string json_field_str(const std::string& body, const std::string& key)
{
    std::string search = "\"" + key + "\":\"";
    auto p = body.find(search);
    if (p == std::string::npos) return "";
    p += search.size();
    auto e = body.find('"', p);
    return (e == std::string::npos) ? "" : body.substr(p, e - p);
}

static bool json_field_bool(const std::string& body, const std::string& key)
{
    std::string search = "\"" + key + "\":";
    auto p = body.find(search);
    if (p == std::string::npos) return false;
    return body.compare(p + search.size(), 4, "true") == 0;
}

static void health_probe_loop()
{
    bool ever_ok = false;
    int tick = 0;
    while (true) {
        bool ok = false;
        http::get("/api/health", &ok);
        g_state.backend_ok = ok;

        if (ok && !ever_ok) {
            ever_ok = true;
            bool _o = false;
            std::string gpu = http::get("/api/gpu-info", &_o);
            if (_o) {
                g_state.gpu_device = json_field_str(gpu, "device");
                g_state.gpu_fp16_faster = json_field_bool(gpu, "fp16_faster");
            }
        }
        if (ok && (tick % 3 == 0)) {
            bool _o = false;
            std::string st = http::get("/api/status", &_o);
            if (_o) {
                std::string wm = json_field_str(st, "weight_mode");
                if (!wm.empty()) g_state.weight_mode = wm;
            }
        }
        ++tick;
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

        VkClearValue clear{{0.059f, 0.059f, 0.090f, 1.0f}};
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
