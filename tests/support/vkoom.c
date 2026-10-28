/* LD_PRELOAD shim: denies device-local vkAllocateMemory (VK_ERROR_OUT_OF_DEVICE_MEMORY) while the
 * file named by VKOOM_DENY_FILE exists, so a test can fail GPU allocations at load or mid-run.
 * Adapted from MidMeeting's tests/support/vkoom.c. */
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    VkDevice device;
    VkPhysicalDeviceMemoryProperties mem_props;
    int valid;
} device_entry;

#define MAX_DEVICES 16
static device_entry g_devices[MAX_DEVICES];
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

static PFN_vkGetInstanceProcAddr real_gipa = NULL;
static PFN_vkGetDeviceProcAddr real_gdpa = NULL;
static PFN_vkCreateDevice real_create_device = NULL;
static PFN_vkAllocateMemory real_alloc_memory = NULL;
static PFN_vkGetPhysicalDeviceMemoryProperties real_get_mem_props = NULL;

static int armed(void) {
    const char *f = getenv("VKOOM_DENY_FILE");
    return f != NULL && *f != '\0' && access(f, F_OK) == 0;
}

static void resolve(void) {
    if (!real_gipa) real_gipa = (PFN_vkGetInstanceProcAddr)dlsym(RTLD_NEXT, "vkGetInstanceProcAddr");
    if (!real_gdpa) real_gdpa = (PFN_vkGetDeviceProcAddr)dlsym(RTLD_NEXT, "vkGetDeviceProcAddr");
    if (!real_create_device) real_create_device = (PFN_vkCreateDevice)dlsym(RTLD_NEXT, "vkCreateDevice");
    if (!real_alloc_memory) real_alloc_memory = (PFN_vkAllocateMemory)dlsym(RTLD_NEXT, "vkAllocateMemory");
    if (!real_get_mem_props)
        real_get_mem_props = (PFN_vkGetPhysicalDeviceMemoryProperties)dlsym(RTLD_NEXT, "vkGetPhysicalDeviceMemoryProperties");
}

PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    resolve();
    if (strcmp(name, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (strcmp(name, "vkCreateDevice") == 0) return (PFN_vkVoidFunction)vkCreateDevice;
    if (strcmp(name, "vkAllocateMemory") == 0) return (PFN_vkVoidFunction)vkAllocateMemory;
    return real_gipa(instance, name);
}

PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice device, const char *name) {
    resolve();
    if (strcmp(name, "vkAllocateMemory") == 0) return (PFN_vkVoidFunction)vkAllocateMemory;
    return real_gdpa(device, name);
}

VkResult vkCreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, const VkAllocationCallbacks *a,
                        VkDevice *out) {
    resolve();
    VkResult r = real_create_device(pd, ci, a, out);
    if (r == VK_SUCCESS) {
        pthread_mutex_lock(&g_mutex);
        for (int i = 0; i < MAX_DEVICES; i++) {
            if (!g_devices[i].valid) {
                g_devices[i].device = *out;
                if (real_get_mem_props) real_get_mem_props(pd, &g_devices[i].mem_props);
                g_devices[i].valid = 1;
                break;
            }
        }
        pthread_mutex_unlock(&g_mutex);
    }
    return r;
}

VkResult vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo *info, const VkAllocationCallbacks *a,
                          VkDeviceMemory *mem) {
    resolve();
    int device_local = 1;
    pthread_mutex_lock(&g_mutex);
    for (int i = 0; i < MAX_DEVICES; i++) {
        if (g_devices[i].valid && g_devices[i].device == device) {
            uint32_t heap = g_devices[i].mem_props.memoryTypes[info->memoryTypeIndex].heapIndex;
            device_local = (g_devices[i].mem_props.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_mutex);
    if (device_local && armed()) {
        fprintf(stderr, "vkoom: denying device-local allocation of %llu bytes -> VK_ERROR_OUT_OF_DEVICE_MEMORY\n",
                (unsigned long long)info->allocationSize);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return real_alloc_memory(device, info, a, mem);
}
