#include <vulkan/vulkan.h>
#include <iostream>

int main() {
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::cout << "Vulkan fully operational " << extensionCount << std::endl;
    return 0;
}
