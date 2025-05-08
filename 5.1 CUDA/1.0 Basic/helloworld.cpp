#include <iostream>
#include <nvml.h> // NVIDIA Management Library header
using namespace std;

int main()
{
    cout << "This is a hello world program" << endl;
    cout << "GPU information:" << endl;

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS)
    {
        cout << "Failed to initialize NVML: " << nvmlErrorString(result) << endl;
        return 1;
    }

    // Get device count
    unsigned int deviceCount = 0;
    result = nvmlDeviceGetCount(&deviceCount);
    if (result != NVML_SUCCESS)
    {
        cout << "Failed to get device count: " << nvmlErrorString(result) << endl;
        nvmlShutdown();
        return 1;
    }

    cout << "Number of GPUs: " << deviceCount << endl;

    // Clean up
    nvmlShutdown();

    return 0;
}