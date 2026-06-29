import pynvml


class GPUTracker:
    def __enter__(self):
        pynvml.nvmlInit()

    def __exit__(self):
        pynvml.nvmlShutdown()


if __name__=="__main__":
    # Initialize the library
    pynvml.nvmlInit()

    # Get the number of GPUs
    deviceCount = pynvml.nvmlDeviceGetCount()
    for i in range(deviceCount):
        # Get a handle to the GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        # Query various metrics
        name = pynvml.nvmlDeviceGetName(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # GPU compute utilization
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)    # Memory usage
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) # Power draw in milliwatts
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        # Print the information
        print(f"Device {i}: {name}")
        print(f"  GPU Utilization: {util.gpu}%")
        print(f"  Memory Utilization: {util.memory}%")
        print(f"  Memory Used: {mem_info.used / 1024**2:.2f} MB / {mem_info.total / 1024**2:.2f} MB")
        print(f"  Power Draw: {power_usage / 1000:.2f} W")
        print(f"  Temperature: {temperature}°C")
        print()

    # Shutdown when done
    pynvml.nvmlShutdown()
