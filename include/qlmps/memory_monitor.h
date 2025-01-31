/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-31
*
* Description: QuantumLiquids/UltraDMRG project. Memory monitor.
*/


#ifndef QLMPS_MEMORY_MONITOR_H
#define QLMPS_MEMORY_MONITOR_H

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#ifdef USE_GPU
#include <cuda_runtime.h>
namespace qlmps{
class MemoryMonitor {
 public:
  explicit MemoryMonitor(int device_id = 0) : device_id_(device_id) {
    // Set and verify device
    cudaError_t err = cudaSetDevice(device_id_);
    checkCudaError(err, "Failed to set device");

    // Get current device ID (verify actual device)
    err = cudaGetDevice(&device_id_);
    checkCudaError(err, "Failed to get device");

    // Get device properties
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device_id_);
    checkCudaError(err, "Failed to get device properties");
    device_name_ = props.name;
  }

  // Get memory information in GB
  void GetMemoryInfo(double& free_gb, double& used_gb, double& total_gb) {
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    checkCudaError(err, "Failed to query CUDA memory");

    const double GB = 1024.0 * 1024 * 1024;
    free_gb = free / GB;
    total_gb = total / GB;
    used_gb = (total - free) / GB;
  }

  // Display formatted checkpoint
  void Checkpoint(const std::string& message) {
    double free, used, total;
    GetMemoryInfo(free, used, total);

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %X") << "] "
       << "Device " << device_id_ << " (" << device_name_ << ") - " << message << "\n"
       << std::fixed << std::setprecision(2)
       << "  Used: " << used << " GB, Free: " << free << " GB, Total: " << total << " GB\n";

    std::cout << ss.str();
  }

  // Static method to show available devices
  static void ListAvailableDevices() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    checkCudaError(err, "Failed to get device count");

    std::cout << "Available CUDA Devices:\n";
    for(int i = 0; i < device_count; ++i) {
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, i);
      std::cout << "Device " << i << ": " << props.name << "\n";
    }
  }

  // Get current device ID
  int GetDeviceId() const { return device_id_; }

 private:
  int device_id_;
  std::string device_name_;

  static void checkCudaError(cudaError_t err, const std::string& msg) {
    if(err != cudaSuccess) {
      throw std::runtime_error(msg + ": " + cudaGetErrorString(err));
    }
  }
};
}//qlmps

#else //CPU code
#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/vm_statistics.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#endif
namespace qlmps {

class MemoryMonitor {
 public:
  // Device ID parameter kept for API compatibility (ignored for CPU)
  explicit MemoryMonitor(int device_id = 0) {
    // Could add validation if needed
  }

  // Get memory information in GB
  void GetMemoryInfo(double &free_gb, double &used_gb, double &total_gb) {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        if(!GlobalMemoryStatusEx(&memInfo)) {
            throw std::runtime_error("Failed to get system memory info");
        }
        total_gb = memInfo.ullTotalPhys / GB_;
        free_gb = memInfo.ullAvailPhys / GB_;
        used_gb = (memInfo.ullTotalPhys - memInfo.ullAvailPhys) / GB_;
#elif defined(__linux__)
    struct sysinfo memInfo;
        if(sysinfo(&memInfo) != 0) {
            throw std::runtime_error("Failed to get system memory info");
        }
        total_gb = memInfo.totalram * memInfo.mem_unit / GB_;
        free_gb = (memInfo.freeram + memInfo.bufferram) * memInfo.mem_unit / GB_;
        used_gb = total_gb - free_gb;
#elif defined(__APPLE__)
    // Total memory
    uint64_t total_mem;
    size_t size = sizeof(total_mem);
    if (sysctlbyname("hw.memsize", &total_mem, &size, NULL, 0) != 0) {
      throw std::runtime_error("Failed to get total memory");
    }
    total_gb = total_mem / GB_;

    // Free memory
    vm_size_t page_size;
    mach_port_t mach_port;
    mach_msg_type_number_t count;
    vm_statistics64_data_t vm_stats;

    mach_port = mach_host_self();
    count = sizeof(vm_stats) / sizeof(natural_t);
    if (KERN_SUCCESS != host_page_size(mach_port, &page_size) ||
        KERN_SUCCESS != host_statistics64(mach_port, HOST_VM_INFO,
                                          (host_info64_t) &vm_stats, &count)) {
      throw std::runtime_error("Failed to get memory statistics");
    }

    free_gb = (vm_stats.free_count + vm_stats.inactive_count) * page_size / GB_;
    used_gb = total_gb - free_gb;
#endif
  }

  // Display formatted checkpoint
  void Checkpoint(const std::string &message) {
    double free, used, total;
    GetMemoryInfo(free, used, total);

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %X") << "] "
       << "System Memory - " << message << "\n"
       << std::fixed << std::setprecision(2)
       << "  Used: " << used << " GB, Free: " << free << " GB, Total: " << total << " GB\n";

    std::cout << ss.str();
  }

  // Static method to show system memory summary
  static void ListAvailableDevices() {
    double total, used, free;
    MemoryMonitor temp;
    temp.GetMemoryInfo(free, used, total);

    std::cout << "System Memory Summary:\n"
              << std::fixed << std::setprecision(2)
              << "Total: " << total << " GB\n"
              << "Available: " << free << " GB\n"
              << "Currently Used: " << used << " GB\n";
  }

  // Maintain API compatibility (always returns 0 for CPU)
  int GetDeviceId() const { return 0; }

 private:
  static constexpr double GB_ = 1024.0 * 1024 * 1024;
};

}//qlmps
#endif

#endif //QLMPS_MEMORY_MONITOR_H
