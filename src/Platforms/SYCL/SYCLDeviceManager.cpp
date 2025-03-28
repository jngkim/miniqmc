//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Thomas Applencourt, apl@anl.gov, Argonne National Laboratory
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Thomas Applencourt, apl@anl.gov, Argonne National Laboratory
//
//////////////////////////////////////////////////////////////////////////////////////


#include "SYCLDeviceManager.h"
#include <stdexcept>
#include <string>
#include <algorithm>
#include "config.h"
#include "Platforms/Host/OutputManager.h"
#include "Platforms/Host/determineDefaultDeviceNum.h"

#define USE_QUEUE_POOL

namespace qmcplusplus
{

  sycl::queue* SYCLDeviceManager::default_device_queue = nullptr;

  SYCLDeviceManager::SYCLDeviceManager(int& default_device_num, int& num_devices, int local_rank, int local_size)
  {
#ifdef ENABLE_OFFLOAD
    //this could be handled by ENV
    const int sycl_device_count=omp_get_num_devices();
    sycl_default_device_num = determineDefaultDeviceNum(sycl_device_count, local_rank, local_size);

    visible_devices.resize(sycl_device_count);
    { //create interop only for the default device
      int d = sycl_default_device_num;
      {
        omp_interop_t interop;
#pragma omp interop device(sycl_default_device_num) init(prefer_type("sycl"),targetsync: interop)

        int result;
        sycl::queue* omp_queue = static_cast<sycl::queue *>(omp_get_interop_ptr(interop, omp_ipr_targetsync, &result));
        if(result != omp_irc_success)
          throw std::runtime_error("SYCLDeviceManager::SYCLDeviceManager fail to obtain sycl::queue by interop");

        default_device_queue = omp_queue;
      } 
    }
#else
    // Potentially multiple GPU platform.
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    if (platforms.empty())
      throw std::runtime_error("Cannot find SYCL platforms!");

    // find out devices from the first platform with GPUs.
    std::vector<sycl::device> devices;
    app_log() << "Visible SYCL platforms are :" << std::endl;
    for (auto& platform : platforms)
    {
      std::vector<sycl::device> gpu_devices = platform.get_devices(sycl::info::device_type::gpu);
      const auto gpu_count                  = gpu_devices.size();
      bool selected                         = false;
      if (devices.empty() && gpu_count > 0)
      {
        selected = true;
        devices  = std::move(gpu_devices);
      }
      app_log() << (selected ? " ** " : "    ") << platform.get_info<sycl::info::platform::name>() << " with "
        << gpu_count << " GPUs." << std::endl;
    }
    app_log() << std::endl;

    sycl_default_device_num = determineDefaultDeviceNum(devices.size(), local_rank, local_size);

    visible_devices=devices;
    default_device_queue = new sycl::queue(devices[sycl_default_device_num]);
#endif

    if (num_devices > 0)
    {
      if (default_device_num < 0)
        default_device_num = sycl_default_device_num;
      else if (default_device_num != sycl_default_device_num)
        throw std::runtime_error("Inconsistent assigned SYCL devices with the previous record!");

    }

#ifdef USE_QUEUE_POOL
    if(queue_pool.empty())
    {
      queue_pool.resize(omp_get_max_threads());
      for(int i=0; i<queue_pool.size(); ++i)
        queue_pool[i] = 
          new sycl::queue{default_device_queue->get_context(), default_device_queue->get_device(), 
                          sycl::property::queue::in_order()};
    }
#endif
  }

  SYCLDeviceManager::~SYCLDeviceManager()
  {
  }

sycl::queue& SYCLDeviceManager::getDefaultDeviceDefaultQueue()
{
  if (default_device_queue == nullptr)
  throw std::runtime_error("SYCLDeviceManager::getDefaultDeviceQueue() the global instance not initialized.");
  return *default_device_queue;
 }

sycl::queue SYCLDeviceManager::createQueueDefaultDevice() const
{ // copy
  if (default_device_queue == nullptr)
  throw std::runtime_error("SYCLDeviceManager::getDefaultDeviceQueue() the global instance not initialized.");
#ifdef USE_QUEUE_POOL
  int iq = omp_get_thread_num() % queue_pool.size();
  return sycl::queue{*queue_pool[iq]};
#else
  return sycl::queue{default_device_queue->get_context(), default_device_queue->get_device(),
                     sycl::property::queue::in_order()};
#endif
}

} // namespace qmcplusplus
