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


#ifndef QMCPLUSPLUS_SYCLDEVICEMANAGER_H
#define QMCPLUSPLUS_SYCLDEVICEMANAGER_H

#include "config.h"
#include <vector>
#include <memory>
#include <sycl/sycl.hpp>
#include <omp.h>

namespace qmcplusplus
{

/** SYCL device manager
 */
class SYCLDeviceManager
{
  int sycl_default_device_num = -1;
  static sycl::queue* default_device_queue;
  std::vector<sycl::device> visible_devices;

  std::vector<sycl::queue*> queue_pool;

public:

  SYCLDeviceManager() = delete;
  SYCLDeviceManager(int& default_device_num, int& num_devices, int local_rank, int local_size);
  ~SYCLDeviceManager();

  /** access the the DeviceManager owned default queue.
   * Restrict the use of it to performance non-critical operations.
   * Note: CUDA has a default queue but all the SYCL queues are explicit.
   */
  static sycl::queue& getDefaultDeviceDefaultQueue();
  sycl::queue createQueueDefaultDevice() const;
};

} // namespace qmcplusplus

#endif
