//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#include <memory>
#include <vector>
#include <iostream>
#include <random>
#include "Numerics/OhmmsPETE/OhmmsVector.h"
#include "Numerics/OhmmsPETE/OhmmsMatrix.h"
#include "DeviceManager.h"
#include "Platforms/SYCL/syclBLAS.hpp"
#include "Platforms/CPU/BLAS.hpp"
#include "QMCWaveFunctions/syclSolverInverter.hpp"

#include "Utilities/Communicate.h"
#include "Host/OutputManager.h"
#include "mkl_lapacke.h"
#include "Platforms/SYCL/SYCLruntime.hpp"
#include <fstream>
#include <omp.h>
#include <mpi.h>
#include <iomanip>

char pname[MPI_MAX_PROCESSOR_NAME];
std::ofstream fout;
std::vector<double> timers(8,0.0);

namespace qmcplusplus
{

inline float xlange( const char ochar, lapack_int m, lapack_int n, const float* a, lapack_int lda, float* work )
{
  return LAPACKE_slange_work(LAPACK_COL_MAJOR,ochar,m,n,a,lda,work);
}

inline double xlange( const char ochar, lapack_int m, lapack_int n, const double* a, lapack_int lda, double* work )
{
  return LAPACKE_dlange_work(LAPACK_COL_MAJOR,ochar,m,n,a,lda,work);
}


template<typename T, typename DigEng>
void debug_inverse(const std::int64_t M, int niters)
{
  static int ncalled=0;

  int np = omp_get_max_threads();
  std::vector<double> inv_time(np*niters);
#pragma omp parallel
  {
    sycl::queue m_queue{createSYCLInOrderQueueOnDefaultDevice()};

    Matrix<T> A(M,M); 
    Matrix<T> B(M,M); //for validation

    int ip = omp_get_thread_num();

    std::mt19937 rng(ip);
    //std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
    std::normal_distribution<T> udist{T(0), T(1)}; 
    std::generate_n(B.data(),B.size(),[&]() { return udist(rng);});
    std::copy_n(B.data(), B.size(), A.data());

    DigEng diag_eng;

    Matrix<T> Ainv;
    Matrix<T,SYCLAllocator<T>> Ainv_gpu;
    Ainv.resize(M,M);
    Ainv_gpu.resize(M,M);
    //check the identity

    std::complex<double> log_value;
    constexpr T cone(1);
    Matrix<T> C(M,M);

    sycl::ext::oneapi::experimental::prepare_for_device_copy(A.data(), A.size()*sizeof(T), m_queue);
    sycl::ext::oneapi::experimental::prepare_for_device_copy(Ainv.data(), Ainv.size()*sizeof(T), m_queue);

    for(int iter=0; iter < niters; ++iter)
    {
      auto start = std::chrono::high_resolution_clock::now();
      diag_eng.invert_transpose(A, Ainv, Ainv_gpu, log_value, m_queue);
      auto end = std::chrono::high_resolution_clock::now();
      inv_time[ip + iter*np] = static_cast<std::chrono::duration<double>>(end-start).count();
      //m_queue.wait();

      BLAS::gemm('T', 'N', M, M, M, cone, B.data(), M, Ainv.data(), M, T{}, C.data(),M);

      auto norm  = xlange('O', M, M, C.data(), M, Ainv.data());

      //fout << iter << " " << ip << " norm = " << norm << std::endl;

      std::generate_n(B.data(),B.size(),[&]() { return udist(rng);});
      std::copy_n(B.data(), B.size(), A.data());
    }

    sycl::ext::oneapi::experimental::release_from_device_copy(Ainv.data(), m_queue);
    sycl::ext::oneapi::experimental::release_from_device_copy(A.data(), m_queue);
  }

  for(int iter=0; iter < niters; ++iter)
  {
    double sum = 0.0, min = 1000, max = 0;
    for(int ip = 0, iloc=iter*np; ip < np; ++ip, ++iloc)
    {
      sum += inv_time[iloc];
      min = std::min(min,inv_time[iloc]);
      max = std::max(max,inv_time[iloc]);
    }
    fout << "Iter" << ncalled << " " << iter << " " << std::setprecision(3) << sum/np << " " << min << " " << max << "\n";
  }
  ncalled++;
}

template<typename T, typename DiagEng>
struct SyclDetUpdate {
    Matrix<T> A;
    Matrix<T> Ainv;
    Matrix<T,SYCLAllocator<T>> Ainv_gpu;

    DiagEng diag_eng;
    sycl::queue m_queue;
    std::mt19937 rng;
    std::normal_distribution<T> udist;

    SyclDetUpdate(const sycl::queue& q):m_queue(q), rng(911), udist{T(0), T(1)} {}

    ~SyclDetUpdate()
    {
      sycl::ext::oneapi::experimental::release_from_device_copy(Ainv.data(), m_queue);
      sycl::ext::oneapi::experimental::release_from_device_copy(A.data(), m_queue);
    }

    void resize(int M)
    {
      Ainv_gpu.resize(M,M);
      A.resize(M,M);
      Ainv.resize(M,M);
      sycl::ext::oneapi::experimental::prepare_for_device_copy(A.data(), A.size()*sizeof(T), m_queue);
      sycl::ext::oneapi::experimental::prepare_for_device_copy(Ainv.data(), Ainv.size()*sizeof(T), m_queue);
    }

    void solve()
    {
      std::complex<double> log_value;
      std::generate_n(A.data(),A.size(),[&]() { return udist(rng);});
      diag_eng.invert_transpose(A, Ainv, Ainv_gpu, log_value, m_queue);
    }
};

template<typename T, typename DetEng>
void debug_inverse_mpi(Communicate& comm, const std::int64_t M, int niters)
{
  static int ncalled = 0;
  const int np = omp_get_max_threads();
  using DetUpdate_t = SyclDetUpdate<T,DetEng>;

  std::vector<DetUpdate_t*> detEng(np, nullptr);

  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
  for(int ip = 0; ip < np; ++ip)
  {
    //detEng[ip] = new DetUpdate_t(getSYCLInOrderQueueOnDefaultDevice());
    detEng[ip] = new DetUpdate_t(createSYCLInOrderQueueOnDefaultDevice());
    detEng[ip]->resize(M);
  }
  auto end = std::chrono::high_resolution_clock::now();
  comm.barrier();
  auto end_mpi = std::chrono::high_resolution_clock::now();

  timers[0] = static_cast<std::chrono::duration<double>>(end-start).count();
  timers[1] = static_cast<std::chrono::duration<double>>(end_mpi-start).count() - timers[0];

  start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
  for(int ip = 0; ip < np; ++ip)
  {
    detEng[ip]->solve();
  }
  end = std::chrono::high_resolution_clock::now();
  comm.barrier();
  end_mpi = std::chrono::high_resolution_clock::now();

  timers[2] = static_cast<std::chrono::duration<double>>(end-start).count();
  timers[3] = static_cast<std::chrono::duration<double>>(end_mpi-start).count() - timers[2];

  start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
  for(int ip = 0; ip < np; ++ip)
  {
    for(int iter=0; iter < niters; ++iter)
      detEng[ip]->solve();
  }
  end = std::chrono::high_resolution_clock::now();
  comm.barrier();
  end_mpi = std::chrono::high_resolution_clock::now();

  timers[4] = static_cast<std::chrono::duration<double>>(end-start).count()/niters;
  timers[5] = static_cast<std::chrono::duration<double>>(end_mpi-start).count()/niters - timers[4];

#if defined(EBUG)
  for(int t = 0; t < timers.size(); ++t)
    fout << "," << timers[t];
  fout << "\n";
#endif

  for(int ip = 0; ip < np ; ++ip)
    delete detEng[ip];

  ncalled++;
}

} //qmcplusplus

int main(int argc, char* argv[])
{
  int blocks=1;
  int steps=2;
  int seq_id=0;

  int M = 3072;
  if(argc>1)
    M = atoi(argv[1]);
  if(argc>2)
    blocks = atoi(argv[2]);
  if(argc>3)
    seq_id = atoi(argv[3]);

  using namespace qmcplusplus;

  Communicate comm(argc, argv);
  DeviceManager::initializeGlobalDeviceManager(comm.rank(), comm.size());

  if (!comm.root())
  {
    outputManager.shutOff();
  }

  int nnodes = comm.size();

  int pname_len=0;
  int res = MPI_Get_processor_name(pname, &pname_len);
  std::vector<char> pname_all(nnodes*pname_len);
  int ierr = MPI_Gather(pname, pname_len, MPI_CHAR, pname_all.data(), pname_len, MPI_CHAR, 0, MPI_COMM_WORLD);

  std::vector<double> global_timers(timers.size()*nnodes);

  comm.barrier();
  for(int b=0;b < blocks; ++b)
  {
    auto start = std::chrono::high_resolution_clock::now();
    debug_inverse_mpi<float,syclSolverInverter<double>>(comm,M,steps);
    auto end = std::chrono::high_resolution_clock::now();
    comm.barrier();
    auto end_mpi = std::chrono::high_resolution_clock::now();

    timers[6] = static_cast<std::chrono::duration<double>>(end-start).count();
    timers[7] = static_cast<std::chrono::duration<double>>(end_mpi-start).count()-timers[6];

    int root = 0;
    ierr = MPI_Gather(timers.data(), 8, MPI_DOUBLE, global_timers.data(), 8, MPI_DOUBLE, root, MPI_COMM_WORLD);

    if(comm.rank() == root)
    {
      std::ofstream main_fout;
      char fname[256];
      sprintf(fname, "solver.n%d.o%d.s%d.b%d.csv", comm.rank(), omp_get_max_threads(), seq_id, b);
      main_fout.open(fname);
      main_fout <<"node,rank,init_n,init_w,first_n,first_w,main_n,main_w,tot_n,tot_w\n";
      const char* pname_ptr = pname_all.data();
      for(int node = 0, lt = 0; node < nnodes; ++node)
      {
        std::string_view anode(pname_ptr, pname_len);
        main_fout << anode <<"," << node;
        for(int t = 0; t < timers.size(); ++t, ++lt)
          main_fout << "," << global_timers[lt];
        main_fout << "\n";
        pname_ptr += pname_len;
      }
      main_fout.close();

      const int first_timer = 2; // DO NOT CHANGE
      double first_sum=0.0, first_max=0.0, first_min=100000;
      for(int node = 0, offset=first_timer; node < nnodes; ++node, offset += timers.size())
      {
        first_sum += global_timers[offset];
        first_max = std::max(first_max, global_timers[offset]);
        first_min = std::min(first_min, global_timers[offset]);
      }
      qmcplusplus::app_log() << "Startup " << comm.rank() << " " << omp_get_max_threads() << " " << seq_id << " " << b <<  " " 
        << first_sum/nnodes << " " << first_min << " " << first_max << std::endl;
    }
    comm.barrier();
  }

  fout.close();

  return 0;
}
