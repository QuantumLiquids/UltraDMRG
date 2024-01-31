// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-06 
*
* Description: QuantumLiquids/UltraDMRG project. Initialize for two-site update finite size vMPS with MPI Parallel, master side.
*/

/**
 @file   two_site_update_finite_vmps_init.h
 @brief  Initilization for two-site update finite size vMPS with MPI Paralization.
         0. include an overall initial function cover all (at least most) the functions in this file;
         1. Find the left/right boundaries, only between which the tensors need to be update.
            Also make sure the bond dimensions of tensors out of boundaries are sufficient large.
            Move the centre on the left_boundary+1 site (Assuming the before the centre <= left_boundary+1)
         2. Check if .temp exsits, if exsits, check if temp tensors are complete. 
            if one of above if is not, regenerate the environment.
         3. Check if QN sector numbers are enough. (Not do, will deal in tensor contraction functions);
         4. Generate the environment of boundary tensors
         5. Optional function: check if different processors read/write the same disk
*/

#ifndef QLMPS_ALGO_MPI_VMPS_VMPS_MPI_INIT_H
#define QLMPS_ALGO_MPI_VMPS_VMPS_MPI_INIT_H

#include <map>
#include "qlten/qlten.h"
#include "qlmps/one_dim_tn/mps_all.h"
#include "qlmps/algorithm/finite_vmps_sweep_params.h"
#include "qlmps/algo_mpi/vmps/two_site_update_finite_vmps_mpi.h"
#include "boost/mpi.hpp"
#include "qlmps/algorithm/vmps/vmps_init.h"                        // CheckAndUpdateBoundaryMPSTensors...

namespace qlmps {
using namespace qlten;
namespace mpi = boost::mpi;

//forward declarition
template<typename TenElemT, typename QNT>
std::pair<size_t, size_t> CheckAndUpdateBoundaryMPSTensors(
    FiniteMPS<TenElemT, QNT> &,
    const std::string &,
    const size_t
);

template<typename TenElemT, typename QNT>
void UpdateBoundaryEnvs(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const std::string mps_path,
    const std::string temp_path,
    const size_t left_boundary,
    const size_t right_boundary,
    const size_t update_site_num
);

inline bool NeedGenerateRightEnvs(
    const size_t N, //mps size
    const size_t left_boundary,
    const size_t right_boundary,
    const std::string &temp_path
);

template<typename TenElemT, typename QNT>
void InitEnvsMaster(
    FiniteMPS<TenElemT, QNT> &,
    const MPO<QLTensor<TenElemT, QNT>> &,
    const std::string,
    const std::string,
    const size_t,
    mpi::communicator &
);

template<typename TenElemT, typename QNT>
std::pair<size_t, size_t> TwoSiteFiniteVMPSInit(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    mpi::communicator world) {

  assert(world.rank() == 0);
  std::cout << "\n";
  std::cout << "=====> Two-Site MPI Update Sweep Parameters <=====" << "\n";
  std::cout << "MPS/MPO size: \t " << mpo.size() << "\n";
  std::cout << "Sweep times: \t " << sweep_params.sweeps << "\n";
  std::cout << "Bond dimension: \t " << sweep_params.Dmin << "/" << sweep_params.Dmax << "\n";
  std::cout << "Truncation error: \t " << sweep_params.trunc_err << "\n";
  std::cout << "Lanczos max iterations \t" << sweep_params.lancz_params.max_iterations << "\n";
  std::cout << "MPS path: \t" << sweep_params.mps_path << "\n";
  std::cout << "Temp path: \t" << sweep_params.temp_path << std::endl;

  std::cout << "=====> Technical Parameters <=====" << "\n";
  std::cout << "The number of processors(including master): \t" << world.size() << "\n";
  std::cout << "The number of threads per processor: \t" << hp_numeric::GetTensorManipulationThreads() << "\n";

  std::cout << "=====> Checking and updating boundary tensors =====>" << std::endl;
  using Tensor = QLTensor<TenElemT, QNT>;
  auto [left_boundary, right_boundary] = CheckAndUpdateBoundaryMPSTensors(
      mps,
      sweep_params.mps_path,
      sweep_params.Dmax
  );


  //check qumber sct numbers, > 2*slave number, can omp or mpi parallel
  // A best scheme is to write a more robust contraction
  /*
  for(size_t i = left_boundary; i < right_boundary; i++){

  }
  */

  if (NeedGenerateRightEnvs(
      mpo.size(),
      left_boundary,
      right_boundary,
      sweep_params.temp_path)
      ) {
  std::cout << "=====> Creating the environment tensors =====>" << std::endl;
  MasterBroadcastOrder(init_grow_env, world);
  InitEnvsMaster(mps, mpo, sweep_params.mps_path, sweep_params.temp_path, left_boundary + 2, world);
  } else {
    std::cout << "The environment tensors have existed." << std::endl;
  }

  //update the left env of left_boundary site and right env of right_boundary site
  UpdateBoundaryEnvs(mps, mpo, sweep_params.mps_path,
                     sweep_params.temp_path, left_boundary, right_boundary, 2);
  return std::make_pair(left_boundary, right_boundary);
}

/** Generate the environment tensors before the first sweep
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param mps
 * @param mpo
 * @param mps_path
 * @param temp_path
 * @param update_site_num
 */
template<typename TenElemT, typename QNT>
void InitEnvsMaster(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const std::string mps_path,
    const std::string temp_path,
    const size_t update_site_num,
    mpi::communicator &world
) {
  using TenT = QLTensor<TenElemT, QNT>;
  auto N = mps.size();

  TenT renv;
  //Write a trivial right environment tensor to disk
  mps.LoadTen(N - 1, GenMPSTenName(mps_path, N - 1));
  auto mps_trivial_index = mps.back().GetIndexes()[2];
  auto mpo_trivial_index_inv = InverseIndex(mpo.back().GetIndexes()[3]);
  auto mps_trivial_index_inv = InverseIndex(mps_trivial_index);
  renv = TenT({mps_trivial_index, mpo_trivial_index_inv, mps_trivial_index_inv});
  renv({0, 0, 0}) = 1;
  auto file = GenEnvTenName("r", 0, temp_path);
  WriteQLTensorTOFile(renv, file);

  //bulk right environment tensors
  for (size_t i = 1; i <= N - update_site_num; ++i) {
    if (i > 1) { mps.LoadTen(N - i, GenMPSTenName(mps_path, N - i)); }
    auto file = GenEnvTenName("r", i, temp_path);
    MasterBroadcastOrder(init_grow_env_grow, world);
    TenT *prenv_next = MasterGrowRightEnvironmentInit(renv, mpo[N - i], mps[N - i], world);
    renv = std::move(*prenv_next);
    delete prenv_next;
    WriteQLTensorTOFile(renv, file);
    mps.dealloc(N - i);
  }

  //Write a trivial left environment tensor to disk
  TenT lenv;
  mps.LoadTen(0, GenMPSTenName(mps_path, 0));
  mps_trivial_index = mps.front().GetIndexes()[0];
  mpo_trivial_index_inv = InverseIndex(mpo.front().GetIndexes()[0]);
  mps_trivial_index_inv = InverseIndex(mps_trivial_index);
  lenv = TenT({mps_trivial_index, mpo_trivial_index_inv, mps_trivial_index_inv});
  lenv({0, 0, 0}) = 1;
  file = GenEnvTenName("l", 0, temp_path);
  WriteQLTensorTOFile(lenv, file);
  mps.dealloc(0);

  assert(mps.empty());
  MasterBroadcastOrder(init_grow_env_finish, world);
}

}//qlmps
#endif
