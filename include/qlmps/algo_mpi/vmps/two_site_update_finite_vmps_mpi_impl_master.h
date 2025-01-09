// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-11
*
* Description: QuantumLiquids/UltraDMRG project. Two-site update finite size vMPS with MPI Parallel, master node.
*/

/**
@file two_site_update_finite_vmps_mpi.h
@brief Two-site update finite size vMPS with MPI Paralization
*/

#ifndef QLMPS_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_IMPLY_H
#define QLMPS_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_IMPLY_H

#include "qlten/qlten.h"
#include "qlmps/algorithm/lanczos_params.h"                         //LanczosParams
#include "qlmps/algorithm/finite_vmps_sweep_params.h"
#include "qlmps/algo_mpi/mps_algo_order.h"                          //VMPSORDER
#include "qlmps/algo_mpi/env_ten_update_master.h"                   //MasterGrowLeftEnvironment, MasterGrowRightEnvironment
#include "qlmps/algo_mpi/vmps/vmps_mpi_init_master.h"               //MPI vmps initial
#include "qlmps/algo_mpi/vmps/two_site_update_finite_vmps_mpi.h"    //TwoSiteMPIVMPSSweepParams
#include "lanczos_solver_mpi_master.h"                              //MPI Lanczos solver

namespace qlmps {
using namespace qlten;

//forward declaration
template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPS(const MPI_Comm &);

template<typename TenElemT, typename QNT>
inline QLTEN_Double TwoSiteFiniteVMPSWithNoise_(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const MPI_Comm &comm
);

/**
Function to perform two-site update finite vMPS algorithm with MPI parallelization.
  
  @example 
  Using the API in the following way:
  The starting point of `main()` should look like
  ```
      MPI_Init(..)
      MPI_Comm comm =     MPI_Comm_rank;
  ```

  ```
    double e0 = TwoSiteFiniteVMPSWithNoise(mps, mpo, sweep_params, comm);
  ```
  However, except `comm`, variables are only valid in master processor,
  inputs of other processor(s) can be
  arbitrary (Of course the types should be right). Outputs of slave(s)
  are all 0.0. 

  @note  The input MPS will be considered an empty one.
         The true data has be writed into disk.
  @note  The canonical center of input MPS should be set <=left_boundary+1.
        The canonical center of output MPS will move to left_boundary+1.
*/
template<typename TenElemT, typename QNT>
inline QLTEN_Double TwoSiteFiniteVMPS(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const MPI_Comm &comm
) {
  QLTEN_Double e0(0.0);
  int mpi_size, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  if (mpi_size == 1) {
    return TwoSiteFiniteVMPS(mps, mpo, sweep_params);
  }
  if (sweep_params.noise_valid) {
    return TwoSiteFiniteVMPSWithNoise_(mps, mpo, sweep_params, comm);
  }
  if (rank == kMPIMasterRank) {
    e0 = MasterTwoSiteFiniteVMPS(mps, mpo, sweep_params, comm);
  } else {
    SlaveTwoSiteFiniteVMPS<TenElemT, QNT>(mpo, comm);
  }
  return e0;
}

template<typename TenElemT, typename QNT>
QLTEN_Double MasterTwoSiteFiniteVMPS(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const MPI_Comm &comm
) {
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  assert(rank == kMPIMasterRank); //only master can call this function
  assert(mps.size() == mpo.size());

  MasterBroadcastOrder(program_start, kMPIMasterRank, comm);
  for (size_t node = 1; node < mpi_size; node++) {
    size_t node_num;
    hp_numeric::MPI_Recv(node_num, node, 2 * node, comm);
    if (node_num == node) {
      std::cout << "Node " << node << " received the program start order." << std::endl;
    } else {
      std::cout << "unexpected " << std::endl;
      exit(1);
    }
  }
  auto [left_boundary, right_boundary] = TwoSiteFiniteVMPSInit(mps, mpo, sweep_params, comm);
  double e0(0.0);
  mps.LoadTen(left_boundary + 1, GenMPSTenName(sweep_params.mps_path, left_boundary + 1));
  for (size_t sweep = 1; sweep <= sweep_params.sweeps; ++sweep) {
    std::cout << "sweep " << sweep << std::endl;
    Timer sweep_timer("sweep");
    e0 = TwoSiteFiniteVMPSSweep(mps, mpo, sweep_params,
                                left_boundary, right_boundary, comm);

    sweep_timer.PrintElapsed();
    std::cout << std::endl;
  }
  mps.DumpTen(left_boundary + 1, GenMPSTenName(sweep_params.mps_path, left_boundary + 1), true);
  MasterBroadcastOrder(program_final, kMPIMasterRank, comm);
  return e0;
}

/**
Function to perform a single two-site finite vMPS sweep.

@note Before the sweep and after the sweep, the MPS only contains mps[1].
*/
template<typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSSweep(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const size_t left_boundary,
    const size_t right_boundary,
    const MPI_Comm &comm
) {
  auto N = mps.size();
  using TenT = QLTensor<TenElemT, QNT>;
  TenVec<TenT> lenvs(N - 1);
  TenVec<TenT> renvs(N - 1);
  double e0;

  for (size_t i = left_boundary; i <= right_boundary - 2; ++i) {
    // Load to-be-used tensors
    LoadRelatedTensOnTwoSiteAlgWhenRightMoving(mps, lenvs, renvs, i, left_boundary, sweep_params);
    e0 = MasterTwoSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'r', i, comm);
    // Dump related tensor to HD and remove unused tensor from RAM
    DumpRelatedTensOnTwoSiteAlgWhenRightMoving(mps, lenvs, renvs, i, sweep_params);
  }
  for (size_t i = right_boundary; i >= left_boundary + 2; --i) {
    LoadRelatedTensOnTwoSiteAlgWhenLeftMoving(mps, lenvs, renvs, i, right_boundary, sweep_params);
    e0 = MasterTwoSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'l', i, comm);
    DumpRelatedTensOnTwoSiteAlgWhenLeftMoving(mps, lenvs, renvs, i, sweep_params);
  }
  return e0;
}

template<typename TenElemT, typename QNT>
double MasterTwoSiteFiniteVMPSUpdate(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const char dir,
    const size_t target_site,
    const MPI_Comm &comm
) {
  //master
  Timer update_timer("two_site_fvmps_update");
#ifdef QLMPS_TIMING_MODE
  Timer initialize_timer("two_site_fvmps_setup_and_initial_state");
#endif
  // Assign some parameters
  auto N = mps.size();
  std::vector<std::vector<size_t>> init_state_ctrct_axes;
  size_t svd_ldims;
  size_t lsite_idx, rsite_idx;
  size_t lenv_len, renv_len;
  std::string lblock_file, rblock_file;
  init_state_ctrct_axes = {{2},
                           {0}};
  svd_ldims = 2;
  switch (dir) {
    case 'r':lsite_idx = target_site;
      rsite_idx = target_site + 1;
      lenv_len = target_site;
      renv_len = N - (target_site + 2);
      break;
    case 'l':lsite_idx = target_site - 1;
      rsite_idx = target_site;
      lenv_len = target_site - 1;
      renv_len = N - target_site - 1;
      break;
    default:std::cout << "dir must be 'r' or 'l', but " << dir << std::endl;
      exit(1);
  }

  // Lanczos
  using TenT = QLTensor<TenElemT, QNT>;
  std::vector<TenT *> eff_ham(4);
  eff_ham[0] = lenvs(lenv_len);
  // Safe const casts for MPO local tensors.
  eff_ham[1] = const_cast<TenT *>(&mpo[lsite_idx]);
  eff_ham[2] = const_cast<TenT *>(&mpo[rsite_idx]);
  eff_ham[3] = renvs(renv_len);

  auto init_state = new TenT;
  Contract(&mps[lsite_idx], &mps[rsite_idx], init_state_ctrct_axes, init_state);
#ifdef QLMPS_TIMING_MODE
  initialize_timer.PrintElapsed();
#endif
  Timer lancz_timer("two_site_fvmps_lancz");
  MasterBroadcastOrder(lanczos, kMPIMasterRank, comm);
  HANDLE_MPI_ERROR(::MPI_Bcast(&lsite_idx, 1, MPI_UNSIGNED_LONG_LONG, kMPIMasterRank, comm));
  auto lancz_res = MasterLanczosSolver(
      eff_ham, init_state,
      sweep_params.lancz_params,
      comm
  );
#ifdef QLMPS_TIMING_MODE
  auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif

  // SVD and measure entanglement entropy
#ifdef QLMPS_TIMING_MODE
  Timer svd_timer("two_site_fvmps_svd");
#endif

  TenT u, vt;
  using DTenT = QLTensor<QLTEN_Double, QNT>;
  DTenT s;
  QLTEN_Double actual_trunc_err;
  size_t D;
  MasterBroadcastOrder(svd, kMPIMasterRank, comm);
  MPISVDMaster(
      lancz_res.gs_vec,
      svd_ldims, Div(mps[lsite_idx]),
      sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
      &u, &s, &vt, &actual_trunc_err, &D,
      comm
  );
  delete lancz_res.gs_vec;
  auto ee = MeasureEE(s, D);

#ifdef QLMPS_TIMING_MODE
  svd_timer.PrintElapsed();
#endif

  // Update MPS local tensor
#ifdef QLMPS_TIMING_MODE
  Timer update_mps_ten_timer("two_site_fvmps_update_mps_ten");
#endif

  TenT the_other_mps_ten;
  switch (dir) {
    case 'r':mps[lsite_idx] = std::move(u);
      Contract(&s, &vt, {{1},
                         {0}}, &the_other_mps_ten);
      mps[rsite_idx] = std::move(the_other_mps_ten);
      break;
    case 'l':
      Contract(&u, &s, {{2},
                        {0}}, &the_other_mps_ten);
      mps[lsite_idx] = std::move(the_other_mps_ten);
      mps[rsite_idx] = std::move(vt);
      break;
    default:assert(false);
  }

#ifdef QLMPS_TIMING_MODE
  update_mps_ten_timer.PrintElapsed();
#endif

  // Update environment tensors
#ifdef QLMPS_TIMING_MODE
  Timer update_env_ten_timer("two_site_fvmps_update_env_ten");
#endif
  switch (dir) {
    case 'r': {
      MasterBroadcastOrder(growing_left_env, kMPIMasterRank, comm);
      lenvs(lenv_len + 1) = MasterGrowLeftEnvironment(mpo[target_site], mps[target_site], comm);
      /*
      TenT temp1, temp2, lenv_ten;
      Contract(&lenvs[lenv_len], &mps[target_site], {{0}, {0}}, &temp1);
      Contract(&temp1, &mpo[target_site], {{0, 2}, {0, 1}}, &temp2);
      auto mps_ten_dag = Dag(mps[target_site]);
      Contract(&temp2, &mps_ten_dag, {{0 ,2}, {0, 1}}, &lenv_ten);
      lenvs[lenv_len + 1] = std::move(lenv_ten);
      */
    }
      break;
    case 'l': {
      MasterBroadcastOrder(growing_right_env, kMPIMasterRank, comm);
      renvs(renv_len + 1) = MasterGrowRightEnvironment(mpo[target_site], mps[target_site], comm);
      /*
      TenT temp1, temp2, renv_ten;
      Contract(&mps[target_site], eff_ham[3], {{2}, {0}}, &temp1);
      Contract(&temp1, &mpo[target_site], {{1, 2}, {1, 3}}, &temp2);
      auto mps_ten_dag = Dag(mps[target_site]);
      Contract(&temp2, &mps_ten_dag, {{3, 1}, {1, 2}}, &renv_ten);
      renvs[renv_len + 1] = std::move(renv_ten);
      */
    }
      break;
    default:assert(false);
  }

#ifdef QLMPS_TIMING_MODE
  update_env_ten_timer.PrintElapsed();
#endif

  auto update_elapsed_time = update_timer.Elapsed();
  std::cout << "Site " << std::setw(4) << target_site
            << " E0 = " << std::setw(20) << std::setprecision(kLanczEnergyOutputPrecision) << std::fixed
            << lancz_res.gs_eng
            << " TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
            << " D = " << std::setw(5) << D
            << " Iter = " << std::setw(3) << lancz_res.iters
            << " LanczT = " << std::setw(8) << lancz_elapsed_time
            << " TotT = " << std::setw(8) << update_elapsed_time
            << " S = " << std::setw(10) << std::setprecision(7) << ee;
  std::cout << std::scientific << std::endl;
  return lancz_res.gs_eng;
}

template<typename TenElemT, typename QNT>
inline void LoadRelatedTensOnTwoSiteAlgWhenRightMoving(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const size_t left_boundary,
    const FiniteVMPSSweepParams &sweep_params
) {
#ifdef QLMPS_TIMING_MODE
  Timer preprocessing_timer("two_site_fvmps_preprocessing");
#endif
  auto N = mps.size();
  if (target_site != left_boundary) {
    mps.LoadTen(
        target_site + 1,
        GenMPSTenName(sweep_params.mps_path, target_site + 1)
    );
    auto renv_len = N - (target_site + 2);
    auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
    renvs.LoadTen(renv_len, renv_file);
    RemoveFile(renv_file);
  } else {
    mps.LoadTen(
        target_site,
        GenMPSTenName(sweep_params.mps_path, target_site)
    );
    auto renv_len = (N - 1) - (target_site + 1);
    auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
    renvs.LoadTen(renv_len, renv_file);
    RemoveFile(renv_file);
    auto lenv_len = target_site;
    auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
    lenvs.LoadTen(lenv_len, lenv_file);
  }
#ifdef QLMPS_TIMING_MODE
  preprocessing_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT>
inline void LoadRelatedTensOnTwoSiteAlgWhenLeftMoving(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const size_t right_boundary,
    const FiniteVMPSSweepParams &sweep_params
) {
#ifdef QLMPS_TIMING_MODE
  Timer preprocessing_timer("two_site_fvmps_preprocessing");
#endif
  const size_t N = mps.size();
  if (target_site != right_boundary) {
    mps.LoadTen(
        target_site - 1,
        GenMPSTenName(sweep_params.mps_path, target_site - 1)
    );
    auto lenv_len = (target_site + 1) - 2;
    auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
    lenvs.LoadTen(lenv_len, lenv_file);
    RemoveFile(lenv_file);
  } else {
    mps.LoadTen(
        target_site,
        GenMPSTenName(sweep_params.mps_path, target_site)
    );
    auto renv_len = (N - 1) - target_site;
    auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
    renvs.LoadTen(renv_len, renv_file);
    auto lenv_len = target_site - 1;
    auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
    RemoveFile(lenv_file);
  }
#ifdef QLMPS_TIMING_MODE
  preprocessing_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT>
inline void DumpRelatedTensOnTwoSiteAlgWhenRightMoving(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const FiniteVMPSSweepParams &sweep_params
) {
#ifdef QLMPS_TIMING_MODE
  Timer postprocessing_timer("two_site_fvmps_postprocessing");
#endif
  auto N = mps.size();
  lenvs.dealloc(target_site);
  renvs.dealloc(N - (target_site + 2));
  mps.DumpTen(
      target_site,
      GenMPSTenName(sweep_params.mps_path, target_site),
      true
  );
  lenvs.DumpTen(
      target_site + 1,
      GenEnvTenName("l", target_site + 1, sweep_params.temp_path)
  );
#ifdef QLMPS_TIMING_MODE
  postprocessing_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT>
inline void DumpRelatedTensOnTwoSiteAlgWhenLeftMoving(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const FiniteVMPSSweepParams &sweep_params
) {
#ifdef QLMPS_TIMING_MODE
  Timer postprocessing_timer("two_site_fvmps_postprocessing");
#endif
  auto N = mps.size();
  lenvs.dealloc((target_site + 1) - 2);
  renvs.dealloc(N - (target_site + 1));
  mps.DumpTen(
      target_site,
      GenMPSTenName(sweep_params.mps_path, target_site),
      true
  );
  auto next_renv_len = N - target_site;
  renvs.DumpTen(
      next_renv_len,
      GenEnvTenName("r", next_renv_len, sweep_params.temp_path)
  );

#ifdef QLMPS_TIMING_MODE
  postprocessing_timer.PrintElapsed();
#endif
}

}//qlmps

#endif