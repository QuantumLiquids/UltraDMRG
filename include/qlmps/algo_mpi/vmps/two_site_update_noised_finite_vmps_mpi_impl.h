// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-31
*
* Description: QuantumLiquids/UltraDMRG project. Two-site update noised finite size vMPS with MPI Paralization
*/

/**
@file two_site_update_noised_finite_vmps_mpi_impl.h
@brief Two-site update noised finite size vMPS with MPI Paralization
*/
#ifndef QLMPS_ALGO_MPI_VMPS_TWO_SITE_UPDATE_NOISED_FINITE_VMPS_MPI_IMPL_H
#define QLMPS_ALGO_MPI_VMPS_TWO_SITE_UPDATE_NOISED_FINITE_VMPS_MPI_IMPL_H

#include "qlten/qlten.h"
#include "qlmps/algorithm/lanczos_params.h"                                  //LanczosParams
#include "qlmps/algo_mpi/mps_algo_order.h"                                   //VMPSORDER
#include "qlmps/algo_mpi/vmps/vmps_mpi_init_master.h"                        //MPI vmps initial
#include "qlmps/algo_mpi/vmps/vmps_mpi_init_slave.h"                         //MPI vmps initial
#include "qlmps/algo_mpi/vmps/two_site_update_finite_vmps_mpi.h"             //TwoSiteMPIVMPSSweepParams
#include "qlmps/algo_mpi/vmps/two_site_update_noised_finite_vmps_mpi.h"      //FiniteVMPSSweepParams
#include "lanczos_solver_mpi_master.h"                        //MPI Lanczos solver
#include "qlmps/algo_mpi/vmps/two_site_update_finite_vmps_mpi_impl_master.h" //SlaveTwoSiteFiniteVMPS
#include <thread>                                                             //thread

namespace qlmps {
using namespace qlten;

//forward declaration
template<typename TenElemT, typename QNT>
inline void LoadRelatedTensOnTwoSiteAlgWhenNoisedRightMoving(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const size_t left_boundary,
    const FiniteVMPSSweepParams &sweep_params
);

template<typename TenElemT, typename QNT>
inline void LoadRelatedTensOnTwoSiteAlgWhenNoisedLeftMoving(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const size_t right_boundary,
    const FiniteVMPSSweepParams &sweep_params
);

/**
 * @note We suggest that the tensor manipulation thread number in master process = that in workers' - 2
 *       so that the remaining 2 threads can be used to read/dump tensors.
 *       (usually the thread numbers in different MPI process are uniform.)
 */
template<typename TenElemT, typename QNT>
inline QLTEN_Double TwoSiteFiniteVMPSWithNoise_(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const MPI_Comm &comm
) {
  QLTEN_Double e0(0.0);
  if (sweep_params.noises.size() == 0) {
    if (sweep_params.noise_valid == true) {
      std::cerr << "noise parameters setting conflict" << std::endl;
      exit(-1);
    }
    TwoSiteFiniteVMPS(mps, mpo, sweep_params, comm);
  }
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == kMPIMasterRank) {
    e0 = MasterTwoSiteFiniteVMPSWithNoise(mps, mpo, sweep_params, comm);
  } else {
    SlaveTwoSiteFiniteVMPS<TenElemT, QNT>(mpo, comm);
  }
  return e0;
}

template<typename TenElemT, typename QNT>
QLTEN_Double MasterTwoSiteFiniteVMPSWithNoise(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const MPI_Comm &comm
) {
  assert(mps.size() == mpo.size());
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  std::cout << "***** Two-Site Noised Update VMPS Program (with MPI Parallel) *****" << "\n";
  MasterBroadcastOrder(program_start, rank, comm);
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
  std::cout << "Preseted noises: \t[";
  for (size_t i = 0; i < sweep_params.noises.size(); i++) {
    std::cout << sweep_params.noises[i];
    if (i != sweep_params.noises.size() - 1) {
      std::cout << ", ";
    } else {
      std::cout << "]" << std::endl;
    }
  }
  double e0(0.0);
  double noise;
  mps.LoadTen(left_boundary, GenMPSTenName(sweep_params.mps_path, left_boundary));
  mps.LoadTen(left_boundary + 1, GenMPSTenName(sweep_params.mps_path, left_boundary + 1));
  for (size_t sweep = 1; sweep <= sweep_params.sweeps; ++sweep) {
    if ((sweep - 1) < sweep_params.noises.size()) {
      noise = sweep_params.noises[sweep - 1];
    }
    std::cout << "sweep " << sweep << std::endl;
    Timer sweep_timer("sweep");
    e0 = TwoSiteFiniteVMPSSweep(mps, mpo, sweep_params,
                                left_boundary, right_boundary,
                                noise, comm);

    sweep_timer.PrintElapsed();
    std::cout << "\n";
  }
  mps.LeftCanonicalizeTen(left_boundary);//make sure the central is as left_boundary
  mps.DumpTen(left_boundary, GenMPSTenName(sweep_params.mps_path, left_boundary), true);
  mps.DumpTen(left_boundary + 1, GenMPSTenName(sweep_params.mps_path, left_boundary + 1), true);
  assert(mps.empty());
  NoisedVMPSPostProcess(mps, sweep_params, left_boundary);
  MasterBroadcastOrder(program_final, rank, comm);
  return e0;
}

///< Move center from left_boundary_ + 1 to 0
template<typename TenElemT, typename QNT>
void NoisedVMPSPostProcess(
    FiniteMPS<TenElemT, QNT> &mps,
    const FiniteVMPSSweepParams &sweep_params,
    size_t left_boundary) {
  size_t center = left_boundary + 1;
  mps.LoadTen(sweep_params.mps_path, center);
  for (size_t site = center; site > 0; site--) {
    mps.LoadTen(sweep_params.mps_path, site - 1);
    mps.RightCanonicalizeTen(site);
    mps.DumpTen(sweep_params.mps_path, site);
  }
  mps.DumpTen(sweep_params.mps_path, 0);
  std::cout << "Moved the center of MPS to 0." << std::endl;
}

/**
 *
 * @return
 * At the end of the function, the center of MPS is moved to two sites (left_boundary, left_boundary+1).
 * The center has two sites because expansion appear between this two sites.
 * In MPS, only tensors in these two sites are stored in memory.
 */
template<typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSSweep(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const size_t left_boundary,
    const size_t right_boundary,
    const double noise,
    const MPI_Comm &comm
) {
  auto N = mps.size();
  using TenT = QLTensor<TenElemT, QNT>;
  TenVec<TenT> lenvs(N - 1);
  TenVec<TenT> renvs(N - 1);
  double e0;
  const size_t update_site_size = right_boundary - left_boundary - 1;
  std::thread load_related_tens_thread;
  std::thread dump_related_tens_thread;
  LoadRelatedTensOnTwoSiteAlgWhenNoisedRightMoving(mps, lenvs, renvs, left_boundary, left_boundary, sweep_params);
  for (size_t i = left_boundary; i <= right_boundary - 2; ++i) {
    // Load to-be-used tensors
    if (i < right_boundary - 2) {
      load_related_tens_thread = std::thread(
          LoadRelatedTensOnTwoSiteAlgWhenNoisedRightMoving<TenElemT, QNT>,
          std::ref(mps),
          std::ref(lenvs),
          std::ref(renvs),
          i + 1, //note here is different,
          left_boundary,
          std::ref(sweep_params)
      );
    }
    // LoadRelatedTensOnTwoSiteAlgWhenNoisedRightMoving(mps, lenvs, renvs, i, left_boundary, sweep_params);
    e0 = MasterTwoSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'r', i, noise, comm);
    // Dump related tensor to HD and remove unused tensor from RAM
    if (i > left_boundary) {
      dump_related_tens_thread.join();
    }
    dump_related_tens_thread = std::thread(
        DumpRelatedTensOnTwoSiteAlgWhenRightMoving<TenElemT, QNT>,
        std::ref(mps),
        std::ref(lenvs),
        std::ref(renvs),
        i,
        std::ref(sweep_params)
    );
    if (i < right_boundary - 2) {
      load_related_tens_thread.join();
    }
  }
  dump_related_tens_thread.join();

  LoadRelatedTensOnTwoSiteAlgWhenNoisedLeftMoving(mps, lenvs, renvs, right_boundary, right_boundary, sweep_params);
  for (size_t i = right_boundary; i >= left_boundary + 2; --i) {
    if (i > left_boundary + 2) {
      load_related_tens_thread = std::thread(
          LoadRelatedTensOnTwoSiteAlgWhenNoisedLeftMoving<TenElemT, QNT>,
          std::ref(mps),
          std::ref(lenvs),
          std::ref(renvs),
          i - 1, //note here is different,
          right_boundary,
          std::ref(sweep_params)
      );
    }
    // LoadRelatedTensOnTwoSiteAlgWhenNoisedLeftMoving(mps, lenvs, renvs, i, right_boundary, sweep_params);
    e0 = MasterTwoSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'l', i, noise, comm);
    if (i < right_boundary) {
      dump_related_tens_thread.join();
    }
    dump_related_tens_thread = std::thread(
        DumpRelatedTensOnTwoSiteAlgWhenLeftMoving<TenElemT, QNT>,
        std::ref(mps),
        std::ref(lenvs),
        std::ref(renvs),
        i,
        std::ref(sweep_params)
    );
    if (i > left_boundary + 2) {
      load_related_tens_thread.join();
    }
  }
  dump_related_tens_thread.join();
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
    double noise,
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

  bool need_expand(true);
  if (fabs(noise) < 1e-10) {
    noise = 0.0;    //just for output
    need_expand = false;
  } else {
    const size_t physical_dim_l = mps[lsite_idx].GetShape()[1];
    const size_t physical_dim_r = mps[rsite_idx].GetShape()[1];
    const QNSectorVec<QNT> *qnscts_right;
    const QNSectorVec<QNT> *qnscts_left;
    Index<QNT> fused_index1, fused_index2;
    if (physical_dim_l == 2) {
      qnscts_left = &(mps[lsite_idx].GetIndexes()[0].GetQNScts());
    } else {
      std::vector<qlten::QNSctsOffsetInfo> qnscts_offset_info_list;
      fused_index1 = FuseTwoIndexAndRecordInfo(
          mps[lsite_idx].GetIndexes()[0],
          InverseIndex(mps[lsite_idx].GetIndexes()[1]),
          qnscts_offset_info_list
      );
      qnscts_left = &(fused_index1.GetQNScts());
    }

    if (physical_dim_r == 2) {
      qnscts_right = &(mps[rsite_idx].GetIndexes()[2].GetQNScts());
    } else {
      std::vector<qlten::QNSctsOffsetInfo> qnscts_offset_info_list;
      fused_index2 = FuseTwoIndexAndRecordInfo(
          mps[rsite_idx].GetIndexes()[1],
          mps[rsite_idx].GetIndexes()[2],
          qnscts_offset_info_list
      );
      qnscts_right = &(fused_index2.GetQNScts());
    }

    if (dir == 'r' &&
        IsQNCovered(*qnscts_right, *qnscts_left)
        ) {
      noise = 0.0;
      need_expand = false;
    } else if (dir == 'l' &&
        IsQNCovered(*qnscts_left, *qnscts_right)
        ) {
      noise = 0.0;
      need_expand = false;
    }
  }

  if (need_expand) {
    if (dir == 'r') {
      MasterBroadcastOrder(contract_for_right_moving_expansion, kMPIMasterRank, comm);
      MasterTwoSiteFiniteVMPSRightMovingExpand(
          mps,
          lancz_res.gs_vec,
          eff_ham,
          target_site,
          noise,
          comm
      );
    } else {
      MasterBroadcastOrder(contract_for_left_moving_expansion, kMPIMasterRank, comm);
      MasterTwoSiteFiniteVMPSLeftMovingExpand(
          mps,
          lancz_res.gs_vec,
          eff_ham,
          target_site,
          noise,
          comm
      );
    }
  }

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
    }
      break;
    case 'l': {
      MasterBroadcastOrder(growing_right_env, kMPIMasterRank, comm);
      renvs(renv_len + 1) = MasterGrowRightEnvironment(mpo[target_site], mps[target_site], comm);
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
void MasterTwoSiteFiniteVMPSRightMovingExpand(
    FiniteMPS<TenElemT, QNT> &mps,
    QLTensor<TenElemT, QNT> *gs_vec,
    const std::vector<QLTensor<TenElemT, QNT> *> &eff_ham,
    const size_t target_site,
    const double noise,
    const MPI_Comm &comm
) {
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);
  // note: The expanded tensors are saved in *gs_vec, and mps[next_next_site]
  using TenT = QLTensor<TenElemT, QNT>;
  // we suppose mps contain mps[targe_site], mps[next_site],  mps[next_next_site]
#ifdef QLMPS_TIMING_MODE
  Timer contract_timer("\t Contract, fuse index and scale for expansion");
#endif

#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_state_timer("expansion_broadcast_state_send");
#endif
  MPI_Bcast(*gs_vec, kMPIMasterRank, comm);
  HANDLE_MPI_ERROR(::MPI_Bcast(const_cast<double *>(&noise), 1, MPI_DOUBLE, kMPIMasterRank, comm));
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
#endif
  const size_t split_idx = 0;
  const Index<QNT> &splited_index = eff_ham[0]->GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();//total task number
  const QNSectorVec<QNT> &split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_size);
  const size_t slave_size = mpi_size - 1; //total number of slaves

  IndexVec<QNT> ten_tmp_indexes(4);
  ten_tmp_indexes[1] = splited_index;
  ten_tmp_indexes[2] = eff_ham[1]->GetIndexes()[2];
  ten_tmp_indexes[3] = eff_ham[2]->GetIndexes()[2];

  Index<QNT> index_a = gs_vec->GetIndexes()[3];
  std::vector<qlten::QNSctsOffsetInfo> qnscts_offset_info_list;
  Index<QNT> index_b = FuseTwoIndexAndRecordInfo(
      index_a,
      eff_ham[2]->GetIndexes()[3],
      qnscts_offset_info_list
  );
  std::map<size_t, int> b_idx_qnsct_coor_expanded_idx_qnsct_coor_map;
  ten_tmp_indexes[0] = ExpandIndexMCAndRecordInfo(
      index_a,
      index_b,
      b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
  );

  TenT ten_tmp_shell = TenT(ten_tmp_indexes);
  for (size_t j = 0; j < task_size; j++) {
    res_list.push_back(ten_tmp_shell);
  }
  if (slave_size < task_size) {
    std::vector<size_t> task_difficuty(task_size);
    for (size_t i = 0; i < task_size; i++) {
      task_difficuty[i] = split_qnscts[i].GetDegeneracy();
    }
    std::vector<size_t> arraging_tasks(task_size - slave_size);
    std::iota(arraging_tasks.begin(), arraging_tasks.end(), slave_size);
    std::sort(arraging_tasks.begin(),
              arraging_tasks.end(),
              [&task_difficuty](size_t task1, size_t task2) {
                return task_difficuty[task1] > task_difficuty[task2];
              });
    for (size_t i = 0; i < task_size - slave_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      MPI_Status status = bsdt.MPI_Recv(comm, MPI_ANY_SOURCE, MPI_ANY_TAG);
      int worker = status.MPI_SOURCE;
      hp_numeric::MPI_Send(arraging_tasks[i], worker, 2 * worker, comm);
    }
  }
  for (size_t i = std::max(task_size, slave_size) - slave_size; i < task_size; i++) {
    auto &bsdt = res_list[i].GetBlkSparDataTen();
    MPI_Status status = bsdt.MPI_Recv(comm, MPI_ANY_SOURCE, MPI_ANY_TAG);
    int worker = status.MPI_SOURCE;
    hp_numeric::MPI_Send(2 * task_size, worker, 2 * worker, comm);//finish signal
  }
#ifdef QLMPS_MPI_TIMING_MODE
  Timer sum_state_timer(" parallel_summation_reduce");
#endif
  TenT *ten_tmp = new TenT();
  CollectiveLinearCombine(res_list, *ten_tmp);
#ifdef QLMPS_MPI_TIMING_MODE
  sum_state_timer.PrintElapsed();
#endif

#ifdef QLMPS_TIMING_MODE
  contract_timer.PrintElapsed();
  Timer expansion_timer("\t Magic expansion time");
#endif
  gs_vec->Transpose({3, 0, 1, 2});
  TenT expanded_ten;
  ExpandQNBlocks(gs_vec, ten_tmp, {0}, &expanded_ten);
  expanded_ten.Transpose({1, 2, 3, 0});
  (*gs_vec) = std::move(expanded_ten);
#ifdef QLMPS_TIMING_MODE
  expansion_timer.PrintElapsed();

  expansion_timer.ClearAndRestart();
#endif
  size_t next_next_site = target_site + 2;
  auto expanded_index = InverseIndex(ten_tmp->GetIndexes()[0]);
  TenT expanded_zero_ten = TenT({
                                    expanded_index,
                                    mps[next_next_site].GetIndexes()[1],
                                    mps[next_next_site].GetIndexes()[2]
                                });
  (*ten_tmp) = TenT();
  ExpandQNBlocks(mps(next_next_site), &expanded_zero_ten, {0}, ten_tmp);
  delete mps(next_next_site);
  mps(next_next_site) = ten_tmp;
#ifdef QLMPS_TIMING_MODE
  expansion_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPSRightMovingExpand(
    const std::vector<QLTensor<TenElemT, QNT> *> &eff_ham,
    const MPI_Comm &comm
) {
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);
  using TenT = QLTensor<TenElemT, QNT>;
  TenT ground_state;
  double noise;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_state_timer("expansion_broadcast_state_recv");
#endif
  MPI_Bcast(ground_state, kMPIMasterRank, comm);
  ::MPI_Bcast(&noise, 1, MPI_DOUBLE, kMPIMasterRank, comm);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
  size_t task_count = 0;
#endif
  TenT state_shell(ground_state.GetIndexes());
  state_shell.Transpose({3, 0, 1, 2});
  const size_t split_idx = 0; //index of mps tensor
  const Index<QNT> &splited_index = eff_ham[0]->GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();
  const size_t worker = rank;//number from 1
  if (worker > task_size) {
#ifdef QLMPS_MPI_TIMING_MODE
    std::cout << "Slave has done task_count = " << task_count << std::endl;
#endif
    return;
  }
#ifdef QLMPS_MPI_TIMING_MODE
  Timer salve_communication_timer(" slave " + std::to_string(worker) + "'s communication");
  salve_communication_timer.Suspend();
  Timer slave_work_timer(" slave " + std::to_string(worker) + "'s work");
#endif
  size_t task = worker - 1;
  TenT eff_ham0_times_state;
  TenT temp, res;
  //First contract
  TensorContraction1SectorExecutor<TenElemT, QNT> ctrct_executor(
      eff_ham[0],
      split_idx,
      task,
      &ground_state,
      {{2},
       {0}},
      &eff_ham0_times_state
  );

  ctrct_executor.Execute();

  Contract(&eff_ham0_times_state, eff_ham[1], {{1, 2},
                                               {0, 1}}, &temp);
  eff_ham0_times_state.GetBlkSparDataTen().Clear();// save for memory
  Contract(&temp, eff_ham[2], {{4, 1},
                               {0, 1}}, &res);
  temp.GetBlkSparDataTen().Clear();
  res.FuseIndex(1, 4);
  TenT res1;
  ExpandQNBlocks(&state_shell, &res, {0}, &res1);
  res1 *= noise;
  res.GetBlkSparDataTen().Clear();

  auto &bsdt = res1.GetBlkSparDataTen();
#ifdef QLMPS_MPI_TIMING_MODE
  task_count++;
  salve_communication_timer.Restart();
#endif
  bsdt.MPI_Send(comm, kMPIMasterRank, task);//tag = task
  hp_numeric::MPI_Recv(task, kMPIMasterRank, 2 * worker, comm);//tag = 2*worker
#ifdef QLMPS_MPI_TIMING_MODE
  salve_communication_timer.Suspend();
#endif
  while (task < task_size) {
    TenT temp, res;
    ctrct_executor.SetSelectedQNSect(task);
    ctrct_executor.Execute();
    Contract(&eff_ham0_times_state, eff_ham[1], {{1, 2},
                                                 {0, 1}}, &temp);
    eff_ham0_times_state.GetBlkSparDataTen().Clear();// save for memory
    Contract(&temp, eff_ham[2], {{4, 1},
                                 {0, 1}}, &res);
    temp.GetBlkSparDataTen().Clear();
    res.FuseIndex(1, 4);
    TenT res1;
    ExpandQNBlocks(&state_shell, &res, {0}, &res1);
    res1 *= noise;
    res.GetBlkSparDataTen().Clear();

    auto &bsdt = res1.GetBlkSparDataTen();
#ifdef QLMPS_MPI_TIMING_MODE
    task_count++;
    salve_communication_timer.Restart();
#endif
    bsdt.MPI_Send(comm, kMPIMasterRank, task);//tag = task
    hp_numeric::MPI_Recv(task, kMPIMasterRank, 2 * worker, comm);
#ifdef QLMPS_MPI_TIMING_MODE
    salve_communication_timer.Suspend();
#endif
  }
#ifdef QLMPS_MPI_TIMING_MODE
  slave_work_timer.PrintElapsed();
  salve_communication_timer.PrintElapsed();
  std::cout << "Slave " << worker << " has done task_count = " << task_count << std::endl;
#endif
}

template<typename TenElemT, typename QNT>
void MasterTwoSiteFiniteVMPSLeftMovingExpand(
    FiniteMPS<TenElemT, QNT> &mps,
    QLTensor<TenElemT, QNT> *gs_vec,
    const std::vector<QLTensor<TenElemT, QNT> *> &eff_ham,
    const size_t target_site,
    const double noise,
    const MPI_Comm &comm
) {
  using TenT = QLTensor<TenElemT, QNT>;
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);
  size_t next_next_site = target_site - 2;
#ifdef QLMPS_TIMING_MODE
  Timer contract_timer("\t Contract, fuse index and scale for expansion");
#endif

#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_state_timer("expansion_broadcast_state_send");
#endif
  MPI_Bcast(*gs_vec, kMPIMasterRank, comm);
  ::MPI_Bcast(const_cast<double *>(&noise), 1, MPI_DOUBLE, kMPIMasterRank, comm);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
#endif
  const size_t split_idx = 0;
  const Index<QNT> &splited_index = eff_ham[3]->GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();//total task number
  const QNSectorVec<QNT> &split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_size);
  const size_t slave_size = mpi_size - 1; //total number of slaves

  IndexVec<QNT> ten_tmp_indexes(4);
  ten_tmp_indexes[1] = eff_ham[1]->GetIndexes()[2];
  ten_tmp_indexes[2] = eff_ham[2]->GetIndexes()[2];
  ten_tmp_indexes[3] = splited_index;

  Index<QNT> index_a = gs_vec->GetIndexes()[0];
  std::vector<qlten::QNSctsOffsetInfo> qnscts_offset_info_list;
  Index<QNT> index_b = FuseTwoIndexAndRecordInfo(
      index_a,
      eff_ham[1]->GetIndexes()[0],
      qnscts_offset_info_list
  );
  std::map<size_t, int> b_idx_qnsct_coor_expanded_idx_qnsct_coor_map;
  ten_tmp_indexes[0] = ExpandIndexMCAndRecordInfo(
      index_a,
      index_b,
      b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
  );

  TenT ten_tmp_shell = TenT(ten_tmp_indexes);
  for (size_t j = 0; j < task_size; j++) {
    res_list.push_back(ten_tmp_shell);
  }

  if (slave_size < task_size) {
    std::vector<size_t> task_difficuty(task_size);
    for (size_t i = 0; i < task_size; i++) {
      task_difficuty[i] = split_qnscts[i].GetDegeneracy();
    }
    std::vector<size_t> arraging_tasks(task_size - slave_size);
    std::iota(arraging_tasks.begin(), arraging_tasks.end(), slave_size);
    std::sort(arraging_tasks.begin(),
              arraging_tasks.end(),
              [&task_difficuty](size_t task1, size_t task2) {
                return task_difficuty[task1] > task_difficuty[task2];
              });
    for (size_t i = 0; i < task_size - slave_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      MPI_Status status = bsdt.MPI_Recv(comm, MPI_ANY_SOURCE, MPI_ANY_TAG);
      int worker = status.MPI_SOURCE;
      hp_numeric::MPI_Send(arraging_tasks[i], worker, 2 * worker, comm);
    }
  }
  size_t final_signal = FinalSignal(task_size);
  for (size_t i = std::max(task_size, slave_size) - slave_size; i < task_size; i++) {
    auto &bsdt = res_list[i].GetBlkSparDataTen();
    MPI_Status status = bsdt.MPI_Recv(comm, MPI_ANY_SOURCE, MPI_ANY_TAG);
    int worker = status.MPI_SOURCE;
    hp_numeric::MPI_Send(final_signal, worker, 2 * worker, comm);//finish signal
  }
#ifdef QLMPS_MPI_TIMING_MODE
  Timer sum_state_timer(" parallel_summation_reduce");
#endif
  TenT *ten_tmp = new TenT();
  CollectiveLinearCombine(res_list, *ten_tmp);
#ifdef QLMPS_MPI_TIMING_MODE
  sum_state_timer.PrintElapsed();
#endif

#ifdef QLMPS_TIMING_MODE
  contract_timer.PrintElapsed();
  Timer expansion_timer("\t Magic expansion time");
#endif
  TenT expanded_ten;
  ExpandQNBlocks(gs_vec, ten_tmp, {0}, &expanded_ten);
  *gs_vec = std::move(expanded_ten);
#ifdef QLMPS_TIMING_MODE
  expansion_timer.PrintElapsed();

  expansion_timer.ClearAndRestart();
#endif

  auto expanded_index = InverseIndex(ten_tmp->GetIndexes()[0]);
  TenT expanded_zero_ten = TenT({
                                    mps[next_next_site].GetIndexes()[0],
                                    mps[next_next_site].GetIndexes()[1],
                                    expanded_index
                                });
  *ten_tmp = TenT();
  ExpandQNBlocks(mps(next_next_site), &expanded_zero_ten, {2}, ten_tmp);
  delete mps(next_next_site);
  mps(next_next_site) = ten_tmp;
#ifdef QLMPS_TIMING_MODE
  expansion_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPSLeftMovingExpand(
    const std::vector<QLTensor<TenElemT, QNT> *> &eff_ham,
    const MPI_Comm &comm
) {
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);
  using TenT = QLTensor<TenElemT, QNT>;
  TenT ground_state;
  double noise;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_state_timer("expansion_broadcast_state_recv");
#endif
  MPI_Bcast(ground_state, kMPIMasterRank, comm);
  ::MPI_Bcast(&noise, 1, MPI_DOUBLE, kMPIMasterRank, comm);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
  size_t task_count = 0;
#endif
  TenT state_shell(ground_state.GetIndexes());
  const size_t split_idx = 0; //index of mps tensor
  const Index<QNT> &splited_index = eff_ham[3]->GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();
  const size_t worker = rank;//number from 1
  if (worker > task_size) {
#ifdef QLMPS_MPI_TIMING_MODE
    std::cout << "Slave has done task_count = " << task_count << std::endl;
#endif
    return;
  }
#ifdef QLMPS_MPI_TIMING_MODE
  Timer salve_communication_timer(" slave " + std::to_string(worker) + "'s communication");
  salve_communication_timer.Suspend();
  Timer slave_work_timer(" slave " + std::to_string(worker) + "'s work");
#endif
  size_t task = worker - 1;
  TenT eff_ham0_times_state;
  TenT temp, res;
  //First contract
  TensorContraction1SectorExecutor<TenElemT, QNT> ctrct_executor(
      eff_ham[3],
      split_idx,
      task,
      &ground_state,
      {{2},
       {3}},
      &eff_ham0_times_state
  );

  ctrct_executor.Execute();
  Contract(&eff_ham0_times_state, eff_ham[2], {{4, 1},
                                               {1, 3}}, &temp);
  eff_ham0_times_state.GetBlkSparDataTen().Clear();// save for memory
  Contract(&temp, eff_ham[1], {{2, 3},
                               {1, 3}}, &res);
  temp.GetBlkSparDataTen().Clear();
  res.Transpose({1, 3, 4, 2, 0});
  res.FuseIndex(0, 1);
  TenT res1;
  ExpandQNBlocks(&state_shell, &res, {0}, &res1);
  res1 *= noise;
  auto &bsdt = res1.GetBlkSparDataTen();
  res.GetBlkSparDataTen().Clear();
#ifdef QLMPS_MPI_TIMING_MODE
  task_count++;
  salve_communication_timer.Restart();
#endif
  bsdt.MPI_Send(comm, kMPIMasterRank, task);//tag = task
  hp_numeric::MPI_Recv(task, kMPIMasterRank, 2 * worker, comm);//tag = 2*worker
#ifdef QLMPS_MPI_TIMING_MODE
  salve_communication_timer.Suspend();
#endif
  while (task < task_size) {
    TenT temp, res;
    ctrct_executor.SetSelectedQNSect(task);
    ctrct_executor.Execute();
    Contract(&eff_ham0_times_state, eff_ham[2], {{4, 1},
                                                 {1, 3}}, &temp);
    eff_ham0_times_state.GetBlkSparDataTen().Clear();// save for memory
    Contract(&temp, eff_ham[1], {{2, 3},
                                 {1, 3}}, &res);
    temp.GetBlkSparDataTen().Clear();
    res.Transpose({1, 3, 4, 2, 0});
    res.FuseIndex(0, 1);
    TenT res1;
    ExpandQNBlocks(&state_shell, &res, {0}, &res1);

    res1 *= noise;
    auto &bsdt = res1.GetBlkSparDataTen();
#ifdef QLMPS_MPI_TIMING_MODE
    task_count++;
    salve_communication_timer.Restart();
#endif
    bsdt.MPI_Send(comm, kMPIMasterRank, task);//tag = task
    hp_numeric::MPI_Recv(task, kMPIMasterRank, 2 * worker, comm);
#ifdef QLMPS_MPI_TIMING_MODE
    salve_communication_timer.Suspend();
#endif
  }
#ifdef QLMPS_MPI_TIMING_MODE
  slave_work_timer.PrintElapsed();
  salve_communication_timer.PrintElapsed();
  std::cout << "Slave " << worker << " has done task_count = " << task_count << std::endl;
#endif
}

}//qlmps
#endif
