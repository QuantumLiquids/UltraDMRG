// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-7-30
*
* Description: QuantumLiquids/UltraDMRG project. Implementation details for noised two-site algorithm.
*/

/**
@file two_site_update_noise_finite_vmps_impl.h
@brief Implementation details for noised two-site algorithm.
*/

#ifndef QLMPS_ALGORITHM_VMPS_TWO_SITE_UPDATE_NOISE_FINITE_VMPS_IMPL_H
#define QLMPS_ALGORITHM_VMPS_TWO_SITE_UPDATE_NOISE_FINITE_VMPS_IMPL_H


#include <stdio.h>                                                // remove
#include <iomanip>

#include "qlten/qlten.h"
#include "qlten/utility/timer.h"                                  // Timer

#include "qlmps/algorithm/vmps/vmps_init.h"                      // FiniteVMPSInit
#include "qlmps/algorithm/vmps/lanczos_vmps_solver_impl.h"       // LanczosSolver, LanczosParams
#include "qlmps/algorithm/finite_vmps_sweep_params.h"            // FiniteVMPSSweepParams
#include "qlmps/one_dim_tn/mpo/mpo.h"                            // MPO
#include "qlmps/one_dim_tn/mps/finite_mps/finite_mps.h"          // FiniteMPS
#include "qlmps/utilities.h"                                     // IsPathExist, CreatPath
#include "qlmps/one_dim_tn/framework/ten_vec.h"                  // TenVec
#include "qlmps/consts.h"

#ifdef Release
#define NDEBUG
#endif

#include <assert.h>

namespace qlmps {
using namespace qlten;


// Forward declarition
template<typename DTenT>
inline double MeasureEE(const DTenT &s, const size_t sdim);

template<typename TenElemT, typename QNT>
std::pair<size_t, size_t> CheckAndUpdateBoundaryMPSTensors(FiniteMPS<TenElemT, QNT> &,
                                                           const std::string &,
                                                           const size_t);

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

template<typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSSweep(//also a overload
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const size_t left_boundary,
    const size_t right_boundary,
    double &noise_start
);

template<typename QNT>
bool IsQNCovered(const QNSectorVec<QNT> &, const QNSectorVec<QNT> &);

/**
 Function to perform two-site noised update finite vMPS algorithm.

 @note The input MPS will be considered an empty one.
 @note The canonical center of MPS should be set at around left side
*/
template<typename TenElemT, typename QNT>
QLTEN_Double TwoSiteFiniteVMPSWithNoise( //same function name, overload by class of FiniteVMPSSweepParams
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params
) {
  if (sweep_params.noises.size() == 0) {
    FiniteVMPSSweepParams sweep_params_copy = sweep_params;
    sweep_params_copy.noise_valid = false;
    TwoSiteFiniteVMPS(mps, mpo, sweep_params_copy);
  }
  assert(mps.size() == mpo.size());

  std::cout << "\n";
  std::cout << "***** Two-Site Noised Update VMPS Program *****" << "\n";
  auto [left_boundary, right_boundary] = FiniteVMPSInit(mps, mpo, sweep_params);

  std::cout << "Preseted noises: \t[";
  for (size_t i = 0; i < sweep_params.noises.size(); i++) {
    std::cout << sweep_params.noises[i];
    if (i != sweep_params.noises.size() - 1) {
      std::cout << ", ";
    } else {
      std::cout << "]" << std::endl;
    }
  }
  QLTEN_Double e0;
  double noise_start;
  mps.LoadTen(left_boundary, GenMPSTenName(sweep_params.mps_path, left_boundary));
  mps.LoadTen(left_boundary + 1, GenMPSTenName(sweep_params.mps_path, left_boundary + 1));
  for (size_t sweep = 1; sweep <= sweep_params.sweeps; ++sweep) {
    if ((sweep - 1) < sweep_params.noises.size()) {
      noise_start = sweep_params.noises[sweep - 1];
    }
    std::cout << "sweep " << sweep << std::endl;
    Timer sweep_timer("sweep");
    e0 = TwoSiteFiniteVMPSSweep(mps, mpo, sweep_params,
                                left_boundary, right_boundary, noise_start);
    sweep_timer.PrintElapsed();
    std::cout << std::endl;
  }
  mps.LeftCanonicalizeTen(left_boundary); // make sure the central is at left_boundary + 1
  mps.DumpTen(left_boundary, GenMPSTenName(sweep_params.mps_path, left_boundary), true);
  mps.DumpTen(left_boundary + 1, GenMPSTenName(sweep_params.mps_path, left_boundary + 1), true);
  return e0;
}


/**
Two-site (noised) update DMRG algorithm refer to 10.1103/PhysRevB.91.155115
*/
template<typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSSweep(//also a overload
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const size_t left_boundary,
    const size_t right_boundary,
    double &noise_start
) {
  auto N = mps.size();
  using TenT = QLTensor<TenElemT, QNT>;
  TenVec<TenT> lenvs(N), renvs(N);
  double e0(0.0), actual_e0(0.0), actual_laststep_e0(0.0);

  const double alpha = sweep_params.alpha;
  const double noise_decrease = sweep_params.noise_decrease;
  const double noise_increase = sweep_params.noise_increase;
  const double max_noise = sweep_params.max_noise;

  double &noise_running = noise_start;
  for (size_t i = left_boundary; i < right_boundary - 1; ++i) {
    //The last two site [right_boudary-1, right_boundary] will update when sweep back
    LoadRelatedTensOnTwoSiteAlgWhenNoisedRightMoving(mps, lenvs, renvs, i, left_boundary, sweep_params);
    // mps[i+1](do not need load), mps[i+2](need load)
    // lenvs[i](do not need load), and mps[i+1]'s renvs
    // mps[i+1]'s renvs file can be removed
    actual_e0 = CalEnergyEptTwoSite(mps, mpo, lenvs, renvs, i, i + 1);
    if ((actual_e0 - e0) <= 0.0) {
      // expand and truncate let the energy lower or not change
      // this case is very rare, but include the boundary mps tensor case
      // so we do nothing now
    } else if ((actual_e0 - e0) >= alpha * fabs(actual_laststep_e0 - e0)) {
      // below two case suppose actual_laststep_e0-laststep_e0>0, usually it is right
      noise_running = noise_running * noise_decrease;
    } else {
      noise_running = std::min(noise_running * noise_increase, max_noise);
    }
    e0 = TwoSiteFiniteVMPSUpdate(
        mps,
        lenvs, renvs,
        mpo,
        sweep_params, 'r', i,
        noise_running
    );
    actual_laststep_e0 = actual_e0;
    DumpRelatedTensTwoSiteAlgNoiseCase(mps, lenvs, renvs, i, 'r',
                                       sweep_params);    // note: here we need dump mps[i](free memory),
    // lenvs[i+1](without free memory)
  }

  for (size_t i = right_boundary; i > left_boundary + 1; --i) {
    LoadRelatedTensOnTwoSiteAlgWhenNoisedLeftMoving(mps, lenvs, renvs, i, right_boundary, sweep_params);
    actual_e0 = CalEnergyEptTwoSite(mps, mpo, lenvs, renvs, i - 1, i);
    if ((actual_e0 - e0) <= 0.0) {
    } else if ((actual_e0 - e0) >= alpha * fabs(actual_laststep_e0 - e0)) {
      noise_running = noise_running * noise_decrease;
    } else {
      noise_running = std::min(noise_running * noise_increase, max_noise);
    }
    e0 = TwoSiteFiniteVMPSUpdate(
        mps,
        lenvs, renvs,
        mpo,
        sweep_params, 'l', i,
        noise_running
    );
    actual_laststep_e0 = actual_e0;
    DumpRelatedTensTwoSiteAlgNoiseCase(mps, lenvs, renvs, i, 'l', sweep_params);
  }
  return e0;
}


/**  Single step for two site noised update.
This function includes below procedure:
- update `mps[target]` and `mps[next_site]` tensors according corresponding environment tensors and the mpo tensor, using lanczos algorithm;
- expand `mps[target]*mps[next_site]` and `mps[next_next_site]` by noise, if need;
- canonicalize mps to `mps[next_site]` by SVD, while truncate tensor `mps[target]` if need;
- generate the next environment in the direction.

When using this function, one must make sure memory at least contains `mps[target]` tensor,
`mps[next_site]`, its environment tensors, and `mps[next_next_site]`
*/
template<typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSUpdate(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const char dir,
    const size_t target_site,
    const double preset_noise
) {
  Timer update_timer("two_site_fvmps_update");

#ifdef QLMPS_TIMING_MODE
  Timer preprocessing_timer("two_site_fvmps_initial_state");
#endif
  double noise = preset_noise;
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
    case 'r':
      lsite_idx = target_site;
      rsite_idx = target_site + 1;
      lenv_len = target_site;
      renv_len = N - (target_site + 2);
      break;
    case 'l':
      lsite_idx = target_site - 1;
      rsite_idx = target_site;
      lenv_len = target_site - 1;
      renv_len = N - target_site - 1;
      break;
    default:
      std::cout << "dir must be 'r' or 'l', but " << dir << std::endl;
      exit(1);
  }

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
  preprocessing_timer.PrintElapsed();
#endif
  // Lanczos
  Timer lancz_timer("two_site_fvmps_lancz");
  auto lancz_res = LanczosSolver(
      eff_ham, init_state,
      &eff_ham_mul_two_site_state,
      sweep_params.lancz_params
  );//Note here init_state is deleted
#ifdef QLMPS_TIMING_MODE
  auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif


#ifdef QLMPS_TIMING_MODE
  Timer expand_timer("two_site_fvmps_add_noise");
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
    TwoSiteFiniteVMPSExpand(
        mps,
        lancz_res.gs_vec,
        eff_ham,
        dir,
        target_site,
        noise
    );
  }

#ifdef QLMPS_TIMING_MODE
  expand_timer.PrintElapsed();
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
  SVD(
      lancz_res.gs_vec,
      svd_ldims, Div(mps[lsite_idx]),
      sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
      &u, &s, &vt, &actual_trunc_err, &D
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
    case 'r':
      mps[lsite_idx] = std::move(u);
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
    default:
      assert(false);
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
      lenvs[lenv_len + 1] = std::move(UpdateSiteLenvs(lenvs[lenv_len], mps[target_site], mpo[target_site]));
    }
      break;
    case 'l': {
      renvs[renv_len + 1] = std::move(UpdateSiteRenvs(renvs[renv_len], mps[target_site], mpo[target_site]));
    }
      break;
    default:
      assert(false);
  }

#ifdef QLMPS_TIMING_MODE
  update_env_ten_timer.PrintElapsed();
#endif


  auto update_elapsed_time = update_timer.Elapsed();
  std::cout << "Site " << std::setw(4) << target_site
            << " E0 = " << std::setw(20) << std::setprecision(kLanczEnergyOutputPrecision) << std::fixed
            << lancz_res.gs_eng
            << " noise = " << std::setprecision(2) << std::scientific << noise << std::fixed
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
void TwoSiteFiniteVMPSExpand(
    FiniteMPS<TenElemT, QNT> &mps,
    QLTensor<TenElemT, QNT> *gs_vec,
    const std::vector<QLTensor<TenElemT, QNT> *> &eff_ham,
    const char dir,
    const size_t target_site,
    const double noise
) {
  // note: The expanded tensors are saved in *gs_vec, and mps[next_next_site]
  using TenT = QLTensor<TenElemT, QNT>;
  TenT *ten_tmp = new TenT();
  // we suppose mps contain mps[targe_site], mps[next_site],  mps[next_next_site]
  if (dir == 'r') {
#ifdef QLMPS_TIMING_MODE
    Timer contract_timer("\t Contract time for expansion");
#endif
    TenT temp_ten1, temp_ten2;
    Contract(eff_ham[0], gs_vec, {{2},
                                  {0}}, &temp_ten1);
    Contract<TenElemT, QNT, true, true>(temp_ten1, *eff_ham[1], 1, 0, 2, temp_ten2);
    Contract<TenElemT, QNT, true, true>(temp_ten2, *eff_ham[2], 4, 0, 2, *ten_tmp);

#ifdef QLMPS_TIMING_MODE
    contract_timer.PrintElapsed();
    Timer fuse_timer("\t Fuse index time for expansion");
#endif
    ten_tmp->FuseIndex(0, 4);
#ifdef QLMPS_TIMING_MODE
    fuse_timer.PrintElapsed();
    Timer scalar_timer("\t Scalar multiplication time fo expansion");
#endif
    (*ten_tmp) *= noise;
#ifdef QLMPS_TIMING_MODE
    scalar_timer.PrintElapsed();
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
  } else if (dir == 'l') {
    size_t next_next_site = target_site - 2;
#ifdef QLMPS_TIMING_MODE
    Timer contract_timer("\t Contract time for expansion");
#endif
    Contract(gs_vec, eff_ham[3], {{3},
                                  {2}}, ten_tmp);

    InplaceContract(ten_tmp, eff_ham[2], {{2, 4},
                                          {1, 3}});
    InplaceContract(ten_tmp, eff_ham[1], {{1, 3},
                                          {1, 3}});
#ifdef QLMPS_TIMING_MODE
    contract_timer.PrintElapsed();
    Timer fuse_timer("\t Fuse index time for expansion");
#endif
    ten_tmp->Transpose({0, 3, 4, 2, 1});
    ten_tmp->FuseIndex(0, 1);
#ifdef QLMPS_TIMING_MODE
    fuse_timer.PrintElapsed();
    Timer scalar_timer("\t Scalar multiplication time fo expansion");
#endif
    (*ten_tmp) *= noise;
#ifdef QLMPS_TIMING_MODE
    scalar_timer.PrintElapsed();
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
}


template<typename TenElemT, typename QNT>
inline void LoadRelatedTensOnTwoSiteAlgWhenNoisedRightMoving(
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
        target_site + 2,
        GenMPSTenName(sweep_params.mps_path, target_site + 2)
    );
    auto renv_len = N - (target_site + 2);
    auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
    renvs.LoadTen(renv_len, renv_file);
    RemoveFile(renv_file);
  } else {
    mps.LoadTen(
        target_site + 2,
        GenMPSTenName(sweep_params.mps_path, target_site + 2)
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
inline void LoadRelatedTensOnTwoSiteAlgWhenNoisedLeftMoving(
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
        target_site - 2,
        GenMPSTenName(sweep_params.mps_path, target_site - 2)
    );
    auto lenv_len = (target_site + 1) - 2;
    auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
    lenvs.LoadTen(lenv_len, lenv_file);
    RemoveFile(lenv_file);
  } else {
    mps.LoadTen(
        target_site - 2,
        GenMPSTenName(sweep_params.mps_path, target_site - 2)
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
void DumpRelatedTensTwoSiteAlgNoiseCase(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const char dir,
    const FiniteVMPSSweepParams &sweep_params
) {
#ifdef QLMPS_TIMING_MODE
  Timer postprocessing_timer("two_site_fvmps_dump_tensors");
#endif
  auto N = mps.size();
  switch (dir) {
    case 'r': {
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
    }
      break;
    case 'l': {
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
    }
      break;
    default:
      assert(false);
  }
#ifdef QLMPS_TIMING_MODE
  postprocessing_timer.PrintElapsed();
#endif
}


template<typename TenElemT, typename QNT>
double CalEnergyEptTwoSite(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t lsite,
    const size_t rsite
) {
#ifdef QLMPS_TIMING_MODE
  Timer preprocessing_timer("two_site_fvmps_pre_calculate_energy");
#endif
  using TenT = QLTensor<TenElemT, QNT>;
  std::vector<TenT *> eff_ham(4);
  size_t lenv_len = lsite;
  size_t renv_len = mps.size() - rsite - 1;
  eff_ham[0] = lenvs(lenv_len);
  // Safe const casts for MPO local tensors.
  eff_ham[1] = const_cast<TenT *>(&mpo[lsite]);
  eff_ham[2] = const_cast<TenT *>(&mpo[rsite]);
  eff_ham[3] = renvs(renv_len);
  TenT wave_function;
  Contract(mps(lsite), mps(rsite), {{2},
                                    {0}}, &wave_function);
  TenT *h_mul_state = eff_ham_mul_two_site_state(eff_ham, &wave_function);
  TenT scalar_ten;
  TenT wave_function_dag = Dag(wave_function);
  Contract(h_mul_state, &wave_function_dag, {{0, 1, 2, 3},
                                             {0, 1, 2, 3}}, &scalar_ten);
  delete h_mul_state;
  double energy = Real(scalar_ten());
#ifdef QLMPS_TIMING_MODE
  preprocessing_timer.PrintElapsed();
#endif
  return energy;
}

/**
 * if qnsectors1's qn set contains qnsectors2's qn set
 */
template<typename QNT>
bool IsQNCovered(const QNSectorVec<QNT> &qnsectors1,
                 const QNSectorVec<QNT> &qnsectors2) {
  size_t size1(qnsectors1.size()), size2(qnsectors2.size());
  if (size1 < size2) {
    return false;
  }
  std::vector<size_t> hash_set_of_qns_in_qnsectors1, hash_set_of_qns_in_qnsectors2;
  hash_set_of_qns_in_qnsectors1.reserve(size1);
  hash_set_of_qns_in_qnsectors2.reserve(size2);
  for (const QNSector<QNT> &qnsector: qnsectors1) {
    hash_set_of_qns_in_qnsectors1.push_back(qnsector.GetQn().Hash());
  }
  std::sort(hash_set_of_qns_in_qnsectors1.begin(),
            hash_set_of_qns_in_qnsectors1.end());
  for (const QNSector<QNT> &qnsector: qnsectors2) {
    hash_set_of_qns_in_qnsectors2.push_back(qnsector.GetQn().Hash());
  }
  std::sort(hash_set_of_qns_in_qnsectors2.begin(),
            hash_set_of_qns_in_qnsectors2.end());

  if (hash_set_of_qns_in_qnsectors1.back() < hash_set_of_qns_in_qnsectors2.back()) {
    return false;
  }

  auto iter1 = hash_set_of_qns_in_qnsectors1.begin();
  auto iter2 = hash_set_of_qns_in_qnsectors2.begin();
  while (iter2 < hash_set_of_qns_in_qnsectors2.end()) {
    if ((*iter1) < (*iter2)) {
      iter1++;
    } else if ((*iter1) == (*iter2)) {
      iter1++;
      iter2++;
    } else {
      return false;
    }
  }
  return true;
}
}//qlmps
#endif
