// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
*         Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-6-12
*
* Description: QuantumLiquids/UltraDMRG project. Implementation for Single-site vMPS algorithm.
*/

/**
@file single_site_update_finite_vmps_impl.h
@brief Implementation details for single-site finite variational MPS algorithm.
*/

#ifndef QLMPS_ALGORITM_VMPS_ONE_SITE_UPDATE_FINITE_VMPS_IMPL_H
#define QLMPS_ALGORITM_VMPS_ONE_SITE_UPDATE_FINITE_VMPS_IMPL_H


#include <stdio.h>                                                // remove
#include <iomanip>

#include "qlten/qlten.h"
#include "qlten/utility/timer.h"                                  // Timer

#include "qlmps/algorithm/vmps/lanczos_vmps_solver_impl.h"       // LanczosSolver, LanczosParams
#include "qlmps/algorithm/finite_vmps_sweep_params.h"            // FiniteVMPSSweepParams
#include "qlmps/one_dim_tn/mpo/mpo.h"                            // MPO
#include "qlmps/one_dim_tn/mps/finite_mps/finite_mps.h"          // FiniteMPS
#include "qlmps/utilities.h"                                     // IsPathExist, CreatPath
#include "qlmps/one_dim_tn/framework/ten_vec.h"                  // TenVec
#include "qlmps/consts.h"

#include <stdio.h>    // remove

#ifdef Release
#define NDEBUG
#endif

#include <assert.h>

namespace qlmps {
using namespace qlten;

// Helpers
template<typename DTenT>
inline double MeasureEE(const DTenT &s, const size_t sdim);

/**
  Function to perform single-site update finite vMPS algorithm.

  @note The input MPS will be considered an empty one.
  @note The canonical center of MPS should be set at site 0.
  TODO: remove update of boundary tensors.
*/
template<typename TenElemT, typename QNT>
QLTEN_Double SingleSiteFiniteVMPS(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    FiniteVMPSSweepParams &sweep_params
) {
  assert(mps.size() == mpo.size());

  std::cout << std::endl;
  std::cout << "=====> Sweep Parameter <=====" << std::endl;
  std::cout << "MPS/MPO size: \t " << mpo.size() << std::endl;
  std::cout << "The number of sweep times: \t " << sweep_params.sweeps << std::endl;
  std::cout << "Bond dimension: \t " << sweep_params.Dmin << "/" << sweep_params.Dmax << std::endl;
  std::cout << "Cut off truncation error: \t " << sweep_params.trunc_err << std::endl;
  std::cout << "Lanczos max iterations \t" << sweep_params.lancz_params.max_iterations << std::endl;
  std::cout << "Preseted noises: \t[";
  for (size_t i = 0; i < sweep_params.noises.size(); i++) {
    std::cout << sweep_params.noises[i];
    if (i != sweep_params.noises.size() - 1) {
      std::cout << ", ";
    } else {
      std::cout << "]" << std::endl;
    }
  }
  std::cout << "MPS path: \t" << sweep_params.mps_path << std::endl;
  std::cout << "Temp path: \t" << sweep_params.temp_path << std::endl;

  // If the runtime temporary directory does not exit, create it and initialize
  // the left/right environments
  if (!IsPathExist(sweep_params.temp_path)) {
    CreatPath(sweep_params.temp_path);
    InitEnvs(mps, mpo, sweep_params.mps_path, sweep_params.temp_path, 1);
    std::cout << "no exsiting path " << sweep_params.temp_path
              << ", thus progress created it and generated environment tensors."
              << std::endl;
  } else {
    std::cout << "finded exsiting path " << sweep_params.temp_path
              << ", thus progress will use the present environment tensors."
              << std::endl;
  }

  QLTEN_Double e0;

  if (sweep_params.noises.size() == 0) { sweep_params.noises.push_back(0.0); }
  double noise_start;
  mps.LoadTen(0, GenMPSTenName(sweep_params.mps_path, 0));
  for (size_t sweep = 1; sweep <= sweep_params.sweeps; ++sweep) {
    if ((sweep - 1) < sweep_params.noises.size()) {
      noise_start = sweep_params.noises[sweep - 1];
    }
    std::cout << "sweep " << sweep << std::endl;
    Timer sweep_timer("sweep");
    e0 = SingleSiteFiniteVMPSSweep(mps, mpo, sweep_params, noise_start);
    sweep_timer.PrintElapsed();
    std::cout << std::endl;
  }
  mps.DumpTen(0, GenMPSTenName(sweep_params.mps_path, 0), true);
  return e0;
}


/**
Single-site update DMRG algorithm refer to 10.1103/PhysRevB.91.155115
*/
template<typename TenElemT, typename QNT>
double SingleSiteFiniteVMPSSweep(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
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
  for (size_t i = 0; i < N - 1; ++i) {
    LoadRelatedTensSingleSiteAlg(mps, lenvs, renvs, i, 'r',
                                 sweep_params);    // note: here we need mps[i](do not need load),
    // mps[i+1], lenvs[i](do not need load), and mps[i]'s renvs
    // mps[i]'s renvs can be removed
    actual_e0 = CalEnergyEptSingleSite(mps, mpo, lenvs, renvs, i);
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
    e0 = SingleSiteFiniteVMPSUpdate(
        mps,
        lenvs, renvs,
        mpo,
        sweep_params, 'r', i,
        noise_running
    );
    actual_laststep_e0 = actual_e0;
    DumpRelatedTensSingleSiteAlg(mps, lenvs, renvs, i, 'r',
                                 sweep_params);    // note: here we need dump mps[i](free memory),
    // lenvs[i+1](without free memory)
  }

  for (size_t i = N - 1; i > 0; --i) {
    LoadRelatedTensSingleSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params);
    actual_e0 = CalEnergyEptSingleSite(mps, mpo, lenvs, renvs, i);
    if ((actual_e0 - e0) <= 0.0) {
    } else if ((actual_e0 - e0) >= alpha * fabs(actual_laststep_e0 - e0)) {
      noise_running = noise_running * noise_decrease;
    } else {
      noise_running = std::min(noise_running * noise_increase, max_noise);
    }
    e0 = SingleSiteFiniteVMPSUpdate(
        mps,
        lenvs, renvs,
        mpo,
        sweep_params, 'l', i,
        noise_running
    );
    actual_laststep_e0 = actual_e0;
    DumpRelatedTensSingleSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params);
  }
  return e0;
}


/**  Single step for single site update.
This function includes below procedure:
- update `mps[target]` tensors according corresponding environment tensors and the mpo tensor, using lanczos algorithm;
- expand `mps[target]` and `mps[next_site]` by noise, if need;
- canonicalize mps to `mps[next_site]` by SVD, while truncate tensor `mps[target]` if need;
- generate the next environment in the direction.

When using this function, one must make sure memory at least contains `mps[target]` tensor,
its environment tensors and `mps[next_site]`.
*/
template<typename TenElemT, typename QNT>
double SingleSiteFiniteVMPSUpdate(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const char dir,
    const size_t target_site,
    const double preset_noise
) {
  Timer update_timer("single_site_fvmps_update");

  double noise = preset_noise;
  auto N = mps.size();
  size_t lenv_len = target_site;
  size_t renv_len = N - target_site - 1;
  size_t svd_ldims;
  size_t next_site;
  switch (dir) {
    case 'r':
      svd_ldims = 2;
      next_site = target_site + 1;
      break;
    case 'l':
      svd_ldims = 1;
      next_site = target_site - 1;
      break;
    default:
      std::cout << "dir must be 'r' or 'l', but " << dir << std::endl;
      exit(3);
  }

  using TenT = QLTensor<TenElemT, QNT>;
  std::vector<TenT *> eff_ham(3);
  eff_ham[0] = lenvs(lenv_len);
  eff_ham[1] = const_cast<TenT *>(mpo(target_site));    // Safe const casts for MPO local tensors.
  eff_ham[2] = renvs(renv_len);

  auto mps_ten_shape = mps[target_site].GetShape();
  Timer lancz_timer("single_site_fvmps_lancz");
  auto lancz_res = LanczosSolver(
      eff_ham,
      mps(target_site),
      &eff_ham_mul_single_site_state,
      sweep_params.lancz_params
  );                                   //note here mps(target_site) are destroyed.
#ifdef QLMPS_TIMING_MODE
  auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif

#ifdef QLMPS_TIMING_MODE
  Timer expand_timer("single_site_fvmps_add_noise");
#endif

  bool need_expand(true);
  if (fabs(noise) < 1e-15) {
    need_expand = false;
  } else if (
      (target_site < N / 2 && mps_ten_shape[0] * mps_ten_shape[1] <= mps_ten_shape[2]) ||
      (target_site > N / 2 && mps_ten_shape[2] * mps_ten_shape[1] <= mps_ten_shape[0])
      ) {
    noise = 0.0;            //just for output
    need_expand = false;
  }
  if (need_expand) {
    SingleSiteFiniteVMPSExpand(
        mps,
        lancz_res.gs_vec,
        eff_ham,
        dir,
        target_site,
        noise
    );
    delete lancz_res.gs_vec;
  } else {
    mps(target_site) = lancz_res.gs_vec;
  }

#ifdef QLMPS_TIMING_MODE
  expand_timer.PrintElapsed();
#endif

#ifdef QLMPS_TIMING_MODE
  Timer svd_timer("single_site_fvmps_svd");
#endif

  TenT u, vt;
  QLTensor<QLTEN_Double, QNT> s;
  QLTEN_Double actual_trunc_err;
  size_t D;
  auto zero_div = Div(mps[target_site]) - Div(mps[target_site]);
  auto div_left = (dir == 'r' ? Div(mps[target_site]) : zero_div);
  SVD(
      mps(target_site),
      svd_ldims, div_left,
      sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
      &u, &s, &vt, &actual_trunc_err, &D
  );
  auto ee = MeasureEE(s, D);

#ifdef QLMPS_TIMING_MODE
  svd_timer.PrintElapsed();
#endif


#ifdef QLMPS_TIMING_MODE
  Timer update_mps_ten_timer("single_site_fvmps_update_mps_ten");
#endif

  TenT *temp_ten1 = new TenT();
  TenT *temp_ten2 = new TenT();
  switch (dir) {
    case 'r':
      mps[target_site] = std::move(u);
      Contract(&s, &vt, {{1},
                         {0}}, temp_ten1);
      Contract(temp_ten1, mps(next_site), {{1},
                                           {0}}, temp_ten2);
      delete temp_ten1;
      delete mps(next_site);
      mps(next_site) = temp_ten2;
      break;
    case 'l':
      mps[target_site] = std::move(vt);
      Contract(&u, &s, {{1},
                        {0}}, temp_ten1);
      Contract(mps(next_site), temp_ten1, {{2},
                                           {0}}, temp_ten2);
      delete temp_ten1;
      delete mps(next_site);
      mps(next_site) = temp_ten2;
      break;
  }

#ifdef QLMPS_TIMING_MODE
  update_mps_ten_timer.PrintElapsed();
#endif

// Update environment tensors
#ifdef QLMPS_TIMING_MODE
  Timer update_env_ten_timer("single_site_fvmps_update_env_ten");
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

/**
  SingleSiteFiniteVMPSExpand Function
  @note gs_vec will be changed when dir=='r'
*/
template<typename TenElemT, typename QNT>
void SingleSiteFiniteVMPSExpand(
    FiniteMPS<TenElemT, QNT> &mps,
    QLTensor<TenElemT, QNT> *gs_vec,
    const std::vector<QLTensor<TenElemT, QNT> *> &eff_ham,
    const char dir,
    const size_t target_site,
    const double noise
) {
  // note: The expanded tensors are saved in mps
  using TenT = QLTensor<TenElemT, QNT>;
  TenT *ten_tmp = new TenT();
  mps(target_site) = new TenT();    // we suppose mps only contain mps[next_site]
  if (dir == 'r') {
#ifdef QLMPS_TIMING_MODE
    Timer contract_timer("single_site_fvmps_add_noise_contract");
#endif
    size_t next_site = target_site + 1;

    TenT temp_ten;
    Contract(eff_ham[0], gs_vec, {{2},
                                  {0}}, &temp_ten);
    Contract<TenElemT, QNT, true, true>(temp_ten, *eff_ham[1], 1, 0, 2, *ten_tmp);

#ifdef QLMPS_TIMING_MODE
    contract_timer.PrintElapsed();
  Timer fuse_index_timer("single_site_fvmps_add_noise_fuse_index");
#endif
    ten_tmp->FuseIndex(0, 3);
#ifdef QLMPS_TIMING_MODE
    fuse_index_timer.PrintElapsed();
  Timer scalar_multip_timer("single_site_fvmps_add_noise_scalar_multiplication");
#endif
    (*ten_tmp) *= noise;
#ifdef QLMPS_TIMING_MODE
    scalar_multip_timer.PrintElapsed();
  Timer expansion_timer("single_site_fvmps_add_noise_expansion");
#endif
    gs_vec->Transpose({2, 0, 1});
    Expand(gs_vec, ten_tmp, {0}, mps(target_site));
    mps(target_site)->Transpose({1, 2, 0});

    auto expanded_index = InverseIndex(ten_tmp->GetIndexes()[0]);
    TenT expanded_zero_ten = TenT({
                                      expanded_index,
                                      mps[next_site].GetIndexes()[1],
                                      mps[next_site].GetIndexes()[2]
                                  });
    Expand(mps(next_site), &expanded_zero_ten, {0}, ten_tmp);
    delete mps(next_site);
    mps(next_site) = ten_tmp;
#ifdef QLMPS_TIMING_MODE
    expansion_timer.PrintElapsed();
#endif
  } else if (dir == 'l') {
#ifdef QLMPS_TIMING_MODE
    Timer contract_timer("single_site_fvmps_add_noise_contract");
#endif
    size_t next_site = target_site - 1;
    Contract(gs_vec, eff_ham[2], {{2},
                                  {2}}, ten_tmp);
    InplaceContract(ten_tmp, eff_ham[1], {{1, 3},
                                          {1, 3}});
#ifdef QLMPS_TIMING_MODE
    contract_timer.PrintElapsed();
  Timer fuse_index_timer("single_site_fvmps_add_noise_fuse_index");
#endif
    ten_tmp->Transpose({0, 2, 3, 1});
    ten_tmp->FuseIndex(0, 1);
#ifdef QLMPS_TIMING_MODE
    fuse_index_timer.PrintElapsed();
  Timer scalar_multip_timer("single_site_fvmps_add_noise_scalar_multiplication");
#endif
    (*ten_tmp) *= noise;
#ifdef QLMPS_TIMING_MODE
    scalar_multip_timer.PrintElapsed();
  Timer expansion_timer("single_site_fvmps_add_noise_expansion");
#endif
    Expand(gs_vec, ten_tmp, {0}, mps(target_site));

    auto expanded_index = InverseIndex(ten_tmp->GetIndexes()[0]);
    TenT expanded_zero_ten = TenT({
                                      mps[next_site].GetIndexes()[0],
                                      mps[next_site].GetIndexes()[1],
                                      expanded_index
                                  });
    Expand(mps(next_site), &expanded_zero_ten, {2}, ten_tmp);
    delete mps(next_site);
    mps(next_site) = ten_tmp;
#ifdef QLMPS_TIMING_MODE
    expansion_timer.PrintElapsed();
#endif
  }
}


template<typename TenElemT, typename QNT>
void LoadRelatedTensSingleSiteAlg(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const char dir,
    const FiniteVMPSSweepParams &sweep_params
) {
#ifdef QLMPS_TIMING_MODE
  Timer preprocessing_timer("single_site_fvmps_preprocessing");
#endif
  auto N = mps.size();
  switch (dir) {
    case 'r':
      if (target_site == 0) {
        mps.LoadTen(
            target_site + 1,
            GenMPSTenName(sweep_params.mps_path, target_site + 1)
        );
        auto renv_len = N - target_site - 1;
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);
        RemoveFile(renv_file);
        auto lenv_len = 0;
        auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
        lenvs.LoadTen(lenv_len, lenv_file);
      } else {
        mps.LoadTen(
            target_site + 1,
            GenMPSTenName(sweep_params.mps_path, target_site + 1)
        );
        auto renv_len = N - target_site - 1;
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);
        RemoveFile(renv_file);
      }
      break;
    case 'l':
      if (target_site == N - 1) {
        mps.LoadTen(
            target_site - 1,
            GenMPSTenName(sweep_params.mps_path, target_site - 1)
        );
        auto renv_len = 0;
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);

        auto lenv_len = N - 1;
        auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
        RemoveFile(lenv_file);
      } else {
        mps.LoadTen(
            target_site - 1,
            GenMPSTenName(sweep_params.mps_path, target_site - 1)
        );
        auto lenv_len = target_site;
        auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
        lenvs.LoadTen(lenv_len, lenv_file);
        RemoveFile(lenv_file);
      }
      break;
    default:
      assert(false);
  }
#ifdef QLMPS_TIMING_MODE
  preprocessing_timer.PrintElapsed();
#endif
}


template<typename TenElemT, typename QNT>
void DumpRelatedTensSingleSiteAlg(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const char dir,
    const FiniteVMPSSweepParams &sweep_params
) {
#ifdef QLMPS_TIMING_MODE
  Timer postprocessing_timer("single_site_fvmps_postprocessing");
#endif
  auto N = mps.size();
  lenvs.dealloc(target_site);
  renvs.dealloc(N - target_site - 1);
  mps.DumpTen(
      target_site,
      GenMPSTenName(sweep_params.mps_path, target_site),
      true);
  switch (dir) {
    case 'r': {
      lenvs.DumpTen(
          target_site + 1,
          GenEnvTenName("l", target_site + 1, sweep_params.temp_path));
    }
      break;
    case 'l': {
      auto next_renv_len = N - target_site;
      renvs.DumpTen(
          next_renv_len,
          GenEnvTenName("r", next_renv_len, sweep_params.temp_path));
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
double CalEnergyEptSingleSite(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const size_t target_site
) {
#ifdef QLMPS_TIMING_MODE
  Timer cal_energy_timer("single_site_fvmps_pre_caluclate_energy");
#endif
  using TenT = QLTensor<TenElemT, QNT>;
  std::vector<TenT *> eff_ham(3);
  size_t lenv_len = target_site;
  size_t renv_len = mps.size() - target_site - 1;
  eff_ham[0] = lenvs(lenv_len);
  // Safe const casts for MPO local tensors.
  eff_ham[1] = const_cast<TenT *>(&mpo[target_site]);
  eff_ham[2] = renvs(renv_len);
  TenT *h_mul_state = eff_ham_mul_single_site_state(eff_ham, mps(target_site));
  TenT scalar_ten;
  TenT mps_ten_dag = Dag(mps[target_site]);
  Contract(h_mul_state, &mps_ten_dag, {{0, 1, 2},
                                       {0, 1, 2}}, &scalar_ten);
  delete h_mul_state;
  double energy = Real(scalar_ten());
#ifdef QLMPS_TIMING_MODE
  cal_energy_timer.PrintElapsed();
#endif
  return energy;
}
} /* qlmps */
#endif //QLMPS_ALGORITM_VMPS_ONE_SITE_UPDATE_FINITE_VMPS_IMPL_H
