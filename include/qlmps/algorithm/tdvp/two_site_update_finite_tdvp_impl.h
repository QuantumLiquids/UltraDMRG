// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021/11/1.
*
* Description: QuantumLiquids/UltraDMRG project. Implementation details for two site tdvp.
*/

#ifndef QLMPS_ALGORITHM_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_IMPL_H
#define QLMPS_ALGORITHM_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_IMPL_H

#include "qlten/qlten.h"
#include "qlmps/algorithm/tdvp/tdvp_evolve_params.h"    // TDVPEvolveParams
#include "qlmps/one_dim_tn/mpo/mpo.h"                            // MPO
#include "qlmps/one_dim_tn/mps/finite_mps/finite_mps.h"          // FiniteMPS
#include "qlmps/utilities.h"                                     // IsPathExist, CreatPath
#include "lanczos_expmv_solver_impl.h"           // LanczosExpmvSolver
#include "qlmps/one_dim_tn/mps/finite_mps/finite_mps_measu.h"    // DumpSites, DumpAvgVal
#include "qlmps/algorithm/vmps/two_site_update_finite_vmps_impl.h" //InitEnvs

namespace qlmps {
using namespace qlten;

template<typename TenElemT, typename QNT>
void ActOperatorOnMps(
    const QLTensor<TenElemT, QNT> &op,
    const QLTensor<TenElemT, QNT> &inst,
    const size_t site,
    FiniteMPS<TenElemT, QNT> &mps, //empty
    const std::string &mps_path
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT *res = new TenT();
  mps.LoadTen(site, GenMPSTenName(mps_path, site));
  Contract(&op, mps(site), {{0}, {1}}, res);
  res->Transpose({1, 0, 2});
  delete mps(site);
  mps(site) = res;
  mps.DumpTen(site, GenMPSTenName(mps_path, site), true);

  if (inst == TenT()) {
    return;
  }

  for (long i = site - 1; i >= 0; i--) {
    res = new TenT();
    mps.LoadTen(i, GenMPSTenName(mps_path, i));
    Contract(&inst, mps(i), {{0}, {1}}, res);
    res->Transpose({1, 0, 2});
    delete mps(i);
    mps(i) = res;
    mps.DumpTen(i, GenMPSTenName(mps_path, i), true);
  }
}

template<typename TenElemT, typename QNT>
void ActMpoOnMps(
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    FiniteMPS<TenElemT, QNT> &mps, //empty
    const std::string &mps_path
) {
  using TenT = QLTensor<TenElemT, QNT>;
  for (size_t i = 0; i < mps.size(); i++) {
    TenT *res = new TenT();
    mps.LoadTen(i, GenMPSTenName(mps_path, i));
    Contract(mps(i), {1}, mpo(i), {1}, res);
    res->FuseIndex(0, 2);
    res->FuseIndex(1, 3);
    res->Transpose({1, 2, 0});
    delete mps(i);
    mps(i) = res;
    mps.DumpTen(i, GenMPSTenName(mps_path, i), true);
  }
}

template<typename TenElemT, typename QNT, char dir>
void TwoSiteFiniteTDVPEvolution(
    FiniteMPS<TenElemT, QNT> &,
    TenVec<QLTensor<TenElemT, QNT>> &,
    TenVec<QLTensor<TenElemT, QNT>> &,
    const MPO<QLTensor<TenElemT, QNT>> &,
    const TDVPEvolveParams<QNT> &,
    const size_t
);

template<typename TenElemT, typename QNT, char dir>
void SingleSiteFiniteTDVPBackwardEvolution(
    FiniteMPS<TenElemT, QNT> &,
    TenVec<QLTensor<TenElemT, QNT>> &,
    TenVec<QLTensor<TenElemT, QNT>> &,
    const MPO<QLTensor<TenElemT, QNT>> &,
    const TDVPEvolveParams<QNT> &,
    const size_t
);

template<typename TenElemT, typename QNT>
std::vector<TenElemT> CalPsi1OpPsi2(
    const SiteVec<TenElemT, QNT> &,
    const TDVPEvolveParams<QNT> &
);

template<typename AvgT>
struct DynamicMeasuResElem {
  DynamicMeasuResElem(void) = default;
  DynamicMeasuResElem(const std::vector<size_t> &sites,
                      const std::vector<double> &times,
                      const AvgT avg) :
      sites(sites), times(times), avg(avg) {}
  std::vector<size_t> sites;  ///< Site indexes of the operators.
  std::vector<double> times;
  AvgT avg;                 ///< average of the observation.
};

template<typename AvgT>
using DynamicMeasuRes = std::vector<DynamicMeasuResElem<AvgT>>;

template<typename AvgT>
void DumpMeasuRes(
    const DynamicMeasuRes<AvgT> &res,
    const std::string &basename
);

template<typename TenElemT, typename QNT>
DynamicMeasuRes<TenElemT> TwoSiteFiniteTDVP(
    FiniteMPS<TenElemT, QNT> &mps, //initial state, empty, saved in disk
    const MPO<QLTensor<TenElemT, QNT>> &ham_mpo, //saved in memory,
    const TDVPEvolveParams<QNT> &sweep_params,
    const std::string measure_file_base_name
) {
  //TenElemT == QLTEN_Complex
  assert(mps.size() == ham_mpo.size());
  const size_t site_0 = sweep_params.site_0;
  std::cout << "Please make sure the central of mps in disk is less than " << site_0 << "." << "\n";

  if (!IsPathExist(sweep_params.initial_mps_path)) {
    //cp the directory
    CreatPath(sweep_params.initial_mps_path);
    for (size_t i = 0; i < mps.size(); i++) {
      mps.LoadTen(i, GenMPSTenName(sweep_params.mps_path, i));
      mps.DumpTen(i, GenMPSTenName(sweep_params.initial_mps_path, i), true);
    }
  }
  ActOperatorOnMps(sweep_params.local_op0, sweep_params.inst0, site_0, mps, sweep_params.mps_path);
//  ActMpoOnMps(sweep_params.op0_mpo, mps, sweep_params.mps_path);

  size_t right_canonicalize_start_site = sweep_params.local_op_corr ? sweep_params.site_0 : (mps.size() - 1);
  mps.LoadTen(right_canonicalize_start_site, GenMPSTenName(sweep_params.mps_path, sweep_params.site_0));
  for (size_t i = right_canonicalize_start_site; i > 0; i--) {
    mps.LoadTen(i - 1, GenMPSTenName(sweep_params.mps_path, i - 1));
    mps.RightCanonicalizeTen(i);
    mps.DumpTen(i, GenMPSTenName(sweep_params.mps_path, i), true);
  }
  mps.DumpTen(0, GenMPSTenName(sweep_params.mps_path, 0), true);

  if (!IsPathExist(sweep_params.temp_path)) {
    CreatPath(sweep_params.temp_path);
  }
  InitEnvs(mps, ham_mpo, FiniteVMPSSweepParams(sweep_params), 1);

  if (!IsPathExist(sweep_params.measure_temp_path)) {
    CreatPath(sweep_params.measure_temp_path);
  }

  std::cout << "\n";

  const size_t N = mps.size();
  const size_t measure_res_num = sweep_params.local_op_corr ? (N * (sweep_params.step + 1)) : (sweep_params.step + 1);
  DynamicMeasuRes<TenElemT> measure_res(measure_res_num);

  double time = 0.0;
  // equal time correlation
  std::vector<TenElemT> correlation = CalPsi1OpPsi2(mps.GetSitesInfo(), sweep_params);
  for (size_t i = 0; i < N; i++) {
    measure_res[i].times = {0.0, time};
    measure_res[i].sites = {site_0, i};
    measure_res[i].avg = correlation[i];
  }

  for (size_t step = 0; step < sweep_params.step; step++) {
    std::cout << "step = " << step << "\n";
    Timer sweep_timer("sweep");

    TwoSiteFiniteTDVPSweep(mps.GetSitesInfo(), ham_mpo, sweep_params);
    sweep_timer.PrintElapsed();

    Timer measure_timer("measure");
    time = (step + 1) * sweep_params.tau;
    correlation = CalPsi1OpPsi2(mps.GetSitesInfo(), sweep_params);

    for (size_t i = 0; i < N; i++) {
      measure_res[(step + 1) * N + i].times = {0.0, time};
      measure_res[(step + 1) * N + i].sites = {site_0, i};
      measure_res[(step + 1) * N + i].avg =
          correlation[i] * qlmps::complex_exp(QLTEN_Complex(0.0, sweep_params.e0) * time);
    }
    measure_timer.PrintElapsed();
    std::cout << "\n";
  }
  DumpMeasuRes(measure_res, measure_file_base_name);

  return measure_res;
}

template<typename TenElemT, typename QNT>
void TwoSiteFiniteTDVPSweep(
    const SiteVec<TenElemT, QNT> &site_vec,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const TDVPEvolveParams<QNT> &sweep_params
) {

  using TenT = QLTensor<TenElemT, QNT>;

  FiniteMPS<TenElemT, QNT> mps(site_vec);
  auto N = mps.size();
  TenVec<TenT> lenvs(N - 1);
  TenVec<TenT> renvs(N - 1);

  FiniteVMPSSweepParams sweep_params_temp = FiniteVMPSSweepParams(sweep_params); //used to load and dump data
  TDVPEvolveParams<QNT> sweep_params_full_step = sweep_params;
  sweep_params_full_step.tau = sweep_params.tau * 2; // used in last two site

  mps.LoadTen(
      1,
      GenMPSTenName(sweep_params.mps_path, 1)
  );

  for (size_t i = 0; i < N - 2; ++i) {
    // Load to-be-used tensors
    LoadRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'r', sweep_params_temp);
    TwoSiteFiniteTDVPEvolution<TenElemT, QNT, 'r'>(mps, lenvs, renvs, mpo, sweep_params, i);
    SingleSiteFiniteTDVPBackwardEvolution<TenElemT, QNT, 'n'>(mps, lenvs, renvs, mpo, sweep_params, i + 1);
    // Dump related tensor to HD and remove unused tensor from RAM
    DumpRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'r', sweep_params_temp);
  }

  for (size_t i = N - 1; i > 1; --i) {
    LoadRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params);
    if (i == N - 1) {
      TwoSiteFiniteTDVPEvolution<TenElemT, QNT, 'l'>(mps, lenvs, renvs, mpo, sweep_params_full_step, i);
    } else {
      TwoSiteFiniteTDVPEvolution<TenElemT, QNT, 'l'>(mps, lenvs, renvs, mpo, sweep_params, i);
    }
    SingleSiteFiniteTDVPBackwardEvolution<TenElemT, QNT, 'n'>(mps, lenvs, renvs, mpo, sweep_params, i - 1);
    DumpRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params);
  }

  mps.LoadTen(
      0,
      GenMPSTenName(sweep_params.mps_path, 0)
  );
  lenvs.LoadTen(
      0,
      GenEnvTenName("l", 0, sweep_params.temp_path)
  );
  TwoSiteFiniteTDVPEvolution<TenElemT, QNT, 'r'>(mps, lenvs, renvs, mpo, sweep_params, 0);
  mps.DumpTen(
      0,
      GenMPSTenName(sweep_params.mps_path, 0),
      true
  );
  mps.DumpTen(
      1,
      GenMPSTenName(sweep_params.mps_path, 1),
      true
  );

  lenvs.dealloc(0);
  lenvs.dealloc(1);
  renvs.dealloc(N - 2);
}

template<typename TenElemT, typename QNT, char dir>
void TwoSiteFiniteTDVPEvolution(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const TDVPEvolveParams<QNT> &sweep_params,
    const size_t target_site
) {
  static_assert((dir == 'r' || dir == 'l'),
                "Direction template parameter of function TwoSiteFiniteTDVPEvolution is wrong.");
  Timer update_timer("two_site_ftdvp_update");
  // Assign some parameters
  auto N = mps.size();
  std::vector<std::vector<size_t>> init_state_ctrct_axes;
  size_t svd_ldims;
  size_t lsite_idx, rsite_idx;
  size_t lenv_len, renv_len;
  std::string lblock_file, rblock_file;
  init_state_ctrct_axes = {{2}, {0}};
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
  Timer lancz_timer("two_site_ftdvp_lancz");
  ExpmvRes<TenT> lancz_res = LanczosExpmvSolver(
      eff_ham, init_state,
      &eff_ham_mul_two_site_state,
      sweep_params.tau / 2,
      sweep_params.lancz_params
  );
#ifdef QLMPS_TIMING_MODE
  auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif

  // SVD and measure entanglement entropy
#ifdef QLMPS_TIMING_MODE
  Timer svd_timer("two_site_ftdvp_svd");
#endif

  TenT u, vt;
  using DTenT = QLTensor<QLTEN_Double, QNT>;
  DTenT s;
  QLTEN_Double actual_trunc_err;
  size_t D;
  SVD(
      lancz_res.expmv,
      svd_ldims, Div(mps[lsite_idx]),
      sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
      &u, &s, &vt, &actual_trunc_err, &D
  );
  delete lancz_res.expmv;
  auto ee = MeasureEE(s, D);

#ifdef QLMPS_TIMING_MODE
  svd_timer.PrintElapsed();
#endif

  // Update MPS local tensor
#ifdef QLMPS_TIMING_MODE
  Timer update_mps_ten_timer("two_site_ftdvp_update_mps_ten");
#endif

  TenT the_other_mps_ten;
  switch (dir) {
    case 'r':mps[lsite_idx] = std::move(u);
      Contract(&s, &vt, {{1}, {0}}, &the_other_mps_ten);
      mps[rsite_idx] = std::move(the_other_mps_ten);
      break;
    case 'l':Contract(&u, &s, {{2}, {0}}, &the_other_mps_ten);
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
  Timer update_env_ten_timer("two_site_ftdvp_update_env_ten");
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
  }

#ifdef QLMPS_TIMING_MODE
  update_env_ten_timer.PrintElapsed();
#endif

  auto update_elapsed_time = update_timer.Elapsed();
  std::cout << " Site " << std::setw(4) << target_site
            << " TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
            << " D = " << std::setw(5) << D
            << " Iter = " << std::setw(3) << lancz_res.iters
            << " LanczT = " << std::setw(8) << lancz_elapsed_time
            << " TotT = " << std::setw(8) << update_elapsed_time
            << " S = " << std::setw(10) << std::setprecision(7) << ee;
  std::cout << std::scientific << std::endl;
}

template<typename TenElemT, typename QNT, char dir = 'n'>
void SingleSiteFiniteTDVPBackwardEvolution(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const TDVPEvolveParams<QNT> &sweep_params,
    const size_t target_site
) {
  Timer update_timer("single_site_ftdvp_update");

  auto N = mps.size();
  size_t lenv_len = target_site;
  size_t renv_len = N - target_site - 1;
  size_t svd_ldims;
  size_t next_site;
  switch (dir) {
    case 'r':svd_ldims = 2;
      next_site = target_site + 1;
      break;
    case 'l':svd_ldims = 1;
      next_site = target_site - 1;
      break;
    case 'n':break;
    default:std::cout << "dir must be 'r' or 'l', but " << dir << std::endl;
      exit(3);
  }

  using TenT = QLTensor<TenElemT, QNT>;
  std::vector<TenT *> eff_ham(3);
  eff_ham[0] = lenvs(lenv_len);
  eff_ham[1] = const_cast<TenT *>(mpo(target_site));    // Safe const casts for MPO local tensors.
  eff_ham[2] = renvs(renv_len);

  Timer lancz_timer("single_site_ftdvp_lancz");
  ExpmvRes<TenT> lancz_res = LanczosExpmvSolver(
      eff_ham, mps(target_site),
      &eff_ham_mul_single_site_state,
      -sweep_params.tau / 2,
      sweep_params.lancz_params
  );   //note here mps(target_site) are destroyed.
#ifdef QLMPS_TIMING_MODE
  auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif

  mps(target_site) = lancz_res.expmv;

#ifdef QLMPS_TIMING_MODE
  Timer svd_timer("single_site_ftdvp_svd");
#endif

  if (dir != 'n') {
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
    Timer update_mps_ten_timer("single_site_ftdvp_update_mps_ten");
#endif

    TenT *temp_ten1 = new TenT();
    TenT *temp_ten2 = new TenT();
    switch (dir) {
      case 'r':mps[target_site] = std::move(u);
        Contract(&s, &vt, {{1}, {0}}, temp_ten1);
        Contract(temp_ten1, mps(next_site), {{1}, {0}}, temp_ten2);
        delete temp_ten1;
        delete mps(next_site);
        mps(next_site) = temp_ten2;
        break;
      case 'l':mps[target_site] = std::move(vt);
        Contract(&u, &s, {{1}, {0}}, temp_ten1);
        Contract(mps(next_site), temp_ten1, {{2}, {0}}, temp_ten2);
        delete temp_ten1;
        delete mps(next_site);
        mps(next_site) = temp_ten2;
        break;
      case 'n':break;
    }

#ifdef QLMPS_TIMING_MODE
    update_mps_ten_timer.PrintElapsed();
#endif

// Update environment tensors
#ifdef QLMPS_TIMING_MODE
    Timer update_env_ten_timer("single_site_ftdvp_update_env_ten");
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
      default:assert(false);
    }
#ifdef QLMPS_TIMING_MODE
    update_env_ten_timer.PrintElapsed();
#endif
  }
  auto update_elapsed_time = update_timer.Elapsed();
  std::cout << " Site " << std::setw(4) << target_site
            << " Iter = " << std::setw(3) << lancz_res.iters
            << " LanczT = " << std::setw(8) << lancz_elapsed_time
            << " TotT = " << std::setw(8) << update_elapsed_time;
  std::cout << std::scientific << std::endl;
}

/*
 * <Psi_2 | Op | Psi_1>
 * where |Psi_1> is the evolved state,
 * and |Psi_2> is the initial state.
 * The states are read from disk.
 *
 */
template<typename TenElemT, typename QNT>
std::vector<TenElemT> CalPsi1OpPsi2(
    const SiteVec<TenElemT, QNT> &site_vec,
    const TDVPEvolveParams<QNT> &sweep_params
) {
  const std::string mps1_path = sweep_params.mps_path;
  const std::string mps2_path = sweep_params.initial_mps_path;
  const std::string temp_path = sweep_params.measure_temp_path;
  FiniteMPS<TenElemT, QNT> mps1(site_vec);
  FiniteMPS<TenElemT, QNT> mps2(site_vec);
  const size_t N = mps1.size();
  using TenT = QLTensor<TenElemT, QNT>;
  const TenT &Op = sweep_params.local_op1;
  const TenT &inst = sweep_params.inst1;

  std::vector<TenElemT> res(N);

  mps1.LoadTen(N - 1, GenMPSTenName(mps1_path, N - 1));
  mps2.LoadTen(N - 1, GenMPSTenName(mps2_path, N - 1));

  TenT right_boundary_tensor = TenT({mps2.back().GetIndexes()[2],
                                     InverseIndex(mps1.back().GetIndexes()[2])
                                    });
  right_boundary_tensor({0, 0}) = 1.0;
  std::string file = GenEnvTenName("r", 0, temp_path);
  WriteQLTensorTOFile(right_boundary_tensor, file);

  for (size_t i = 1; i <= N - 1; i++) {
    if (i > 1) {
      mps1.LoadTen(N - i, GenMPSTenName(mps1_path, N - i));
      mps2.LoadTen(N - i, GenMPSTenName(mps2_path, N - i));
    }
    file = GenEnvTenName("r", i, temp_path);
    TenT temp;

    TenT mps2_dag = Dag(mps2[N - i]);
    Contract(&mps2_dag, &right_boundary_tensor, {{2}, {0}}, &temp);
    right_boundary_tensor = TenT();
    Contract(&temp, mps1(N - i), {{1, 2}, {1, 2}}, &right_boundary_tensor);
    WriteQLTensorTOFile(right_boundary_tensor, file);
    mps1.dealloc(N - i);
    mps2.dealloc(N - i);
  }

  mps1.LoadTen(0, GenMPSTenName(mps1_path, 0));
  mps2.LoadTen(0, GenMPSTenName(mps2_path, 0));
  TenT left_boundary_tensor = TenT({
                                       mps2.front().GetIndexes()[0],
                                       InverseIndex(mps1.front().GetIndexes()[0])
                                   });
  left_boundary_tensor({0, 0}) = 1.0;

  for (size_t i = 0; i < N; i++) {
    //get renv for site i
    if (i > 0) {
      right_boundary_tensor = TenT();
      file = GenEnvTenName("r", N - i - 1, temp_path);
      ReadQLTensorFromFile(right_boundary_tensor, file);
      RemoveFile(file);
    }

    //calculate res[i];
    if (i > 0) {
      mps1.LoadTen(i, GenMPSTenName(mps1_path, i));
      mps2.LoadTen(i, GenMPSTenName(mps2_path, i));
    }
    TenT mps2_dag = Dag(mps2[i]);
    TenT temp1, temp2, temp3, temp4;
    Contract(&mps2_dag, &left_boundary_tensor, {{0}, {0}}, &temp1);
    Contract(&Op, &temp1, {{1}, {0}}, &temp2);
    Contract(&temp2, mps1(i), {{2, 0}, {0, 1}}, &temp3);
    Contract(&temp3, &right_boundary_tensor, {{0, 1}, {0, 1}}, &temp4);
    res[i] = temp4();

    // update lenv for next site
    if (i < N - 1) {
      left_boundary_tensor = TenT();
      if (inst == TenT()) {
        Contract(&temp1, mps1(i), {{2, 0}, {0, 1}}, &left_boundary_tensor);
      } else {
        temp2 = TenT();
        Contract(&inst, &temp1, {{1}, {0}}, &temp2);
        Contract(&temp2, mps1(i), {{2, 0}, {0, 1}}, &left_boundary_tensor);
      }
    }
    mps1.dealloc(i);
    mps2.dealloc(i);
  }

  return res;
}

// Data dump.
template<typename AvgT>
void DumpMeasuRes(
    const DynamicMeasuRes<AvgT> &res,
    const std::string &basename
) {
  auto file = basename + ".json";
  std::ofstream ofs(file);

  ofs << "[\n";

  for (auto it = res.begin(); it != res.end(); ++it) {
    ofs << "  [";

    DumpSites(ofs, it->sites);
    DumpSites(ofs, it->times);
    DumpAvgVal(ofs, it->avg);

    if (it == res.end() - 1) {
      ofs << "]\n";
    } else {
      ofs << "],\n";
    }
  }

  ofs << "]";

  ofs.close();
}

}//qlmps




#endif //QLMPS_ALGORITHM_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_IMPL_H
