// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-12-21
*
* Description: QuantumLiquids/UltraDMRG project. Two-site update finite size TDVP with MPI Parallel, slave side.
*/

#ifndef QLMPS_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_IMPL_H
#define QLMPS_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_IMPL_H

#include "qlmps/algo_mpi/mps_algo_order.h"                            //kMasterRank, Order...
#include "qlmps/algo_mpi/tdvp/two_site_update_finite_tdvp_mpi.h" //MPITDVPSweepParams
#include "qlmps/algorithm/tdvp/tdvp_evolve_params.h"    //DynamicMeasuRes..
#include "lanczos_expmv_solver_mpi.h"             //MasterLanczosExpmvSolver, SlaveLanczosExpmvSolver
#include "qlmps/algo_mpi/env_tensor_update_slave.h"

namespace qlmps {
using namespace qlten;
namespace mpi = boost::mpi;

//Forward declarations.
template<typename TenElemT, typename QNT, char dir>
void MasterTwoSiteFiniteTDVPEvolution(
    FiniteMPS<TenElemT, QNT> &,
    TenVec<QLTensor<TenElemT, QNT>> &,
    TenVec<QLTensor<TenElemT, QNT>> &,
    const MPO<QLTensor<TenElemT, QNT>> &,
    const TDVPEvolveParams<QNT> &,
    const size_t,
    mpi::communicator &
);

template<typename TenElemT, typename QNT>
void MasterSingleSiteFiniteTDVPBackwardEvolution(
    FiniteMPS<TenElemT, QNT> &,
    TenVec<QLTensor<TenElemT, QNT>> &,
    TenVec<QLTensor<TenElemT, QNT>> &,
    const MPO<QLTensor<TenElemT, QNT>> &,
    const TDVPEvolveParams<QNT> &,
    const size_t,
    mpi::communicator &
);

template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteTDVP(const MPO<QLTensor<TenElemT, QNT>> &, mpi::communicator &);


/**
 *
 * @tparam TenElemT     should be QLTEN_Complex
 * @tparam QNT
 * @param mps
 * @param mpo
 * @param sweep_params
 * @param measure_file_base_name
 * @param world
 * @return
 */
template<typename TenElemT, typename QNT>
DynamicMeasuRes<TenElemT> TwoSiteFiniteTDVP(
    FiniteMPS<TenElemT, QNT> &mps, //initial state, empty, saved in disk
    const MPO<QLTensor<TenElemT, QNT>> &mpo, //saved in memory,
    const MPITDVPSweepParams<QNT> &sweep_params,
    const std::string &measure_file_base_name,
    mpi::communicator &world
) {
  const size_t N = mps.size();
  DynamicMeasuRes<TenElemT> measure_res(N * (sweep_params.step + 1));
  if (world.rank() == kMasterRank) {
    measure_res = MasterTwoSiteFiniteTDVP(mps, mpo, sweep_params, measure_file_base_name, world);
  } else {
    SlaveTwoSiteFiniteTDVP<TenElemT, QNT>(mpo, world);
  }
  return measure_res;
}

template<typename TenElemT, typename QNT>
DynamicMeasuRes<TenElemT> MasterTwoSiteFiniteTDVP(
    FiniteMPS<TenElemT, QNT> &mps, //initial state, empty, saved in disk
    const MPO<QLTensor<TenElemT, QNT>> &mpo, //saved in memory,
    const MPITDVPSweepParams<QNT> &sweep_params,
    const std::string measure_file_base_name,
    mpi::communicator &world
) {
  assert(world.rank() == kMasterRank);
  assert(mps.size() == mpo.size());

  MasterBroadcastOrder(program_start, world);
  for (size_t node = 1; node < world.size(); node++) {
    int node_num;
    world.recv(node, 2 * node, node_num);
    if (node_num == node) {
      std::cout << "Node " << node << " received the program start order." << std::endl;
    } else {
      std::cout << "unexpected " << std::endl;
      exit(1);
    }
  }
  const size_t site_0 = sweep_params.site_0;
  std::cout << "\nPlease make sure the central of mps in disk is less than " << site_0 << "." << "\n";

  if (!IsPathExist(sweep_params.initial_mps_path)) {
    CreatPath(sweep_params.initial_mps_path);
    for (size_t i = 0; i < mps.size(); i++) {
      mps.LoadTen(i, GenMPSTenName(sweep_params.mps_path, i));
      mps.DumpTen(i, GenMPSTenName(sweep_params.initial_mps_path, i), true);
    }
  }

  ActOperatorOnMps(sweep_params.op0, sweep_params.inst0, site_0, mps);
  mps.LoadTen(sweep_params.site_0, GenMPSTenName(sweep_params.mps_path, sweep_params.site_0));
  for (size_t i = site_0; i > 0; i--) {
    mps.LoadTen(i - 1, GenMPSTenName(sweep_params.mps_path, i - 1));
    mps.RightCanonicalizeTen(i);
    mps.DumpTen(i, GenMPSTenName(sweep_params.mps_path, i), true);
  }
  double mps_norm = mps(0)->Normalize();
  mps.DumpTen(0, GenMPSTenName(sweep_params.mps_path, 0), true);

  if (!IsPathExist(sweep_params.temp_path)) {
    CreatPath(sweep_params.temp_path);
  }
  InitEnvs(mps, mpo, FiniteVMPSSweepParams(sweep_params), 1);

  if (!IsPathExist(sweep_params.measure_temp_path)) {
    CreatPath(sweep_params.measure_temp_path);
  }

  std::cout << "\n";
  const size_t N = mps.size();
  DynamicMeasuRes<TenElemT> measure_res(N * (sweep_params.step + 1));

  double time = 0.0;
  std::vector<TenElemT> correlation = CalPsi1OpPsi2(mps.GetSitesInfo(), sweep_params);
  for (size_t i = 0; i < N; i++) {
    measure_res[i].times = {0.0, time};
    measure_res[i].sites = {site_0, i};
    measure_res[i].avg = mps_norm * correlation[i];
  }
  for (size_t step = 0; step < sweep_params.step; step++) {
    std::cout << "step = " << step << "\n";
    Timer sweep_timer("sweep");

    TwoSiteFiniteTDVPSweep(mps.GetSitesInfo(), mpo, sweep_params, world);

    sweep_timer.PrintElapsed();

    Timer measure_timer("measure");
    time = (step + 1) * sweep_params.tau;
    correlation = CalPsi1OpPsi2(mps.GetSitesInfo(), sweep_params);

    for (size_t i = 0; i < N; i++) {
      measure_res[(step + 1) * N + i].times = {0.0, time};
      measure_res[(step + 1) * N + i].sites = {site_0, i};
      measure_res[(step + 1) * N + i].avg = mps_norm * correlation[i] * std::exp(QLTEN_Complex(0.0, sweep_params.e0) * time);
    }
    measure_timer.PrintElapsed();
    std::cout << "\n";
  }
  DumpMeasuRes(measure_res, measure_file_base_name);
  MasterBroadcastOrder(program_final, world);
  return measure_res;
}

template<typename TenElemT, typename QNT>
void TwoSiteFiniteTDVPSweep(
    const SiteVec<TenElemT, QNT> &site_vec,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const MPITDVPSweepParams<QNT> &sweep_params,
    mpi::communicator &world
) {
  //world.rank() == 0
  using TenT = QLTensor<TenElemT, QNT>;
  FiniteMPS<TenElemT, QNT> mps(site_vec);
  auto N = mps.size();
  TenVec<TenT> lenvs(N - 1);
  TenVec<TenT> renvs(N - 1);
  FiniteVMPSSweepParams sweep_params_temp = FiniteVMPSSweepParams(sweep_params); //used to load and dump data
  MPITDVPSweepParams<QNT> sweep_params_full_step = sweep_params;
  sweep_params_full_step.tau = sweep_params.tau * 2; // used in last two site

  mps.LoadTen(
      1,
      GenMPSTenName(sweep_params.mps_path, 1)
  );
  for (size_t i = 0; i < N - 2; ++i) {
    LoadRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'r', sweep_params_temp);
    MasterTwoSiteFiniteTDVPEvolution<TenElemT, QNT, 'r'>(mps, lenvs, renvs, mpo, sweep_params, i, world);
    MasterSingleSiteFiniteTDVPBackwardEvolution<TenElemT, QNT>(mps, lenvs, renvs, mpo, sweep_params, i + 1, world);
    DumpRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'r', sweep_params_temp);
  }

  for (size_t i = N - 1; i > 1; --i) {
    LoadRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params);
    if (i == N - 1) {
      MasterTwoSiteFiniteTDVPEvolution<TenElemT, QNT, 'l'>(mps, lenvs, renvs, mpo, sweep_params_full_step, i, world);
    } else {
      MasterTwoSiteFiniteTDVPEvolution<TenElemT, QNT, 'l'>(mps, lenvs, renvs, mpo, sweep_params, i, world);
    }
    MasterSingleSiteFiniteTDVPBackwardEvolution<TenElemT, QNT>(mps, lenvs, renvs, mpo, sweep_params, i - 1, world);
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
void MasterTwoSiteFiniteTDVPEvolution(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const TDVPEvolveParams<QNT> &sweep_params,
    const size_t target_site,
    mpi::communicator &world
) {
  static_assert((dir == 'r' || dir == 'l'),
                "Direction template parameter of function TwoSiteFiniteTDVPEvolution is wrong.");
  //master process
  Timer update_timer("two_site_ftdvp_update");
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
  //TODO: subroutine for lanczos
  MasterBroadcastOrder(lanczos, world);
  broadcast(world,lsite_idx, kMasterRank);
  auto lancz_res = MasterLanczosExpmvSolver(
      eff_ham, init_state,
      sweep_params.tau / 2,
      sweep_params.lancz_params,
      world
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
  MasterBroadcastOrder(svd, world);
  MPISVDMaster(
      lancz_res.expmv,
      svd_ldims, Div(mps[lsite_idx]),
      sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
      &u, &s, &vt, &actual_trunc_err, &D,
      world
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
      MasterBroadcastOrder(growing_left_env, world);
      lenvs(lenv_len + 1) = MasterGrowLeftEnvironment(lenvs[lenv_len], mpo[target_site], mps[target_site], world);
    }
      break;
    case 'l': {
      MasterBroadcastOrder(growing_right_env, world);
      renvs(renv_len + 1) = MasterGrowRightEnvironment(*eff_ham[3], mpo[target_site], mps[target_site], world);
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

/// no svd
template<typename TenElemT, typename QNT>
void MasterSingleSiteFiniteTDVPBackwardEvolution(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<QLTensor<TenElemT, QNT>> &lenvs,
    TenVec<QLTensor<TenElemT, QNT>> &renvs,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const TDVPEvolveParams<QNT> &sweep_params,
    const size_t target_site,
    mpi::communicator &world
) {
  Timer update_timer("single_site_ftdvp_update");
  auto N = mps.size();
  size_t lenv_len = target_site;
  size_t renv_len = N - target_site - 1;

  using TenT = QLTensor<TenElemT, QNT>;
  std::vector<TenT *> eff_ham(3);
  eff_ham[0] = lenvs(lenv_len);
  eff_ham[1] = const_cast<TenT *>(mpo(target_site));    // Safe const casts for MPO local tensors.
  eff_ham[2] = renvs(renv_len);

  Timer lancz_timer("single_site_ftdvp_lancz");
  //TODO: parallel
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

  auto update_elapsed_time = update_timer.Elapsed();
  std::cout << " Site " << std::setw(4) << target_site
            << " Iter = " << std::setw(3) << lancz_res.iters
            << " LanczT = " << std::setw(8) << lancz_elapsed_time
            << " TotT = " << std::setw(8) << update_elapsed_time;
  std::cout << std::scientific << std::endl;
}

}

#endif //QLMPS_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_IMPL_H