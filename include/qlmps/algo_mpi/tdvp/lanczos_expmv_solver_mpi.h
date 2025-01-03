// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-12-21
*
* Description: QuantumLiquids/UltraDMRG project. Lanczos solver based on distributed memory parallel.
*/

#ifndef QLMPS_ALGO_MPI_LANCZOS_EXPMV_SOLVER_MPI_H
#define QLMPS_ALGO_MPI_LANCZOS_EXPMV_SOLVER_MPI_H

namespace qlmps {
using namespace qlten;
namespace mpi = boost::mpi;

//Forward deceleration
template<typename ElemT, typename QNT>
QLTEN_Double master_two_site_eff_ham_mul_state(
    const std::vector<QLTensor<ElemT, QNT> *> &,
    QLTensor<ElemT, QNT> *,
    QLTensor<ElemT, QNT> &,
    mpi::communicator
);

template<typename ElemT, typename QNT>
void slave_two_site_eff_ham_mul_state(
    const std::vector<QLTensor<ElemT, QNT> *> &,
    mpi::communicator
);

template<typename TenT>
ExpmvRes<TenT> MasterLanczosExpmvSolver(
    const std::vector<TenT *> &rpeff_ham,
    TenT *pinit_state,
    const double step_length,
    const LanczosParams &params,
    mpi::communicator &world
) {
  const size_t eff_ham_eff_dim = pinit_state->size();
  const size_t eff_ham_size = pinit_state->Rank();
#ifdef QLMPS_TIMING_MODE
  Timer broadcast_eff_ham_timer("broadcast_eff_ham_send");
#endif
  SendBroadCastQLTensor(world, (*rpeff_ham[0]), kMasterRank);
  SendBroadCastQLTensor(world, (*rpeff_ham[eff_ham_size - 1]), kMasterRank);

#ifdef QLMPS_TIMING_MODE
  broadcast_eff_ham_timer.PrintElapsed();
  Timer lancozs_after_send_ham("lancz_total_time_after_send_ham");
#endif

  ExpmvRes<TenT> expmv_res;
  std::vector<std::vector<size_t>> energy_measu_ctrct_axes;
  if (eff_ham_size == 3) {            // single site update
    energy_measu_ctrct_axes = {{0, 1, 2}, {0, 1, 2}};
  } else if (pinit_state->Rank() == 4) {    // two site update
    energy_measu_ctrct_axes = {{0, 1, 2, 3}, {0, 1, 2, 3}};
  }

  std::vector < TenT * > bases(params.max_iterations);
  std::vector<QLTEN_Double> a(params.max_iterations, 0.0);
  std::vector<QLTEN_Double> b(params.max_iterations, 0.0);

#ifdef QLMPS_TIMING_MODE
  Timer normalize_timer("lancz_normlize");
#endif
  pinit_state->Normalize();
  bases[0] = pinit_state;
#ifdef QLMPS_TIMING_MODE
  normalize_timer.PrintElapsed();
  Timer mat_vec_timer("lancz_mat_vec_and_overlap");
#endif

  TenT *last_mat_mul_vec_res = new TenT();
  a[0] = master_two_site_eff_ham_mul_state(
      rpeff_ham,
      bases[0],
      *last_mat_mul_vec_res,
      world
  );
#ifdef QLMPS_TIMING_MODE
  mat_vec_timer.PrintElapsed();
#endif
  size_t m = 0;

  QLTEN_Complex *combination_factor = new QLTEN_Complex[params.max_iterations];//combination of bases
  QLTEN_Complex *last_combination_factor = new QLTEN_Complex[params.max_iterations];

  while (true) {
    m += 1;
#ifdef QLMPS_TIMING_MODE
    Timer linear_combine_timer("lancz_linear_combine");
#endif
    TenT *gamma = last_mat_mul_vec_res;
    if (m == 1) {
      LinearCombine({-a[m - 1]}, {bases[m - 1]}, 1.0, gamma);
    } else {
      LinearCombine(
          {-a[m - 1], -b[m - 2]},
          {bases[m - 1], bases[m - 2]},
          1.0,
          gamma
      );
    }
#ifdef QLMPS_TIMING_MODE
    linear_combine_timer.PrintElapsed();
    normalize_timer.ClearAndRestart();
#endif
    QLTEN_Double norm_gamma = gamma->Normalize();
#ifdef QLMPS_TIMING_MODE
    normalize_timer.PrintElapsed();
    Timer trigssolver("lancz_triadiag_solver_total");
    trigssolver.Suspend();
#endif

    if (norm_gamma == 0.0) {
      expmv_res.iters = m;
      if (m == 1) { //initial state is just an eigenstate
        expmv_res.expmv = new TenT();
        QLTEN_Complex evolution_phase_factor{0.0, -step_length * a[0]};
        (*expmv_res.expmv) = (qlmps::complex_exp(evolution_phase_factor)) * (*bases[0]);
      } else {
#ifdef QLMPS_TIMING_MODE
        trigssolver.Restart();
#endif
        TridiagExpme1Solver(a, b, m, step_length, combination_factor);
#ifdef QLMPS_TIMING_MODE
        trigssolver.PrintElapsed();
        Timer final_linear_combine_timer("lancz_finial_linear_combine");
#endif
        expmv_res.expmv = new TenT(bases[0]->GetIndexes());
        LinearCombine(m, combination_factor, bases, QLTEN_Complex(0.0), expmv_res.expmv);
#ifdef QLMPS_TIMING_MODE
        final_linear_combine_timer.PrintElapsed();
#endif
      }
#ifdef QLMPS_TIMING_MODE
      Timer lancozs_post_precessing("lancz_post_processing");
#endif
      MasterBroadcastOrder(lanczos_finish, world);
      LanczosFree(combination_factor, bases, last_mat_mul_vec_res);
      delete[] last_combination_factor;
#ifdef QLMPS_TIMING_MODE
      lancozs_post_precessing.PrintElapsed();
      lancozs_after_send_ham.PrintElapsed();
#endif
      return expmv_res;
    }

    b[m - 1] = norm_gamma;
    bases[m] = gamma;
#ifdef QLMPS_TIMING_MODE
    mat_vec_timer.ClearAndRestart();
#endif
    MasterBroadcastOrder(lanczos_mat_vec_dynamic, world);
    last_mat_mul_vec_res = new TenT();
    a[m] = master_two_site_eff_ham_mul_state(
        rpeff_ham,
        bases[m],
        *last_mat_mul_vec_res,
        world
    );
#ifdef QLMPS_TIMING_MODE
    mat_vec_timer.PrintElapsed();
    trigssolver.Restart();
#endif

    TridiagExpme1Solver(a, b, m + 1, step_length, combination_factor);
#ifdef QLMPS_TIMING_MODE
    trigssolver.Suspend();
#endif
    double distance = Distance(last_combination_factor, combination_factor, m + 1);
    if (distance < params.error ||
        m == eff_ham_eff_dim ||
        m == params.max_iterations - 1
        ) {
#ifdef QLMPS_TIMING_MODE
      trigssolver.PrintElapsed();
      Timer final_linear_combine_timer("lancz_finial_linear_combine");
#endif
      expmv_res.iters = m + 1;
      expmv_res.expmv = new TenT(bases[0]->GetIndexes());
      LinearCombine(m + 1, combination_factor, bases, QLTEN_Complex(0.0), expmv_res.expmv);
#ifdef QLMPS_TIMING_MODE
      final_linear_combine_timer.PrintElapsed();
#endif
#ifdef QLMPS_TIMING_MODE
      Timer lancozs_post_precessing("lancz_post_processing");
#endif
      MasterBroadcastOrder(lanczos_finish, world);
      LanczosFree(combination_factor, bases, last_mat_mul_vec_res);
      delete[] last_combination_factor;
#ifdef QLMPS_TIMING_MODE
      lancozs_post_precessing.PrintElapsed();
      lancozs_after_send_ham.PrintElapsed();
#endif
      return expmv_res;
    }
    std::swap(last_combination_factor, combination_factor);
  }
}

}

#endif //QLMPS_ALGO_MPI_LANCZOS_EXPMV_SOLVER_MPI_H
