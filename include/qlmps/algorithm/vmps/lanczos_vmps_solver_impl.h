// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-24 17:46
*
* Description: QuantumLiquids/UltraDMRG project. Implementation details for Lanczos solver.
*/

/**
@file lanczos_solver_impl.h
@brief A Lanczos solver for the effective Hamiltonian in MPS-MPO based algorithms.
*/

#ifndef QLMPS_ALGORITHM_LANCZOS_SOLVER_IMPL_H
#define QLMPS_ALGORITHM_LANCZOS_SOLVER_IMPL_H

#include <cstring>

#include "qlten/qlten.h"

#if defined(USE_OPENBLAS)
#include <cblas.h>                              // Use CBLAS header
#include <lapacke.h>
#else
#include "mkl.h"                                // Use MKL header
#endif

#include "qlmps/algorithm/lanczos_params.h"    // LanczosParams
#include "qlten/utility/timer.h"                // Timer
#include "qlmps/utilities.h"                   // Real

namespace qlmps {

using namespace qlten;

// Forward declarations.
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> *eff_ham_mul_two_site_state(
    const std::vector<QLTensor<TenElemT, QNT> *> &,
    QLTensor<TenElemT, QNT> *
);

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> *eff_ham_mul_single_site_state(
    const std::vector<QLTensor<TenElemT, QNT> *> &eff_ham,
    QLTensor<TenElemT, QNT> *state
);

void TridiagGsSolver(
    const std::vector<double> &, const std::vector<double> &, const size_t,
    double &, double *&, const char);

// Helpers.
template<typename TenT>
inline void InplaceContract(
    TenT *&lhs, const TenT *rhs,
    const std::vector<std::vector<size_t>> &axes) {
  auto res = new TenT;
  Contract(lhs, rhs, axes, res);
  delete lhs;
  lhs = res;
}

template<typename TenT, typename T>
inline void LanczosFree(
    T *&a,
    std::vector<TenT *> &b,
    TenT *&last_mat_mul_vec_res
) {
  if (a != nullptr) { delete[] a; }
  for (auto &ptr : b) { delete ptr; }
  delete last_mat_mul_vec_res;
}

//multithread version
template<typename TenT, typename T>
inline void LanczosFree(
    T *&a,
    std::vector<TenT *> &b,
    const size_t b_size,
    TenT *&last_mat_mul_vec_res
) {
  if (a != nullptr) { delete[] a; }
#ifndef  USE_GPU
  const int ompth = hp_numeric::tensor_manipulation_num_threads;
#else
  const int ompth = 4;
#endif
#pragma omp parallel for default(shared) num_threads(ompth) schedule(static)
  for (size_t i = 0; i < b_size; i++) {
    delete b[i];
  }

  delete last_mat_mul_vec_res;
}

// Lanczos solver.
template<typename TenT>
struct LanczosRes {
  size_t iters;
  double gs_eng;
  TenT *gs_vec;
};

/**
Obtain the lowest energy eigenvalue and corresponding eigenstate from the effective
Hamiltonian and a initial state using Lanczos algorithm.

@param rpeff_ham Effective Hamiltonian as a vector of pointer-to-tensors.
@param pinit_state Pointer to initial state for Lanczos iteration.
@param eff_ham_mul_state Function pointer to effective Hamiltonian multiply to state.
@param params Parameters for Lanczos solver.
*/
template<typename TenT>
LanczosRes<TenT> LanczosSolver(
    const std::vector<TenT *> &rpeff_ham,
    TenT *pinit_state,
    TenT *(*eff_ham_mul_state)(const std::vector<TenT *> &, TenT *),     //this is a pointer pointing to a function
    const LanczosParams &params
) {
  // Take care that init_state will be destroyed after call the solver
  size_t eff_ham_eff_dim = pinit_state->size();

  LanczosRes<TenT> lancz_res;

  std::vector<std::vector<size_t>> energy_measu_ctrct_axes;
  if (pinit_state->Rank() == 3) {            // For single site update algorithm
    energy_measu_ctrct_axes = {{0, 1, 2},
                               {0, 1, 2}};
  } else if (pinit_state->Rank() == 4) {    // For two site update algorithm
    energy_measu_ctrct_axes = {{0, 1, 2, 3},
                               {0, 1, 2, 3}};
  }

  std::vector<TenT *> bases(params.max_iterations, nullptr);
  std::vector<QLTEN_Double> a(params.max_iterations, 0.0);
  std::vector<QLTEN_Double> b(params.max_iterations, 0.0);
  std::vector<QLTEN_Double> N(params.max_iterations, 0.0);

  // Initialize Lanczos iteration.
  pinit_state->QuasiNormalize();
  bases[0] = pinit_state;

#ifdef QLMPS_TIMING_MODE
  Timer mat_vec_timer("lancz_mat_vec");
#endif

  auto last_mat_mul_vec_res = (*eff_ham_mul_state)(rpeff_ham, bases[0]);

#ifdef QLMPS_TIMING_MODE
  mat_vec_timer.PrintElapsed();
#endif

  TenT temp_scalar_ten;
  auto base_dag = Dag(*bases[0]);
  Contract(
      last_mat_mul_vec_res, &base_dag,
      energy_measu_ctrct_axes,
      &temp_scalar_ten
  );
  a[0] = Real(temp_scalar_ten());;
  N[0] = 0.0;
  size_t m = 0;
  QLTEN_Double energy0;
  energy0 = a[0];
  // Lanczos iterations.
  while (true) {
    m += 1;
    TenT *gamma = last_mat_mul_vec_res;
    if constexpr (TenT::IsFermionic()) {
      gamma->ActFermionPOps();
    }
    if (m == 1) {
      LinearCombine({-a[m - 1]}, {bases[m - 1]}, 1.0, gamma);
    } else {
      LinearCombine(
          {-a[m - 1], -std::sqrt(N[m - 1])},
          {bases[m - 1], bases[m - 2]},
          1.0,
          gamma
      );
    }
    auto norm_gamma = gamma->QuasiNormalize();
    QLTEN_Double eigval;
    QLTEN_Double *eigvec = nullptr;
    if (norm_gamma == 0.0) {
      if (m == 1) {
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = new TenT(*bases[0]);
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        return lancz_res;
      } else {
        TridiagGsSolver(a, b, m, eigval, eigvec, 'V');
        auto gs_vec = new TenT(bases[0]->GetIndexes());
        LinearCombine(m, eigvec, bases, 0.0, gs_vec);
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = gs_vec;
        LanczosFree(eigvec, bases, m, last_mat_mul_vec_res);
        return lancz_res;
      }
    }

    N[m] = norm_gamma * norm_gamma;
    b[m - 1] = norm_gamma;
    bases[m] = gamma;

#ifdef QLMPS_TIMING_MODE
    mat_vec_timer.ClearAndRestart();
#endif

    last_mat_mul_vec_res = (*eff_ham_mul_state)(rpeff_ham, bases[m]);

#ifdef QLMPS_TIMING_MODE
    mat_vec_timer.PrintElapsed();
#endif

    TenT temp_scalar_ten;
    auto base_dag = Dag(*bases[m]);
    Contract(
        last_mat_mul_vec_res,
        &base_dag,
        energy_measu_ctrct_axes,
        &temp_scalar_ten
    );
    a[m] = Real(temp_scalar_ten());
    TridiagGsSolver(a, b, m + 1, eigval, eigvec, 'N');
    auto energy0_new = eigval;
    if (
        ((energy0 - energy0_new) < params.error) ||
            (m == eff_ham_eff_dim) ||
            (m == params.max_iterations - 1)
        ) {
      TridiagGsSolver(a, b, m + 1, eigval, eigvec, 'V');
      energy0 = energy0_new;
      auto gs_vec = new TenT(bases[0]->GetIndexes());
      LinearCombine(m + 1, eigvec, bases, 0.0, gs_vec);
      lancz_res.iters = m + 1;
      lancz_res.gs_eng = energy0;
      lancz_res.gs_vec = gs_vec;
      LanczosFree(eigvec, bases, m + 1, last_mat_mul_vec_res);
      return lancz_res;
    } else {
      energy0 = energy0_new;
    }
  }
}

/*
 * |----0                       0-----
 * |          2         2            |
 * |          |         |            |
 * |----1 0-------3 0--------3  1-----
 * |          |        |             |
 * |          1       1 2            |
 * |          |        |             |
 * |----2 0-------------------3 2----|
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> *eff_ham_mul_two_site_state(
    const std::vector<QLTensor<TenElemT, QNT> *> &eff_ham,
    QLTensor<TenElemT, QNT> *state
) {
  using TenT = QLTensor<TenElemT, QNT>;
  auto res = new TenT;
  TenT temp_ten1, temp_ten2, temp_ten3;
  Contract(eff_ham[0], state, {{2},
                               {0}}, &temp_ten1);
  Contract<TenElemT, QNT, true, true>(temp_ten1, *eff_ham[1], 1, 0, 2, temp_ten2);
  Contract<TenElemT, QNT, true, true>(temp_ten2, *eff_ham[2], 4, 0, 2, temp_ten3);
  Contract<TenElemT, QNT, true, false>(temp_ten3, *eff_ham[3], 4, 1, 2, *res);
#pragma omp parallel sections num_threads(3)
  {
#pragma omp section
    {
      temp_ten1.GetBlkSparDataTen().Clear();
    }
#pragma omp section
    {
      temp_ten2.GetBlkSparDataTen().Clear();
    }
#pragma omp section
    {
      temp_ten3.GetBlkSparDataTen().Clear();
    }
  }
  return res;
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> *eff_ham_mul_single_site_state(
    const std::vector<QLTensor<TenElemT, QNT> *> &eff_ham,
    QLTensor<TenElemT, QNT> *state
) {
  using TenT = QLTensor<TenElemT, QNT>;
  auto res = new TenT;
  TenT temp_ten;
  Contract(eff_ham[0], state, {{2},
                               {0}}, res);
  Contract<TenElemT, QNT, true, true>(*res, *eff_ham[1], 1, 0, 2, temp_ten);
  Contract<TenElemT, QNT, true, false>(temp_ten, *eff_ham[2], 3, 1, 2, *res);
  return res;
}

inline void TridiagGsSolver(
    const std::vector<double> &a, const std::vector<double> &b, const size_t n,
    double &gs_eng, double *&gs_vec, const char jobz) {
  auto d = new double[n];
  std::memcpy(d, a.data(), n * sizeof(double));
  auto e = new double[n];
  std::memcpy(e, b.data(), (n - 1) * sizeof(double));
  long ldz;
  auto stev_err_msg = "?stev error.";
  auto stev_jobz_err_msg = "jobz must be  'N' or 'V', but ";
  switch (jobz) {
    case 'N':ldz = 1;
      break;
    case 'V':ldz = n;
      break;
    default:std::cout << stev_jobz_err_msg << jobz << std::endl;
      std::cout << stev_err_msg << std::endl;
      exit(1);
  }
  auto z = new double[ldz * n];
  auto info = LAPACKE_dstev(    // TODO: Try dstevd dstevx some day.
      LAPACK_ROW_MAJOR, jobz,
      n,
      d, e,
      z,
      n);     // TODO: Why can not use ldz???
  if (info != 0) {
    std::cout << stev_err_msg << info << ", n : " << n << std::endl;
    if (info == -4) {
      for (size_t i = 0; i < n; i++) {
        std::cout << d[i] << std::endl;
      }
    }
    exit(1);
  }
  switch (jobz) {
    case 'N':break;
    case 'V':gs_vec = new double[n];
      for (size_t i = 0; i < n; ++i) { gs_vec[i] = z[i * n]; }
      break;
    default:std::cout << stev_jobz_err_msg << jobz << std::endl;
      std::cout << stev_err_msg << std::endl;
      exit(1);
  }
  gs_eng = d[0];
  delete[] d;
  delete[] e;
  delete[] z;

}
} /* qlmps */
#endif //QLMPS_ALGORITHM_LANCZOS_SOLVER_IMPL_H