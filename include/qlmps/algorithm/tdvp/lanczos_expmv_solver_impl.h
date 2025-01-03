// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-10-10
*
* Description: QuantumLiquids/UltraDMRG project. Implementation details for Lanczos solver of `exp(i \delta A)v`
*/

/**
@file lanczos_solver_expmv_impl.h
@brief Implementation details for Lanczos solver of `exp(i \delta A)v`
*/

#ifndef QLMPS_ALGORITHM_LANCZOS_SOLVER_EXPMV_IMPL_H
#define QLMPS_ALGORITHM_LANCZOS_SOLVER_EXPMV_IMPL_H

#include "qlten/qlten.h"
#include "qlten/utility/timer.h"                           // Timer
#include "qlmps/consts.h"                                  // exp
#include "qlmps/algorithm/lanczos_params.h"                // LanczosParams, LanczosFree...
#include "qlmps/algorithm/vmps/lanczos_vmps_solver_impl.h" // InplaceContract

namespace qlmps {

using namespace qlten;

// Forward declarations.
template <typename TenT>
TenT *eff_ham_mul_two_site_state(const std::vector<TenT *> &, TenT *);
template <typename TenT>
TenT *eff_ham_mul_single_site_state(const std::vector<TenT *> &, TenT *);

inline void TridiagExpme1Solver(
    const std::vector<double> &a,
    const std::vector<double> &b,
    const size_t n,
    const double step_length,
    QLTEN_Complex *res
);

template <typename ElemType>
inline double Distance(
    ElemType *v1,
    const ElemType *v2,
    const size_t n
);

template <typename TenT>
struct ExpmvRes {
  size_t iters; //dimension of Krylov space
  TenT *expmv;
};

/**
 * Obtain real time evolution on the given initial state.
 * Hamiltonian are given by effective hamiltonian forms on single or two site.
 * and the function pointer should consist with effective hamiltonian.
 * Step length can be negative implying a backward evolution.
 * In a formula,
 *    $$ \exp(-i \delta H_{eff}) |\psi\rangle $$
 *
 * @tparam TenT    only can be QLTENSOR<QLTEN_Complex> now, because time is real
 * @param rpeff_ham
 * @param pinit_state
 * @param eff_ham_mul_state
 * @param step_length
 * @param params
 * @return
 */
template <typename TenT>
ExpmvRes<TenT> LanczosExpmvSolver(
    const std::vector<TenT *> &rpeff_ham,
    TenT *pinit_state,
    TenT *(* eff_ham_mul_state)(const std::vector<TenT *> &, TenT *), //eff_ham_mul_two_site_state or eff_ham_mul_single_site_state
    const double step_length,
    const LanczosParams &params
    ) {
  // Take care that init_state will be destroyed after call the solver
  const size_t eff_ham_eff_dim = pinit_state->size();

  ExpmvRes<TenT> expmv_res;
  std::vector<std::vector<size_t>> energy_measu_ctrct_axes;
  if (pinit_state->Rank() == 3) {            // single site update
    energy_measu_ctrct_axes = {{0, 1, 2}, {0, 1, 2}};
  } else if (pinit_state->Rank() == 4) {    // two site update
    energy_measu_ctrct_axes = {{0, 1, 2, 3}, {0, 1, 2, 3}};
  }

  std::vector<TenT *> bases(params.max_iterations);
  std::vector<QLTEN_Double> a(params.max_iterations, 0.0);
  std::vector<QLTEN_Double> b(params.max_iterations, 0.0);

  QLTEN_Double initial_norm = pinit_state->Normalize();
  bases[0] = pinit_state;
  auto last_mat_mul_vec_res = (*eff_ham_mul_state)(rpeff_ham, bases[0]);

  TenT temp_scalar_ten;
  auto base_dag = Dag(*bases[0]);
  Contract(
      last_mat_mul_vec_res, &base_dag,
      energy_measu_ctrct_axes,
      &temp_scalar_ten
  );
  a[0] = Real(temp_scalar_ten());
  size_t m = 0; //counting the iteration number

  QLTEN_Complex *combination_factor = new QLTEN_Complex[params.max_iterations];//combination of bases
  QLTEN_Complex *last_combination_factor = new QLTEN_Complex[params.max_iterations];
  while (true) {
    m += 1;
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
    QLTEN_Double norm_gamma = gamma->Normalize();

    if (norm_gamma == 0.0) {
      expmv_res.iters = m;
      if (m == 1) { //initial state is just an eigenstate
        expmv_res.expmv = new TenT();
        QLTEN_Complex evolution_phase_factor{0.0, -step_length * a[0]};
        (*expmv_res.expmv) = (initial_norm * qlmps::complex_exp(evolution_phase_factor)) * (*bases[0]);
      } else {
        TridiagExpme1Solver(a, b, m, step_length, combination_factor);
        expmv_res.expmv = new TenT(bases[0]->GetIndexes());
#ifndef USE_GPU
        hp_numeric::VectorScale(combination_factor, m, initial_norm);
#else
        for(size_t i = 0; i < m; i++){
          combination_factor[i] *= initial_norm;
        }
#endif
        LinearCombine(m, combination_factor, bases, QLTEN_Complex(0.0), expmv_res.expmv);
      }
      LanczosFree(combination_factor, bases, last_mat_mul_vec_res);
      delete[] last_combination_factor;
      return expmv_res;
    }

    b[m - 1] = norm_gamma;
    bases[m] = gamma;

    last_mat_mul_vec_res = (*eff_ham_mul_state)(rpeff_ham, bases[m]);

    TenT temp_scalar_ten;
    auto base_dag = Dag(*bases[m]);
    Contract(
        last_mat_mul_vec_res,
        &base_dag,
        energy_measu_ctrct_axes,
        &temp_scalar_ten
    );
    a[m] = Real(temp_scalar_ten());

    TridiagExpme1Solver(a, b, m + 1, step_length, combination_factor);
    double distance = Distance(last_combination_factor, combination_factor, m + 1);
    if (distance < params.error ||
        m == eff_ham_eff_dim ||
        m == params.max_iterations - 1
        ) {
      expmv_res.iters = m + 1;
      expmv_res.expmv = new TenT(bases[0]->GetIndexes());
#ifndef USE_GPU
      hp_numeric::VectorScale(combination_factor, m + 1, initial_norm);
#else
      for(size_t i = 0; i < m+1; i++){
        combination_factor[i] *= initial_norm;
      }
#endif
      LinearCombine(m + 1, combination_factor, bases, QLTEN_Complex(0.0), expmv_res.expmv);
      LanczosFree(combination_factor, bases, last_mat_mul_vec_res);
      delete[] last_combination_factor;
      return expmv_res;
    }
    std::swap(last_combination_factor, combination_factor);
  }
}

/**
 * $\exp( - 1i  \delta  A)  e_1$, where $e_1 = (1,0,....,0)^T$, and $A$ is tridiagonal real symmetric matrix.
 *  res is the exponential of matrix multiplied on vector $e_1$.
 *
 *  method: first we do an eigenvalue decomposition: $A = V D V^T$, where every column of $V$ is the eigenvector of $A$.
 *  $$\exp( - 1i  \delta  A)  e_1  = V \exp( - 1i  \delta  D) V^T e_1 $$
 *
 * @param a diagonal elements of matrix A, length n, all is real(double)
 * @param b second-diagonal elements of matrix A, length (n-1), all is real(double)
 * @param n
 * @param step_length  $\delta$
 * @param res result vector, length n, all is real(double).
 */
inline void TridiagExpme1Solver(
    const std::vector<double> &a,
    const std::vector<double> &b,
    const size_t n,
    const double step_length,
    QLTEN_Complex *res
) {
  double *d = (double *) malloc(n * sizeof(double));
  memcpy(d, a.data(), n * sizeof(double));
  double *e = (double *) malloc(n * sizeof(double));
  memcpy(e, b.data(), (n - 1) * sizeof(double));
  double *z = (double *) malloc(n * n * sizeof(double));
  const char *stev_err_msg = "?stev error.";
  auto info = LAPACKE_dstev(
      LAPACK_ROW_MAJOR,
      'V',
      n,
      d, e,
      z,
      n);
  if (info != 0) {
    std::cout << stev_err_msg << std::endl;
    exit(1);
  }
  //if performance is suffered by here,  aligned malloc here
  QLTEN_Complex *exp_of_eigenvals_mul_first_raw = (QLTEN_Complex *) malloc(n * sizeof(QLTEN_Complex));
  for (size_t i = 0; i < n; i++) {
    exp_of_eigenvals_mul_first_raw[i] = exp(QLTEN_Complex(0.0, -step_length * d[i])) * z[i];
  }

  for (size_t i = 0; i < n; i++) {
    QLTEN_Complex sum{0.0, 0.0};
    for (size_t j = 0; j < n; j++) {
      sum += z[i * n + j] * exp_of_eigenvals_mul_first_raw[j];
    }
    res[i] = sum;
  }
  free(d);
  free(e);
  free(z);
  free(exp_of_eigenvals_mul_first_raw);
}

/**
 * Euclidean distance of complex vector `v1` and `v2`. Namely,
 *  $$ ||v_1 - v_2||_2 $$
 * @note v1 is not a const type vector, it will be changed after call this function.
 *
 * @param v1
 * @param v2
 * @param n
 * @return
 */
template<typename ElemType>
inline double Distance(
    ElemType *v1,
    const ElemType *v2,
    const size_t n
) {
#ifndef USE_GPU
  qlten::hp_numeric::VectorAddTo(v2, n, v1, -1.0);
  return qlten::hp_numeric::Vector2Norm(v1, n);
#else
  double sum(0);
  for(size_t i = 0; i < n;i++){
    auto diff = v2[i] - v1[i];
    sum += qlten::norm(diff);
  }
  return qlten::sqrt(sum);
#endif
}
}//qlmps
#endif //QLMPS_ALGORITHM_LANCZOS_SOLVER_EXPMV_IMPL_H
