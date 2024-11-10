// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Haoxin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2022/05/10
*
* Description: Generate the exponential of MPO, which can be used in, e.g. XTRG algorithm.
*/


#ifndef QLMPS_ONE_DIM_TN_MPO_FINITE_MPO_EXP_FINITE_MPO_H
#define QLMPS_ONE_DIM_TN_MPO_FINITE_MPO_EXP_FINITE_MPO_H

#include "./finite_mpo.h"
#include "./finite_mpo_utility.h"

namespace qlmps {
using namespace qlten;
//forward declaration
inline std::string GenHnMpoPath(
    const std::string &mpo_path_prefix,
    const std::size_t n, //the power
    const double tau
);
/* FYI, lanczos figure
 * |----0                       0-----
 * |          2         2            |
 * |          |         |            |
 * |----1 0-------3 0--------3  1-----
 * |          |        |             |
 * |          1       1 2            |
 * |          |        |             |
 * |----2 0-------------------3 2----|
 */
/**
 * Find the MPO representation of density matrix rho
 * with temperature 1/tau by Taylor expansion.
 * The output density_matrix is normalized in the sense of 2-norm.
 *
 * @param hamiltonian
 * @param params
 * @return (norm, density matrix) = exp(-tau * H)
 */
template<typename TenElemT, typename QNT>
FiniteMPO<TenElemT, QNT> ExpFiniteMPO(
    const FiniteMPO<TenElemT, QNT> &hamiltonian,
    const double tau,
    const size_t max_taylor_order,
    const QLTEN_Double tol_taylor_expansion,
    const double trunc_err, const size_t Dmin, const size_t Dmax,
    const size_t sweep_max,
    const double variation_converge_tolerance,
    const std::string mpo_path_prefix,
    const std::string temp_path = kRuntimeTempPath
) {
  using MPOT = FiniteMPO<TenElemT, QNT>;
  std::cout << "Constructing MPO representation of the density matrix with tau = " << tau
            << "using Taylor expansion" << std::endl;
  const size_t N = hamiltonian.size();
  Timer taylor_expansion_timer("taylor_expansion");

  const size_t bond_dimension_of_H = hamiltonian.GetMaxBondDimension();

  std::cout << "Bond dimension of hamiltonian : "
            << bond_dimension_of_H << std::endl;
  MPOT m_tau_h(hamiltonian); // minus tau times hamiltonian
  m_tau_h.Scale(-tau);
  FiniteMPO<TenElemT, QNT> density_matrix = GenerateIndentiyMPO(hamiltonian);
  TenElemT t0 = density_matrix.Trace();
  std::cout << "power n = 0, Identity, Trace(0) : "
            << std::setprecision(7) << std::scientific << t0 << std::endl;
  MPOT taylor_term(m_tau_h);  //temp mpo's
// taylor_term =  (-tau * hamiltonian) ^ i / ( i! )
// temp_multiply_mpo -> taylor_term

  if (!IsPathExist(temp_path)) {
    CreatPath(temp_path);
  }
  for (size_t n = 1; n < max_taylor_order; n++) {
    std::cout << "power n = " << n << std::endl;
    Timer power_n_timer("power_n");
    MPOT tmp_mpo(N);
    std::string mpo_Hn_path = GenHnMpoPath(mpo_path_prefix, n, tau);
    if (n > 1) {
      if (std::pow(bond_dimension_of_H, n) <= Dmax) {
        MpoProduct(taylor_term, m_tau_h, tmp_mpo);
      } else { //need compress
        tmp_mpo.Load(mpo_Hn_path);

        MpoProduct(taylor_term, m_tau_h, tmp_mpo,
                   Dmin, Dmax, trunc_err,
                   sweep_max, variation_converge_tolerance,
                   temp_path);
      }
      taylor_term = std::move(tmp_mpo);
      taylor_term *= 1.0 / n;
    }
    density_matrix += taylor_term;
    taylor_term.Dump(mpo_Hn_path);

    TenElemT t = taylor_term.Trace();
    size_t D_Hn = taylor_term.GetMaxBondDimension();
    if (density_matrix.GetMaxBondDimension() > Dmax) {
      density_matrix.Truncate(1e-16, Dmax, Dmax);
    }
    double power_n_elapsed_time = power_n_timer.Elapsed();
    std::cout << "power n = " << n;
    std::cout << " D(H^" << n << ") = " << std::setw(5) << D_Hn
              << " Time = " << std::setw(8) << power_n_elapsed_time
              << " Trace(" << n << ") = " << std::setprecision(7) << std::scientific <<
              t;
    std::cout << std::scientific << std::endl;
    if (n % 2 == 0) {
      assert(t > 0.0);
      if (std::abs(t / t0) < tol_taylor_expansion) {
        break;
      }
    }
  }
  density_matrix.Centralize(0);
  double norm2 = density_matrix[0].Normalize();
  std::cout << "The series expansion was finished. The bond dimension of rho(tau) = "
            << density_matrix.GetMaxBondDimension()
            << std::endl;
  taylor_expansion_timer.PrintElapsed();
  return std::make_pair(norm2, density_matrix);
}

inline std::string GenHnMpoPath(
    const std::string &mpo_path_prefix,
    const std::size_t n, //the power
    const double tau
) {
  return mpo_path_prefix + "_H" + std::to_string(n) + "_tau" + std::to_string(tau);
}

} //qlmps


#endif //QLMPS_ONE_DIM_TN_MPO_FINITE_MPO_EXP_FINITE_MPO_H



