// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-03-25 13:13
*
* Description: QuantumLiquids/UltraDMRG project. Operations for finite MPS.
*/
/**
@file finite_mps_ops.h
@brief Operations for finite MPS.
*/
#ifndef QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_OPS_H
#define QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_OPS_H

#include "qlmps/one_dim_tn/mps/finite_mps/finite_mps.h"    // FiniteMPS
#include "qlten/qlten.h"

namespace qlmps {
using namespace qlten;

/**
Add two finite MPSs A and B.

@param mps_a The finite MPS A.
@param mps_b The finite MPS B.
@param mps_c The finite MPS which equals to A + B.

@note Two MPSs must have the same quantum number divergence and same local
      Hilbert spaces.
*/
template<typename TenElemT, typename QNT>
void FiniteMPSAdd(
    const FiniteMPS<TenElemT, QNT> &mps_a,
    const FiniteMPS<TenElemT, QNT> &mps_b,
    FiniteMPS<TenElemT, QNT> &mps_c
) {
  assert(Div(mps_a[0]) == Div(mps_b[0]));
  assert(mps_a.size() == mps_b.size());   // TODO: should check SiteVecs
  assert(mps_b.size() == mps_c.size());
  auto N = mps_a.size();
  for (size_t i = 0; i < N; ++i) {
    mps_c.alloc(i);
    if (i == 0) {
      Expand(mps_a(i), mps_b(i), {2}, mps_c(i));
    } else if (i == N - 1) {
      Expand(mps_a(i), mps_b(i), {0}, mps_c(i));
    } else {
      Expand(mps_a(i), mps_b(i), {0, 2}, mps_c(i));
    }
  }
}

/**
 * Inner product of two finite MPSs A and B.
 *
 * @param mps_a The finite MPS A (ket |A\rangle )
 * @param mps_b The finite MPS B (ket |B\rangle )
 * @return The finite MPS which equals to \langle B | A \rangle.
 */
template<typename TenElemT, typename QNT>
TenElemT FiniteMPSInnerProd(
    const FiniteMPS<TenElemT, QNT> &mps_a,
    const FiniteMPS<TenElemT, QNT> &mps_b
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  Index<QNT> left_trivial_index = mps_a[0].GetIndex(0);
  Index<QNT> left_trivial_index_inv = InverseIndex(left_trivial_index);
  Tensor tmp = Tensor({left_trivial_index_inv, left_trivial_index});
  tmp({0, 0}) = TenElemT(1.0);
  for (size_t i = 0; i < mps_a.size(); i++) {
    Tensor tmp2;
    Contract(&tmp, {0}, mps_a(i), {0}, &tmp2);
    Tensor bdag = Dag(mps_b[i]);
    tmp = Tensor();
    Contract(&tmp2, {0, 1}, &bdag, {0, 1}, &tmp);
  }
  return tmp({0, 0});
}
} /* qlmps */
#endif /* ifndef QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_OPS_H */
