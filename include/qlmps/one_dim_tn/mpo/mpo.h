// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024/02/14
*
* Description: QuantumLiquids/UltraDMRG project. General matrix product operator (MPO) class.
*/

/**
@file mpo.h
@brief General matrix product operator (MPO) class.
*/
#ifndef QLMPS_ONE_DIM_TN_MPO_MPO_H
#define QLMPS_ONE_DIM_TN_MPO_MPO_H


#include "qlmps/one_dim_tn/framework/ten_vec.h"    // TenVec


namespace qlmps {
using namespace qlten;

template <typename LocalTenT>
using MPO = TenVec<LocalTenT>;

} /* qlmps */

#include "qlmps/one_dim_tn/mpo/finite_mpo/finite_mpo.h"
#include "qlmps/one_dim_tn/mpo/finite_mpo/finite_mpo_utility.h"

#endif /* ifndef QLMPS_ONE_DIM_TN_MPO_MPO_H */
