// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-21
*
* Description: QuantumLiquids/UltraDMRG project. Matrix Represented matrix product operator (MPO).
*/


#ifndef QLMPS_ONE_DIM_TN_MAT_REPR_MPO_H
#define QLMPS_ONE_DIM_TN_MAT_REPR_MPO_H

#include "qlten/qlten.h"                                        //QLTensor
#include "qlmps/one_dim_tn/mpo/mpogen/symb_alg/sparse_mat.h"   //SparMat

namespace qlmps {
using namespace qlten;

template <typename TenT>
using MatReprMPO = std::vector<SparMat<TenT>>;
}/* qlmps */




#endif //QLMPS_ONE_DIM_TN_MAT_REPR_MPO_H
