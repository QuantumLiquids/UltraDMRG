// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-22
*
* Description: QuantumLiquids/UltraDMRG project. Initialisation for two-site update finite size DMRG.
*/


#ifndef QLMPS_ALGORITHM_DMRG_DMRG_INIT_H
#define QLMPS_ALGORITHM_DMRG_DMRG_INIT_H

#ifdef Release
#define NDEBUG
#endif

#include "qlten/qlten.h"
#include "qlmps/one_dim_tn/mps_all.h"                          //FiniteMPS
#include "qlmps/algorithm/finite_vmps_sweep_params.h"  //FiniteVMPSSweepParams
#include "qlmps/algorithm/vmps/vmps_init.h"                    //CheckAndUpdateBoundaryMPSTensors
#include "qlmps/one_dim_tn/mpo/mat_repr_mpo.h"                     //MatReprMPO
#include "qlmps/algorithm/dmrg/operator_io.h"                  //Gen
#include "qlmps/algorithm/dmrg/dmrg_impl.h"
namespace qlmps {
using namespace qlten;

//forward declaration

template<typename TenElemT, typename QNT>
inline bool NeedGenerateBlockOps(
    const MatReprMPO<QLTensor<TenElemT, QNT>> &,
    const size_t,
    const size_t,
    const std::string &
);

template<typename TenElemT, typename QNT>
void DMRGExecutor<TenElemT, QNT>::PrintExeInfo_() {
  std::cout << "\n";
  std::cout << "=====> Sweep Parameters For DMRG <=====" << "\n";
  std::cout << "MPS/MPO size: \t " << mps_.size() << "\n";
  std::cout << "Sweep times: \t " << sweep_params.sweeps << "\n";
  std::cout << "Bond dimension: \t " << sweep_params.Dmin << "/" << sweep_params.Dmax << "\n";
  std::cout << "Truncation error: \t " << sweep_params.trunc_err << "\n";
  std::cout << "Lanczos max iterations \t" << sweep_params.lancz_params.max_iterations << "\n";
  std::cout << "MPS path: \t" << sweep_params.mps_path << "\n";
  std::cout << "Temp path: \t" << sweep_params.temp_path << std::endl;
#ifndef USE_GPU
  std::cout << "The number of threads: \t" << hp_numeric::GetTensorManipulationThreads() << "\n";
#endif
  std::cout << "Matrix Represented Operators memory usage : " << MemUsage(mat_repr_mpo_) << "GB" << "\n";
}

template<typename TenElemT, typename QNT>
void DMRGExecutor<TenElemT, QNT>::DMRGInit_() {
  std::cout << "=====> Checking and Updating Boundary Tensors =====>" << std::endl;

  auto [left_boundary, right_boundary] = CheckAndUpdateBoundaryMPSTensors(
      mps_,
      sweep_params.mps_path,
      sweep_params.Dmax
  );
  std::cout << "left boundary: \t" << left_boundary << std::endl;
  std::cout << "right boundary: \t" << right_boundary << std::endl;
  left_boundary_ = left_boundary;
  right_boundary_ = right_boundary;

  if (NeedGenerateBlockOps( // If the runtime temporary directory does not exit, create it
      mat_repr_mpo_,
      left_boundary_,
      right_boundary_,
      sweep_params.temp_path)
      ) {
    std::cout << "=====> Generating the block operators =====>" << std::endl;
    InitBlockOps(mps_, mat_repr_mpo_, sweep_params.mps_path, sweep_params.temp_path, left_boundary_);
  } else {
    std::cout << "The block operators exist." << std::endl;
  }
  UpdateBoundaryBlockOps(mps_, mat_repr_mpo_,
                         sweep_params.mps_path, sweep_params.temp_path,
                         left_boundary_,
                         right_boundary_
  );
  assert(mps_.empty());
}

///< if need to generate block operators
template<typename TenElemT, typename QNT>
inline bool NeedGenerateBlockOps(
    const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
    const size_t left_boundary, // the most left site need to update in DMRG
    const size_t right_boundary, // the most right site need to update in DMRG
    const std::string &temp_path
) {
  const size_t N = mat_repr_mpo.size();
  //right operators
  if (IsPathExist(temp_path)) {
    for (size_t site = right_boundary; site >= left_boundary + 1; site--) {
      size_t block_len = (N - 1) - site;
      for (size_t comp_num = 0; comp_num < mat_repr_mpo[site].cols; comp_num++) {
        std::string file = GenOpFileName("r", block_len, comp_num, temp_path);
        if (access(file.c_str(), 4) != 0) {
          std::cout << "Lost file" << file << "." << "\n";
          return true;
        }
      }
    }
    return false;
  } else {
    std::cout << "No temp directory " << temp_path << "\n";
    CreatPath(temp_path);
    return true;
  }
}

template<typename TenElemT, typename QNT>
RightBlockOperatorGroup<QLTensor<TenElemT, QNT>> UpdateRightBlockOps(
    const std::vector<QLTensor<TenElemT, QNT>> &site_block_ops,
    const QLTensor<TenElemT, QNT> &mps
) {
  auto mps_dag = Dag(mps);
  RightBlockOperatorGroup<QLTensor<TenElemT, QNT>> rog_next(site_block_ops.size());
  for (size_t i = 0; i < site_block_ops.size(); i++) {
    QLTensor<TenElemT, QNT> temp;
    Contract(&mps, &site_block_ops[i], {{1, 2}, {0, 1}}, &temp);
    Contract(&temp, &mps_dag, {{1, 2}, {1, 2}}, &rog_next[i]);
  }
  return rog_next;
}

template<typename TenElemT, typename QNT>
RightBlockOperatorGroup<QLTensor<TenElemT, QNT>> UpdateRightBlockOps(
    const RightBlockOperatorGroup<QLTensor<TenElemT, QNT>> &rog,   // site i's env tensors
    const QLTensor<TenElemT, QNT> &mps,                       // site i
    const SparMat<QLTensor<TenElemT, QNT>> &mat_repr_mpo   // site i
) {
  using TenT = QLTensor<TenElemT, QNT>;
  assert(rog.size() == mat_repr_mpo.cols);
  RightBlockOperatorGroup<QLTensor<TenElemT, QNT>> rog_next(mat_repr_mpo.rows);
  TenT mps_dag = Dag(mps);

  for (size_t row = 0; row < mat_repr_mpo.rows; row++) {
    TenT site_block_op;
    for (size_t col = 0; col < mat_repr_mpo.cols; col++) {
      if (!mat_repr_mpo.IsNull(row, col)) {
        TenT temp;
        Contract(&mat_repr_mpo(row, col), &rog[col], {{}, {}}, &temp);
        if (site_block_op == TenT()) {
          site_block_op = temp;
        } else {
          site_block_op += temp;
        }
      }
    }
    TenT temp;
    Contract(&mps, &site_block_op, {{1, 2}, {0, 2}}, &temp);
    Contract(&temp, &mps_dag, {{1, 2}, {1, 2}}, &rog_next[row]);
  }
  return rog_next;
}

template<typename TenElemT, typename QNT>
LeftBlockOperatorGroup<QLTensor<TenElemT, QNT>> UpdateLeftBlockOps(
    const std::vector<QLTensor<TenElemT, QNT>> &block_site_ops,
    const QLTensor<TenElemT, QNT> &mps) {
  auto mps_dag = Dag(mps);
  LeftBlockOperatorGroup<QLTensor<TenElemT, QNT>> log_next(block_site_ops.size());
  for (size_t i = 0; i < block_site_ops.size(); i++) {
    QLTensor<TenElemT, QNT> temp;
    Contract(&block_site_ops[i], &mps, {{2, 3}, {0, 1}}, &temp);
    Contract(&temp, &mps_dag, {{0, 1}, {0, 1}}, &log_next[i]);
  }
  return log_next;
}

template<typename TenElemT, typename QNT>
LeftBlockOperatorGroup<QLTensor<TenElemT, QNT>> UpdateLeftBlockOps(
    const LeftBlockOperatorGroup<QLTensor<TenElemT, QNT>> &log,   // site i's env tensors
    const QLTensor<TenElemT, QNT> &mps,                           // site i
    const SparMat<QLTensor<TenElemT, QNT>> &mat_repr_mpo          // site i
) {
  using TenT = QLTensor<TenElemT, QNT>;
  assert(log.size() == mat_repr_mpo.rows);
  LeftBlockOperatorGroup<QLTensor<TenElemT, QNT>> log_next(mat_repr_mpo.cols);
  TenT mps_dag = Dag(mps);
  for (size_t col = 0; col < mat_repr_mpo.cols; col++) {
    TenT block_site_op;
    for (size_t row = 0; row < mat_repr_mpo.rows; row++) {
      if (!mat_repr_mpo.IsNull(row, col)) {
        TenT temp;
        Contract(&log[row], &mat_repr_mpo(row, col), {{}, {}}, &temp);
        if (block_site_op == TenT()) {
          block_site_op = temp;
        } else {
          block_site_op += temp;
        }
      }
    }
    TenT temp;
    Contract(&block_site_op, &mps, {{0, 2}, {0, 1}}, &temp);
    Contract(&temp, &mps_dag, {{0, 1}, {0, 1}}, &log_next[col]);
  }
  return log_next;
}

template<typename TenElemT, typename QNT>
void UpdateBoundaryBlockOps(
    FiniteMPS<TenElemT, QNT> &mps,
    const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
    const std::string &mps_path,
    const std::string &temp_path,
    const size_t left_boundary,
    const size_t right_boundary
) {
  using TenT = QLTensor<TenElemT, QNT>;
  auto N = mps.size();

  // right operators
  RightBlockOperatorGroup<TenT> rog(1);
  mps.LoadTen(N - 1, GenMPSTenName(mps_path, N - 1));
  auto trivial_index = mps.back().GetIndexes()[2];
  auto trivial_index_inv = InverseIndex(trivial_index);
  auto id_scalar = TenT({trivial_index_inv, trivial_index});
  id_scalar({0, 0}) = 1;
  rog[0] = id_scalar;

  if (right_boundary < N - 1) {
    for (size_t i = 1; i <= N - right_boundary - 1; ++i) {
      if (i > 1) { mps.LoadTen(N - i, GenMPSTenName(mps_path, N - i)); }
      auto rog_next = UpdateRightBlockOps(rog, mps[N - i], mat_repr_mpo[N - i]);
      rog = std::move(rog_next);
      mps.dealloc(N - i);
    }
  }

  WriteOperatorGroup("r", N - 1 - right_boundary, rog, temp_path);

  mps.LoadTen(right_boundary, GenMPSTenName(mps_path, right_boundary));
  auto rog_next = UpdateRightBlockOps(rog, mps[right_boundary], mat_repr_mpo[right_boundary]);
  mps.dealloc(right_boundary);
  WriteOperatorGroup("r", N - right_boundary, rog_next, temp_path);

  // left operators
  LeftBlockOperatorGroup<TenT> log(1);
  mps.LoadTen(0, GenMPSTenName(mps_path, 0));
  trivial_index = mps.front().GetIndexes()[0];
  trivial_index_inv = InverseIndex(trivial_index);
  id_scalar = TenT({trivial_index_inv, trivial_index});
  id_scalar({0, 0}) = 1;
  log[0] = id_scalar;

  if (left_boundary > 0) {
    for (size_t i = 0; i <= left_boundary - 1; ++i) {
      if (i > 0) { mps.LoadTen(i, GenMPSTenName(mps_path, i)); }
      auto log_next = UpdateLeftBlockOps(log, mps[i], mat_repr_mpo[i]);
      log = std::move(log_next);
      mps.dealloc(i);
    }
  } else {
    mps.dealloc(0);
  }

  WriteOperatorGroup("l", left_boundary, log, temp_path);
}

/**
 * Generate the block operators for the preparation of the sweep
 * including left block of left_boundary site,
 * the right blocks of number 1 to (N-1) - (left_boundary + 1) (include)
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param mps
 * @param mat_repr_mpo
 * @param left_boundary the most left site which would be update in DMRG
 *
 * @note we have assumed left_boundary >=1, and right_boundary < N-1
 */
template<typename TenElemT, typename QNT>
void InitBlockOps(
    FiniteMPS<TenElemT, QNT> &mps,
    const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
    const std::string &mps_path,
    const std::string &temp_path,
    const size_t left_boundary
) {
  using TenT = QLTensor<TenElemT, QNT>;
  auto N = mps.size();

  // right operators
  RightBlockOperatorGroup<TenT> rog(1);
  mps.LoadTen(N - 1, GenMPSTenName(mps_path, N - 1));
  auto trivial_index = mps.back().GetIndexes()[2];
  auto trivial_index_inv = InverseIndex(trivial_index);
  auto id_scalar = TenT({trivial_index_inv, trivial_index});
  id_scalar({0, 0}) = 1;
  rog[0] = id_scalar;

  WriteOperatorGroup("r", 0, rog, temp_path);

  for (size_t i = 1; i <= N - (left_boundary + 2); ++i) {
    if (i > 1) { mps.LoadTen(N - i, GenMPSTenName(mps_path, N - i)); }
    auto rog_next = UpdateRightBlockOps(rog, mps[N - i], mat_repr_mpo[N - i]);
    rog = std::move(rog_next);
    WriteOperatorGroup("r", i, rog, temp_path);
    mps.dealloc(N - i);
  }
}

}//qlmps




#endif //QLMPS_ALGORITHM_DMRG_DMRG_INIT_H
