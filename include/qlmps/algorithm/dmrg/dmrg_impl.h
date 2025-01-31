// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-21
*
* Description: QuantumLiquids/UltraDMRG project. Density matrix renormalization group.
*/

#ifndef QLMPS_ALGORITHM_DMRG_DMRG_IMPL_H
#define QLMPS_ALGORITHM_DMRG_DMRG_IMPL_H

#include <string>
#include "qlmps/consts.h"                                            // kMpsPath, kRuntimeTempPath
#include "qlmps/algorithm/lanczos_params.h"                          // LanczParams
#include "qlmps/algorithm/vmps/two_site_update_finite_vmps_impl.h"   //MeasureEE
#include "qlmps/algorithm/dmrg/lanczos_dmrg_solver_impl.h"           //LanczosSolver
#include "qlmps/algorithm/dmrg/operator_io.h"                        //ReadOperatorGroup
#include "qlmps/memory_monitor.h"                                    //MemoryMonitor

namespace qlmps {
using namespace qlten;

//forward declaration
template<typename TenElemT, typename QNT>
LeftBlockOperatorGroup<QLTensor<TenElemT, QNT>> UpdateLeftBlockOps(
    const LeftBlockOperatorGroup<QLTensor<TenElemT, QNT>> &log,   // site i's env tensors
    const QLTensor<TenElemT, QNT> &mps,                       // site i
    const SparMat<QLTensor<TenElemT, QNT>> &mat_repr_mpo   // site i
);

template<typename TenElemT, typename QNT>
RightBlockOperatorGroup<QLTensor<TenElemT, QNT>> UpdateRightBlockOps(
    const RightBlockOperatorGroup<QLTensor<TenElemT, QNT>> &rog,   // site i's env tensors
    const QLTensor<TenElemT, QNT> &mps,                       // site i
    const SparMat<QLTensor<TenElemT, QNT>> &mat_repr_mpo   // site i
);

template<typename TenElemT, typename QNT>
class DMRGExecutor : public Executor {
  using Tensor = QLTensor<TenElemT, QNT>;
 public:
  DMRGExecutor(
      const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
      const FiniteVMPSSweepParams &sweep_params
  );

  ~DMRGExecutor() = default;
  void Execute() override;
  double GetEnergy() const {
    return e0_;
  }

  FiniteVMPSSweepParams sweep_params;
 private:
  void DMRGInit_();
  double DMRGSweep_();
  void DMRGPostProcess_();

  double LoadRelatedTensSweep_();
  void SetEffectiveHamiltonianTerms_();
  double TwoSiteUpdate_();
  void DumpRelatedTensSweep_();

  size_t N_; //number of site
  FiniteMPS<TenElemT, QNT> mps_;
  const MatReprMPO<Tensor> mat_repr_mpo_;
  double e0_;  // energy;

  std::vector<LeftBlockOperatorGroup<Tensor>> lopg_vec_;
  std::vector<RightBlockOperatorGroup<Tensor>> ropg_vec_;
  double op_mem_ = 0;

  size_t left_boundary_;
  size_t right_boundary_;
  char dir_;   // 'l', 'r'

  size_t l_site_;
  size_t r_site_;

  MemoryMonitor memory_monitor_;
  SuperBlockHamiltonianTerms<Tensor> hamiltonian_terms_;
};

template<typename TenElemT, typename QNT>
DMRGExecutor<TenElemT, QNT>::DMRGExecutor(
    const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
    const FiniteVMPSSweepParams &sweep_params
):
    sweep_params(sweep_params),
    N_(mat_repr_mpo.size()),
    mps_(SiteVec<TenElemT, QNT>(mat_repr_mpo[0](0, 0).GetIndexes())),
    mat_repr_mpo_(mat_repr_mpo),
    lopg_vec_(N_), ropg_vec_(N_),
    dir_('r') {
  // Initial for MPS.
  IndexVec<QNT> index_vec(N_);
  for (size_t site = 0; site < N_; site++) {
    index_vec[site] = mat_repr_mpo[site].data.front().GetIndexes()[1];
  }
  auto hilbert_space = SiteVec<TenElemT, QNT>(index_vec);
  mps_ = FiniteMPS(hilbert_space);

  SetStatus(ExecutorStatus::INITED);
}

/**
 * Function to perform finite-size DMRG (Density Matrix Renormalization Group).
 * This function aims to mimic the traditional DMRG workflow, i.e. renormalized operator terminology.
 *
 * ### Key Difference:
 * Unlike `TwoSiteFiniteVMPS`, this function takes the matrix representation
 * of the MPO as input, rather than the MPO itself.
 *
 * ### Notes:
 * - The input MPS (Matrix Product State) is assumed to be empty, with data stored on disk.
 * - The canonical center of the input MPS should be set near site 0.
 * - The canonical center of the output MPS will be enforced to move to site 0.
 */
template<typename TenElemT, typename QNT>
void DMRGExecutor<TenElemT, QNT>::Execute() {
  SetStatus(ExecutorStatus::EXEING);
  assert(mps_.size() == mat_repr_mpo_.size());
  DMRGInit_();

  std::cout << "\n";
  mps_.LoadTen(left_boundary_ + 1, GenMPSTenName(sweep_params.mps_path, left_boundary_ + 1));
  for (size_t sweep = 1; sweep <= sweep_params.sweeps; ++sweep) {
    std::cout << "sweep " << sweep << std::endl;
    Timer sweep_timer("sweep");
    e0_ = DMRGSweep_();
    sweep_timer.PrintElapsed();
    std::cout << "\n";
  }
  mps_.DumpTen(left_boundary_ + 1, GenMPSTenName(sweep_params.mps_path, left_boundary_ + 1), true);
  DMRGPostProcess_();
  SetStatus(ExecutorStatus::FINISH);
}

/**
 * Center will be moved to left_boundary_ + 1 after the sweep
 *
 * @return The estimated energy
 */
template<typename TenElemT, typename QNT>
double DMRGExecutor<TenElemT, QNT>::DMRGSweep_() {
  using TenT = QLTensor<TenElemT, QNT>;

  dir_ = 'r';
  for (size_t i = left_boundary_; i < right_boundary_ - 1; ++i) {
    l_site_ = i;
    r_site_ = i + 1;
    double mem_load = LoadRelatedTensSweep_();
    memory_monitor_.Checkpoint("Loaded Related Tens");
    SetEffectiveHamiltonianTerms_();
    e0_ = TwoSiteUpdate_();
    DumpRelatedTensSweep_();
    memory_monitor_.Checkpoint("Dumped Related Tens");
  }

  dir_ = 'l';
  for (size_t i = right_boundary_; i > left_boundary_ + 1; --i) {
    l_site_ = i - 1;
    r_site_ = i;
    double mem_load = LoadRelatedTensSweep_();
    memory_monitor_.Checkpoint("Loaded Related Tens");
    SetEffectiveHamiltonianTerms_();
    e0_ = TwoSiteUpdate_();
    DumpRelatedTensSweep_();
    memory_monitor_.Checkpoint("Dumped Related Tens");
  }
  return e0_;
}

template<typename TenT>
double EvaluateOpMem(const std::vector<TenT> &ops) {
  double total_mem = 0;
  for (auto &op : ops) {
    total_mem += op.GetRawDataMemUsage();
  }
  return total_mem;
}

template<typename TenElemT, typename QNT>
double DMRGExecutor<TenElemT, QNT>::TwoSiteUpdate_() {
  Timer update_timer("two_site_dmrg_update");
  const std::vector<std::vector<size_t>> init_state_ctrct_axes = {{2}, {0}};
  const size_t svd_ldims = 2;
  const size_t l_block_len = l_site_, r_block_len = N_ - 1 - r_site_;

  auto init_state = new Tensor;
  Contract(&mps_[l_site_], &mps_[r_site_], init_state_ctrct_axes, init_state);
  auto div_left = Div(mps_[l_site_]);
  delete mps_(l_site_);
  delete mps_(r_site_);
  memory_monitor_.Checkpoint("Before Lancz");
  Timer lancz_timer("two_site_dmrg_lancz");
  auto lancz_res = LanczosSolver(
      hamiltonian_terms_, init_state,
      sweep_params.lancz_params
  );
  hamiltonian_terms_.clear();// this clear doesn't effect on the tensor raw data.
#ifdef QLMPS_TIMING_MODE
  auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif
  const double state_mem = lancz_res.gs_vec->GetRawDataMemUsage();

  //svd,
#ifdef QLMPS_TIMING_MODE
  Timer svd_timer("two_site_dmrg_svd");
#endif
  Tensor *u = new Tensor();
  Tensor *vt = new Tensor();
  using DTenT = QLTensor<QLTEN_Double, QNT>;
  DTenT s;
  QLTEN_Double actual_trunc_err;
  size_t D;
  SVD(
      lancz_res.gs_vec,
      svd_ldims, div_left,
      sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
      u, &s, vt, &actual_trunc_err, &D
  );
  delete lancz_res.gs_vec;
  auto ee = MeasureEE(s, D);
#ifdef QLMPS_TIMING_MODE
  svd_timer.PrintElapsed();
#endif
  //update MPS local tensors
#ifdef QLMPS_TIMING_MODE
  Timer update_mps_ten_timer("two_site_dmrg_update_mps_ten");
#endif

  Tensor *the_other_mps_ten = new Tensor();
  switch (dir_) {
    case 'r':mps_(l_site_) = u;
      Contract(&s, vt, {{1}, {0}}, the_other_mps_ten);
      delete vt;
      mps_(r_site_) = the_other_mps_ten;
      break;
    case 'l':Contract(u, &s, {{2}, {0}}, the_other_mps_ten);
      delete u;
      mps_(l_site_) = the_other_mps_ten;
      mps_(r_site_) = vt;
      break;
    default:assert(false);
  }

#ifdef QLMPS_TIMING_MODE
  update_mps_ten_timer.PrintElapsed();
#endif
  // Update block operators
#ifdef QLMPS_TIMING_MODE
  Timer update_block_op_timer("two_site_dmrg_update_block_op");
#endif
  switch (dir_) {
    case 'r': {
      lopg_vec_[l_block_len + 1] = UpdateLeftBlockOps(lopg_vec_[l_block_len], mps_[l_site_], mat_repr_mpo_[l_site_]);
    }
      break;
    case 'l': {
      ropg_vec_[r_block_len + 1] = UpdateRightBlockOps(ropg_vec_[r_block_len], mps_[r_site_], mat_repr_mpo_[r_site_]);
    }
      break;
    default:assert(false);
  }
#ifdef QLMPS_TIMING_MODE
  update_block_op_timer.PrintElapsed();
#endif
  auto update_elapsed_time = update_timer.Elapsed();
  std::cout << "Site (" << std::setw(4) << l_site_ << "," << std::setw(4) << r_site_ << ")"
            << " E0 = " << std::setw(16) << std::setprecision(kLanczEnergyOutputPrecision) << std::fixed
            << lancz_res.gs_eng
            << " TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
            << " D = " << std::setw(5) << D
            << " Iter = " << std::setw(3) << lancz_res.iters
            << " LanczT = " << std::setw(8) << lancz_elapsed_time
            << " TotT = " << std::setw(8) << update_elapsed_time
            << " StateMem = " << std::setw(8) << state_mem << "GB"
            << " OpsMem = " << std::setw(8) << op_mem_ << "GB"
            << " S = " << std::setw(10) << std::setprecision(7) << ee;
  std::cout << std::scientific << std::endl;
  return lancz_res.gs_eng;
}

template<typename TenElemT, typename QNT>
void DMRGExecutor<TenElemT, QNT>::SetEffectiveHamiltonianTerms_() {
  assert(hamiltonian_terms_.empty());
  for (size_t j = 0; j < mat_repr_mpo_[l_site_].cols; j++) { // the middle index
    BlockSiteHamiltonianTermGroup<Tensor> block_site_terms; //left two blocks
    SiteBlockHamiltonianTermGroup<Tensor> site_block_terms; //right two blocks
    for (size_t i = 0; i < lopg_vec_[l_site_].size(); i++) {
      if (!mat_repr_mpo_[l_site_].IsNull(i, j)) {
        BlockSiteHamiltonianTerm<Tensor> block_site_term = {&lopg_vec_[l_site_][i],
                                                            const_cast<Tensor *>(&mat_repr_mpo_[l_site_](i, j))};
        block_site_terms.emplace_back(block_site_term);
      }
    }
    for (size_t k = 0; k < mat_repr_mpo_[r_site_].cols; k++) {
      if (!mat_repr_mpo_[r_site_].IsNull(j, k)) {
        SiteBlockHamiltonianTerm<Tensor> site_block_term = {const_cast<Tensor *>(&mat_repr_mpo_[r_site_](j, k)),
                                                            &ropg_vec_[(N_ - 1) - r_site_][k]};
        site_block_terms.emplace_back(site_block_term);
      }
    }
    hamiltonian_terms_.emplace_back(std::make_pair(block_site_terms, site_block_terms));
  }
}

template<typename TenElemT, typename QNT>
void DMRGExecutor<TenElemT, QNT>::DumpRelatedTensSweep_() {
#ifdef QLMPS_TIMING_MODE
  Timer postprocessing_timer("two_site_dmrg_postprocessing");
#endif
  lopg_vec_[l_site_].clear();
  ropg_vec_[(N_ - 1) - r_site_].clear();
  switch (dir_) {
    case 'r':
      mps_.DumpTen(
          l_site_,
          GenMPSTenName(sweep_params.mps_path, l_site_),
          true
      );
      WriteOperatorGroup("l", r_site_, lopg_vec_[r_site_], sweep_params.temp_path);
      break;
    case 'l':
      mps_.DumpTen(
          r_site_,
          GenMPSTenName(sweep_params.mps_path, r_site_),
          true
      );
      WriteOperatorGroup("r", N_ - 1 - l_site_, ropg_vec_[N_ - 1 - l_site_], sweep_params.temp_path);
      break;
    default:assert(false);
  }
#ifdef QLMPS_TIMING_MODE
  postprocessing_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT>
double DMRGExecutor<TenElemT, QNT>::LoadRelatedTensSweep_(
) {
#ifdef QLMPS_TIMING_MODE
  Timer preprocessing_timer("two_site_dmrg_preprocessing");
#endif
  op_mem_ = 0.0;
  switch (dir_) {
    case 'r':
      if (l_site_ == left_boundary_) {
        mps_.LoadTen(l_site_,
                     GenMPSTenName(sweep_params.mps_path, l_site_)
        );
        auto lblock_len = l_site_;
        size_t lopg_size = mat_repr_mpo_[l_site_].rows;
        lopg_vec_[lblock_len] = LeftBlockOperatorGroup<Tensor>(lopg_size);
        ReadOperatorGroup("l", lblock_len, lopg_vec_[lblock_len], sweep_params.temp_path);
        auto rblock_len = N_ - 1 - r_site_;
        size_t ropg_size = mat_repr_mpo_[r_site_].cols;
        ropg_vec_[rblock_len] = RightBlockOperatorGroup<QLTensor<TenElemT, QNT>>(ropg_size);
        ReadAndRemoveOperatorGroup("r", rblock_len, ropg_vec_[rblock_len], sweep_params.temp_path);

//        mem += mps_[l_site_].GetRawDataMemUsage();
        op_mem_ += EvaluateOpMem(lopg_vec_[lblock_len]);
        op_mem_ += EvaluateOpMem(ropg_vec_[rblock_len]);
      } else {
        mps_.LoadTen(r_site_,
                     GenMPSTenName(sweep_params.mps_path, r_site_)
        );
        auto rblock_len = (N_ - 1) - r_site_;
        size_t ropg_size = mat_repr_mpo_[r_site_].cols;
        ropg_vec_[rblock_len] = RightBlockOperatorGroup<QLTensor<TenElemT, QNT>>(ropg_size);
        ReadAndRemoveOperatorGroup("r", rblock_len, ropg_vec_[rblock_len], sweep_params.temp_path);

//        mem += mps_[r_site_].GetRawDataMemUsage();
        op_mem_ += EvaluateOpMem(ropg_vec_[rblock_len]);
      }
      break;
    case 'l':
      if (r_site_ == right_boundary_) {
        mps_.LoadTen(r_site_,
                     GenMPSTenName(sweep_params.mps_path, r_site_)
        );
        auto rblock_len = (N_ - 1) - r_site_;
        size_t ropg_size = mat_repr_mpo_[r_site_].cols;
        ropg_vec_[rblock_len] = RightBlockOperatorGroup<QLTensor<TenElemT, QNT>>(ropg_size);
        ReadOperatorGroup("r", rblock_len, ropg_vec_[rblock_len], sweep_params.temp_path);
        auto lblock_len = l_site_;
        size_t lopg_size = mat_repr_mpo_[l_site_].rows;
        lopg_vec_[l_site_] = LeftBlockOperatorGroup<QLTensor<TenElemT, QNT>>(lopg_size);
        ReadAndRemoveOperatorGroup("l", lblock_len, lopg_vec_[l_site_], sweep_params.temp_path);

//        mem += mps_[r_site_].GetRawDataMemUsage();
        op_mem_ += EvaluateOpMem(lopg_vec_[lblock_len]);
        op_mem_ += EvaluateOpMem(ropg_vec_[rblock_len]);
      } else {
        mps_.LoadTen(l_site_,
                     GenMPSTenName(sweep_params.mps_path, l_site_)
        );
        auto lblock_len = l_site_;
        size_t lopg_size = mat_repr_mpo_[l_site_].rows;
        lopg_vec_[l_site_] = LeftBlockOperatorGroup<QLTensor<TenElemT, QNT>>(lopg_size);
        ReadAndRemoveOperatorGroup("l", lblock_len, lopg_vec_[l_site_], sweep_params.temp_path);

//        mem += mps_[l_site_].GetRawDataMemUsage();
        op_mem_ += EvaluateOpMem(lopg_vec_[lblock_len]);
      }
      break;
    default:assert(false);
  }
#ifdef QLMPS_TIMING_MODE
  preprocessing_timer.PrintElapsed();
#endif
  return op_mem_;
}

///< Move center from left_boundary_ + 1 to 0
template<typename TenElemT, typename QNT>
void DMRGExecutor<TenElemT, QNT>::DMRGPostProcess_() {
  size_t center = left_boundary_ + 1;
  mps_.LoadTen(sweep_params.mps_path, center);
  for (size_t site = center; site > 0; site--) {
    mps_.LoadTen(sweep_params.mps_path, site - 1);
    mps_.RightCanonicalizeTen(site);
    mps_.DumpTen(sweep_params.mps_path, site);
  }
  mps_.DumpTen(sweep_params.mps_path, 0);
  std::cout << "Moved the center of MPS to 0." << std::endl;
}

template<typename TenElemT, typename QNT>
double FiniteDMRG(FiniteMPS<TenElemT, QNT> &mps,
                  const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
                  const FiniteVMPSSweepParams &sweep_params) {
  DMRGExecutor<TenElemT, QNT> dmrg_executor = DMRGExecutor(mat_repr_mpo, sweep_params);
  dmrg_executor.Execute();
  return dmrg_executor.GetEnergy();
}

}//qlmps

#endif //QLMPS_ALGORITHM_DMRG_DMRG_IMPL_H
