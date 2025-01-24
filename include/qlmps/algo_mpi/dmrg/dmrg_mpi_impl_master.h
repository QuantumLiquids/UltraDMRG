// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-05-11
*
* Description: QuantumLiquids/MPS project. Two-site update finite size DMRG with MPI Parallelization, master side codes.
*/


#ifndef QLMPS_ALGO_MPI_DMRG_DMRG_MPI_IMPL_MASTER_H
#define QLMPS_ALGO_MPI_DMRG_DMRG_MPI_IMPL_MASTER_H

#include "qlmps/consts.h"                      // kMpsPath, kRuntimeTempPath
#include "qlmps/algorithm/vmps/two_site_update_finite_vmps_impl.h"   //MeasureEE

namespace qlmps {
using namespace qlten;

template<typename TenElemT, typename QNT>
class DMRGMPIMasterExecutor : public Executor {
  using Tensor = QLTensor<TenElemT, QNT>;
 public:
  DMRGMPIMasterExecutor(
      const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
      const FiniteVMPSSweepParams &sweep_params,
      const MPI_Comm &comm
  );

  ~DMRGMPIMasterExecutor() = default;
  void Execute() override;
  double GetEnergy() const {
    return e0_;
  }

  FiniteVMPSSweepParams sweep_params;
 private:
  void DMRGInit_();
  void InitBlockOps_();
  RightBlockOperatorGroup<Tensor> InitUpdateRightBlockOps_(
      const RightBlockOperatorGroup<QLTensor<TenElemT, QNT>> &rog,
      const QLTensor<TenElemT, QNT> &mps_ten,               // site i
      const SparMat<QLTensor<TenElemT, QNT>> &mat_repr_mpo   // site i
  );
  double DMRGSweep_();
  void DMRGPostProcess_();

  void LoadRelatedTensSweep_();
  void SetEffectiveHamiltonianTerms_();
  double TwoSiteUpdate_();
  void DumpRelatedTensSweep_();

  LanczosRes<Tensor> LanczosSolver_(Tensor *pinit_state);
  Tensor *DynamicHamiltonianMultiplyState_(Tensor &state);
  Tensor *StaticHamiltonianMultiplyState_(Tensor &state, QLTEN_Double &overlap);

  void GrowLeftBlockOps_();
  void GrowRightBlockOps_();

  void SendBlockSiteHamiltonianTermGroup_(
      const BlockSiteHamiltonianTermGroup<Tensor> &,
      size_t
  );
  void SendSiteBlockHamiltonianTermGroup_(
      const SiteBlockHamiltonianTermGroup<Tensor> &,
      size_t
  );

  size_t N_; //number of site
  FiniteMPS<TenElemT, QNT> mps_;
  const MatReprMPO<Tensor> mat_repr_mpo_;
  double e0_;  // energy;

  std::vector<LeftBlockOperatorGroup<Tensor>> lopg_vec_;
  std::vector<RightBlockOperatorGroup<Tensor>> ropg_vec_;

  size_t left_boundary_;
  size_t right_boundary_;
  char dir_;   // 'l', 'r'

  size_t l_site_;
  size_t r_site_;

  std::vector<Tensor> block_site_ops_;
  std::vector<Tensor> site_block_ops_;
  // H_eff = \sum_i block_site_ops_[i] direct product site_block_ops_[i]
  SuperBlockHamiltonianTerms<Tensor> hamiltonian_terms_;

  const MPI_Comm &comm_;
  int rank_;
  int mpi_size_;
  size_t worker_num_;
};

template<typename TenElemT, typename QNT>
DMRGMPIMasterExecutor<TenElemT, QNT>::DMRGMPIMasterExecutor(
    const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const MPI_Comm &comm
):
    sweep_params(sweep_params),
    N_(mat_repr_mpo.size()),
    mps_(SiteVec<TenElemT, QNT>(mat_repr_mpo[0](0, 0).GetIndexes())),
    mat_repr_mpo_(mat_repr_mpo),
    lopg_vec_(N_), ropg_vec_(N_),
    dir_('r'),
    comm_(comm) {
  MPI_Comm_size(comm, &mpi_size_);
  MPI_Comm_rank(comm, &rank_);
  worker_num_ = mpi_size_ - 1;
  // Initial for MPS.
  IndexVec<QNT> index_vec(N_);
  for (size_t site = 0; site < N_; site++) {
    index_vec[site] = mat_repr_mpo[site].data.front().GetIndexes()[1];
  }
  auto hilbert_space = SiteVec<TenElemT, QNT>(index_vec);
  mps_ = FiniteMPS(hilbert_space);

  SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT>
void DMRGMPIMasterExecutor<TenElemT, QNT>::Execute() {
  SetStatus(ExecutorStatus::EXEING);
  assert(mps_.size() == mat_repr_mpo_.size());
  MasterBroadcastOrder(program_start, rank_, comm_);
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
  MasterBroadcastOrder(program_final, rank_, comm_);
  SetStatus(ExecutorStatus::FINISH);
}

template<typename TenElemT, typename QNT>
double DMRGMPIMasterExecutor<TenElemT, QNT>::DMRGSweep_() {
  dir_ = 'r';
  //TODO : asynchronous IO
  for (size_t i = left_boundary_; i < right_boundary_ - 1; ++i) {
    l_site_ = i;
    r_site_ = i + 1;
    LoadRelatedTensSweep_();
    SetEffectiveHamiltonianTerms_();
    e0_ = TwoSiteUpdate_();
    DumpRelatedTensSweep_();
  }

  dir_ = 'l';
  for (size_t i = right_boundary_; i > left_boundary_ + 1; --i) {
    l_site_ = i - 1;
    r_site_ = i;
    LoadRelatedTensSweep_();
    SetEffectiveHamiltonianTerms_();
    e0_ = TwoSiteUpdate_();
    DumpRelatedTensSweep_();
  }
  return e0_;
}

///< Move center from left_boundary_ + 1 to 0
template<typename TenElemT, typename QNT>
void DMRGMPIMasterExecutor<TenElemT, QNT>::DMRGPostProcess_() {
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
double DMRGMPIMasterExecutor<TenElemT, QNT>::TwoSiteUpdate_() {
  Timer update_timer("two_site_dmrg_update");
  const std::vector<std::vector<size_t>> init_state_ctrct_axes = {{2}, {0}};
  const size_t svd_ldims = 2;
  const size_t l_block_len = l_site_, r_block_len = N_ - 1 - r_site_;

  auto init_state = new Tensor;
  Contract(&mps_[l_site_], &mps_[r_site_], init_state_ctrct_axes, init_state);
  //lanczos,
  Timer lancz_timer("two_site_dmrg_lancz");
  MasterBroadcastOrder(lanczos, rank_, comm_);
  HANDLE_MPI_ERROR(::MPI_Bcast(&l_site_, 1, MPI_UNSIGNED_LONG_LONG, rank_, comm_));
  auto lancz_res = LanczosSolver_(init_state);
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#ifdef QLMPS_TIMING_MODE
  lancz_timer.PrintElapsed();
#endif
  //svd,
#ifdef QLMPS_TIMING_MODE
  Timer svd_timer("two_site_dmrg_svd");
#endif
  Tensor u, vt;
  using DTenT = QLTensor<QLTEN_Double, QNT>;
  DTenT s;
  QLTEN_Double actual_trunc_err;
  size_t D;
  MasterBroadcastOrder(svd, rank_, comm_);
  MPISVDMaster(
      lancz_res.gs_vec,
      svd_ldims, Div(mps_[l_site_]),
      sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
      &u, &s, &vt, &actual_trunc_err, &D,
      comm_
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

  Tensor the_other_mps_ten;
  switch (dir_) {
    case 'r':mps_[l_site_] = std::move(u);
      Contract(&s, &vt, {{1}, {0}}, &the_other_mps_ten);
      mps_[r_site_] = std::move(the_other_mps_ten);
      break;
    case 'l':Contract(&u, &s, {{2}, {0}}, &the_other_mps_ten);
      mps_[l_site_] = std::move(the_other_mps_ten);
      mps_[r_site_] = std::move(vt);
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
      MasterBroadcastOrder(growing_left_env, rank_, comm_);
      GrowLeftBlockOps_();
    }
      break;
    case 'l': {
      MasterBroadcastOrder(growing_right_env, rank_, comm_);
      GrowRightBlockOps_();
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
            << " S = " << std::setw(10) << std::setprecision(7) << ee;
  std::cout << std::scientific << std::endl;
  return lancz_res.gs_eng;
}

template<typename TenElemT, typename QNT>
void DMRGMPIMasterExecutor<TenElemT, QNT>::SetEffectiveHamiltonianTerms_() {
  hamiltonian_terms_.clear();
  block_site_ops_.clear();
  site_block_ops_.clear();
  // in the sense of this function, we should set the block_site_ops_ and site_block_ops_ here
  // but, for the convenience of MPI parallel, we move this part into LanczosSolver_
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
void DMRGMPIMasterExecutor<TenElemT, QNT>::DumpRelatedTensSweep_() {
#ifdef QLMPS_TIMING_MODE
  Timer postprocessing_timer("two_site_dmrg_postprocessing");
#endif
  lopg_vec_[l_site_].clear();
  ropg_vec_[(N_ - 1) - r_site_].clear();
  switch (dir_) {
    case 'r':assert(!mps_[l_site_].IsDefault());
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
void DMRGMPIMasterExecutor<TenElemT, QNT>::LoadRelatedTensSweep_(
) {
#ifdef QLMPS_TIMING_MODE
  Timer preprocessing_timer("two_site_dmrg_preprocessing");
#endif
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
      } else {
        mps_.LoadTen(r_site_,
                     GenMPSTenName(sweep_params.mps_path, r_site_)
        );
        auto rblock_len = (N_ - 1) - r_site_;
        size_t ropg_size = mat_repr_mpo_[r_site_].cols;
        ropg_vec_[rblock_len] = RightBlockOperatorGroup<QLTensor<TenElemT, QNT>>(ropg_size);
        ReadAndRemoveOperatorGroup("r", rblock_len, ropg_vec_[rblock_len], sweep_params.temp_path);
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
      } else {
        mps_.LoadTen(l_site_,
                     GenMPSTenName(sweep_params.mps_path, l_site_)
        );
        auto lblock_len = l_site_;
        size_t lopg_size = mat_repr_mpo_[l_site_].rows;
        lopg_vec_[l_site_] = LeftBlockOperatorGroup<QLTensor<TenElemT, QNT>>(lopg_size);
        ReadAndRemoveOperatorGroup("l", lblock_len, lopg_vec_[l_site_], sweep_params.temp_path);
      }
      break;
    default:assert(false);
  }
#ifdef QLMPS_TIMING_MODE
  preprocessing_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT>
void DMRGMPIMasterExecutor<TenElemT, QNT>::DMRGInit_() {
  std::cout << "\n";
  std::cout << "=====> MPI Sweep Parameters For DMRG <=====" << "\n";
  std::cout << "MPS/MPO size: \t " << mps_.size() << "\n";
  std::cout << "Sweep times: \t " << sweep_params.sweeps << "\n";
  std::cout << "Bond dimension: \t " << sweep_params.Dmin << "/" << sweep_params.Dmax << "\n";
  std::cout << "Truncation error: \t " << sweep_params.trunc_err << "\n";
  std::cout << "Lanczos max iterations \t" << sweep_params.lancz_params.max_iterations << "\n";
  std::cout << "MPS path: \t" << sweep_params.mps_path << "\n";
  std::cout << "Temp path: \t" << sweep_params.temp_path << std::endl;

  std::cout << "=====> Technical Parameters <=====" << "\n";
  std::cout << "The number of processors(including master): \t" << mpi_size_ << "\n";
#ifndef  USE_GPU
  std::cout << "The number of threads per processor: \t" << hp_numeric::GetTensorManipulationThreads() << "\n";
#endif
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
    InitBlockOps_();
  } else {
    MasterBroadcastOrder(init_grow_env_finish, rank_, comm_);
    std::cout << "The block operators exist." << std::endl;
  }
  UpdateBoundaryBlockOpsMaster(mps_, mat_repr_mpo_,
                               sweep_params.mps_path, sweep_params.temp_path,
                               left_boundary_,
                               right_boundary_,
                               comm_
  );
}

template<typename TenElemT, typename QNT>
void DMRGMPIMasterExecutor<TenElemT, QNT>::InitBlockOps_(
) {
  using TenT = QLTensor<TenElemT, QNT>;
  auto N = mps_.size();
  const std::string &mps_path = sweep_params.mps_path;
  const std::string &temp_path = sweep_params.temp_path;

  // right operators
  RightBlockOperatorGroup<TenT> rog(1);
  mps_.LoadTen(N - 1, GenMPSTenName(mps_path, N - 1));
  auto trivial_index = mps_.back().GetIndexes()[2];
  auto trivial_index_inv = InverseIndex(trivial_index);
  auto id_scalar = TenT({trivial_index_inv, trivial_index});
  id_scalar({0, 0}) = 1;
  rog[0] = id_scalar;

  WriteOperatorGroup("r", 0, rog, temp_path);

  for (size_t i = 1; i <= N - (left_boundary_ + 2); ++i) {
    if (i > 1) { mps_.LoadTen(N - i, GenMPSTenName(mps_path, N - i)); }
    MasterBroadcastOrder(init_grow_env_grow, rank_, comm_);
    auto rog_next = InitUpdateRightBlockOps_(rog, mps_[N - i], mat_repr_mpo_[N - i]);
    rog = std::move(rog_next);
    WriteOperatorGroup("r", i, rog, temp_path);
    mps_.dealloc(N - i);
  }
  MasterBroadcastOrder(init_grow_env_finish, rank_, comm_);
}

template<typename TenElemT, typename QNT>
RightBlockOperatorGroup<QLTensor<TenElemT, QNT>> DMRGMPIMasterExecutor<TenElemT, QNT>::InitUpdateRightBlockOps_(
    const RightBlockOperatorGroup<QLTensor<TenElemT, QNT>> &rog,
    const QLTensor<TenElemT, QNT> &mps_ten,               // site i
    const SparMat<QLTensor<TenElemT, QNT>> &mat_repr_mpo   // site i
) {
  size_t res_op_num = mat_repr_mpo.rows;
  std::cout << "right block operator number : " << res_op_num << std::endl;
  size_t &task_num = res_op_num;
  HANDLE_MPI_ERROR(::MPI_Bcast(&task_num, 1, MPI_UNSIGNED_LONG_LONG, rank_, comm_));
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_mps_timer("grow_ops_broadcast_mps_send");
#endif
  const_cast<QLTensor<TenElemT, QNT> *>(&mps_ten)->MPI_Bcast(rank_, comm_);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_mps_timer.PrintElapsed();
#endif

  RightBlockOperatorGroup<Tensor> rog_next(res_op_num);

  if (task_num <= worker_num_) {
    for (size_t i = 0; i < task_num; i++) {
      SiteBlockHamiltonianTermGroup<Tensor> site_block_terms;
      for (size_t k = 0; k < mat_repr_mpo.cols; k++) {
        if (!mat_repr_mpo.IsNull(i, k)) {
          SiteBlockHamiltonianTerm<Tensor> site_block_term = {const_cast<Tensor *>(&mat_repr_mpo(i, k)),
                                                              const_cast<Tensor *>(&rog[k])};
          site_block_terms.emplace_back(site_block_term);
        }
      }
      SendSiteBlockHamiltonianTermGroup_(site_block_terms, i + 1);
    }
    for (size_t i = 0; i < task_num; i++) {
      rog_next[i].MPI_Recv(i + 1, i, comm_);
    }
  } else {
    for (size_t task_id = 0; task_id < worker_num_; task_id++) {
      const size_t worker = task_id + 1;
      hp_numeric::MPI_Send(task_id, worker, worker, comm_);
      SiteBlockHamiltonianTermGroup<Tensor> site_block_terms;
      for (size_t k = 0; k < mat_repr_mpo.cols; k++) {
        if (!mat_repr_mpo.IsNull(task_id, k)) {
          SiteBlockHamiltonianTerm<Tensor> site_block_term = {const_cast<Tensor *>(&mat_repr_mpo(task_id, k)),
                                                              const_cast<Tensor *>(&rog[k])};
          site_block_terms.emplace_back(site_block_term);
        }
      }
      SendSiteBlockHamiltonianTermGroup_(site_block_terms, worker);
    }
    for (size_t task_id = worker_num_; task_id < task_num; task_id++) {
      Tensor res;
      auto status = res.MPI_Recv(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_);
      size_t returned_task_id = status.MPI_TAG;
      size_t worker = status.MPI_SOURCE;
      rog_next[returned_task_id] = std::move(res);
      hp_numeric::MPI_Send(task_id, worker, worker, comm_);
      SiteBlockHamiltonianTermGroup<Tensor> site_block_terms;
      for (size_t k = 0; k < mat_repr_mpo.cols; k++) {
        if (!mat_repr_mpo.IsNull(task_id, k)) {
          SiteBlockHamiltonianTerm<Tensor> site_block_term = {const_cast<Tensor *>(&mat_repr_mpo(task_id, k)),
                                                              const_cast<Tensor *>(&rog[k])};
          site_block_terms.emplace_back(site_block_term);
        }
      }
      SendSiteBlockHamiltonianTermGroup_(site_block_terms, worker);
    }
    for (size_t i = 0; i < worker_num_; i++) {
      Tensor res;
      auto status = res.MPI_Recv(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_);
      rog_next[status.MPI_TAG] = std::move(res);
      hp_numeric::MPI_Send(FinalSignal(task_num), status.MPI_SOURCE, status.MPI_SOURCE, comm_);
    }
  }

  return rog_next;
}

template<typename TenElemT, typename QNT>
void DMRGMPIMasterExecutor<TenElemT, QNT>::SendBlockSiteHamiltonianTermGroup_(
    const BlockSiteHamiltonianTermGroup<Tensor> &block_site_terms,
    size_t id) {
  const size_t num_terms = block_site_terms.size();
  hp_numeric::MPI_Send(num_terms, id, 2 * id, comm_);
  for (size_t i = 0; i < num_terms; i++) {
    block_site_terms[i][0]->MPI_Send(id, i * id, comm_);
    block_site_terms[i][1]->MPI_Send(id, i * id, comm_);
  }
}

template<typename TenElemT, typename QNT>
void DMRGMPIMasterExecutor<TenElemT, QNT>::SendSiteBlockHamiltonianTermGroup_(
    const SiteBlockHamiltonianTermGroup<Tensor> &site_block_hamiltonian_term_group,
    size_t id) {
  const size_t num_terms = site_block_hamiltonian_term_group.size();
  hp_numeric::MPI_Send(num_terms, id, 2 * id, comm_);
  for (size_t i = 0; i < num_terms; i++) {

    Tensor &h_site = *site_block_hamiltonian_term_group[i][0];
    h_site.MPI_Send(id, i * id, comm_);
    Tensor &h_env = *site_block_hamiltonian_term_group[i][1];
    h_env.MPI_Send(id, i * id, comm_);
  }
}

template<typename TenElemT, typename QNT>
void UpdateBoundaryBlockOpsMaster(
    FiniteMPS<TenElemT, QNT> &mps,
    const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
    const std::string &mps_path,
    const std::string &temp_path,
    const size_t left_boundary,
    const size_t right_boundary,
    const MPI_Comm &comm
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

template<typename TenElemT, typename QNT>
void DMRGMPIMasterExecutor<TenElemT, QNT>::GrowLeftBlockOps_() {
  mps_[l_site_].MPI_Bcast(rank_, comm_);
  const size_t l_block_len = l_site_;
  const size_t num_ops = mat_repr_mpo_[l_site_].cols;
  lopg_vec_[l_block_len + 1] = std::vector<Tensor>(num_ops, Tensor());

  for (size_t i = 0; i < num_ops; i++) {
    size_t op_order;
    auto status = hp_numeric::MPI_Recv(op_order, MPI_ANY_SOURCE, MPI_ANY_TAG, comm_);
    auto worker = status.MPI_SOURCE;
    lopg_vec_[l_block_len + 1][op_order].MPI_Recv(worker, worker, comm_);
  }
}

template<typename TenElemT, typename QNT>
void DMRGMPIMasterExecutor<TenElemT, QNT>::GrowRightBlockOps_() {
  mps_[r_site_].MPI_Bcast(rank_, comm_);
  const size_t r_block_len = N_ - 1 - r_site_;
  const size_t num_ops = mat_repr_mpo_[r_site_].rows;
  ropg_vec_[r_block_len + 1] = std::vector<Tensor>(num_ops, Tensor());

  for (size_t i = 0; i < num_ops; i++) {
    size_t op_order;
    auto status = hp_numeric::MPI_Recv(op_order, MPI_ANY_SOURCE, MPI_ANY_TAG, comm_);
    auto worker = status.MPI_SOURCE;
    ropg_vec_[r_block_len + 1][op_order].MPI_Recv(worker, worker, comm_);
  }
}

}//qlmps

#include "lanczos_dmrg_solver_mpi_master.h"

#endif //QLMPS_ALGO_MPI_DMRG_DMRG_MPI_IMPL_MASTER_H
