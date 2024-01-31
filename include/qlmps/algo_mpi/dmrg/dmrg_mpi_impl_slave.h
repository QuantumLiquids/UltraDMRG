// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-05-11
*
* Description: GraceQ/mps2 project. Two-site update finite size DMRG with MPI Parallelization, slave side codes.
*/

#ifndef QLMPS_ALGO_MPI_DMRG_DMRG_MPI_IMPL_SLAVE_H
#define QLMPS_ALGO_MPI_DMRG_DMRG_MPI_IMPL_SLAVE_H

namespace qlmps {
using namespace qlten;

template<typename TenElemT, typename QNT>
class DMRGMPISlaveExecutor : public Executor {
  using Tensor = QLTensor<TenElemT, QNT>;
 public:
  DMRGMPISlaveExecutor(const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
                       mpi::communicator &world
  );
  ~DMRGMPISlaveExecutor() = default;
  void Execute() override;

  size_t GetId() {
    return id_;
  }

 private:

  void DMRGInit_();
  void UpdateRightBlockOpsSlave_();

  void SlaveLanczosSolver_();
  void WorkForDynamicHamiltonianMultiplyState_();
  void WorkForStaticHamiltonianMultiplyState_();

  void WorkForGrowLeftBlockOps_();
  void WorkForGrowRightBlockOps_();

  mpi::status RecvBlockSiteHamiltonianTermGroup_();
  mpi::status RecvSiteBlockHamiltonianTermGroup_();

  size_t N_; //number of site
  const MatReprMPO<Tensor> mat_repr_mpo_;

  char dir_;
  size_t l_site_;
  size_t r_site_;

  std::vector<Tensor> block_site_ops_; //the set of operators which act on block and site hilbert space
  std::vector<Tensor> site_block_ops_;
  std::vector<size_t> ops_num_table_; //has the same length with block_site_ops_ and site_block_ops_

  BlockSiteHamiltonianTermGroup<Tensor> block_site_hamiltonian_term_group_; // a temp datum
  SiteBlockHamiltonianTermGroup<Tensor> site_block_hamiltonian_term_group_; // a temp datum

  const size_t id_;
  mpi::communicator &world_;
};

template<typename TenElemT, typename QNT>
DMRGMPISlaveExecutor<TenElemT, QNT>::DMRGMPISlaveExecutor(
    const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
    mpi::communicator &world
):N_(mat_repr_mpo.size()),
  mat_repr_mpo_(mat_repr_mpo), dir_('r'),
  id_(world.rank()),
  world_(world) {
  SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT>
void DMRGMPISlaveExecutor<TenElemT, QNT>::Execute() {
  SetStatus(ExecutorStatus::EXEING);
  auto order = SlaveGetBroadcastOrder(world_);
  if (order == program_start) {
    world_.send(kMasterRank, 2 * id_, id_);
  } else {
    std::cout << "unexpected " << std::endl;
    exit(1);
  }

  DMRGInit_();

  while (order != program_final) {
    order = SlaveGetBroadcastOrder(world_);
    switch (order) {
      case lanczos: {
        broadcast(world_, l_site_, kMasterRank);
        r_site_ = l_site_ + 1;
        SlaveLanczosSolver_();
      }
        break;
      case svd: {
        MPISVDSlave<TenElemT>(world_);
      }
        break;
      case growing_left_env: {
        WorkForGrowLeftBlockOps_();
      }
        break;
      case growing_right_env: {
        WorkForGrowRightBlockOps_();
      }
        break;
      case program_final:std::cout << "Node" << id_ << " will stop." << std::endl;
        break;
      default:std::cout << "Node " << id_ << " cannot understand the order " << order << std::endl;
        break;
    }
  }

  SetStatus(ExecutorStatus::FINISH);
}

template<typename TenElemT, typename QNT>
void DMRGMPISlaveExecutor<TenElemT, QNT>::DMRGInit_() {
  auto order = SlaveGetBroadcastOrder(world_);
  while (order != init_grow_env_finish) {
    assert(order == init_grow_env_grow);
    UpdateRightBlockOpsSlave_();
    order = SlaveGetBroadcastOrder(world_);
  }
}

template<typename TenElemT, typename QNT>
void DMRGMPISlaveExecutor<TenElemT, QNT>::UpdateRightBlockOpsSlave_() {
  size_t task_num;
  broadcast(world_, task_num, kMasterRank);
  Tensor mps;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_mps_timer("grow_ops_broadcast_mps_recv");
#endif
  RecvBroadCastQLTensor(world_, mps, kMasterRank);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_mps_timer.PrintElapsed();
#endif
  Tensor mps_dag = Dag(mps);

  if (task_num <= world_.size() - 1) {
    if (id_ <= task_num) {
      RecvSiteBlockHamiltonianTermGroup_();
      const size_t terms_num = site_block_hamiltonian_term_group_.size();
      auto psite_block_ops_res_s = std::vector<Tensor *>(terms_num);
      for (size_t j = 0; j < terms_num; j++) {
        psite_block_ops_res_s[j] = new Tensor();
        Contract(site_block_hamiltonian_term_group_[j][0],
                 site_block_hamiltonian_term_group_[j][1],
                 {{}, {}},
                 psite_block_ops_res_s[j]);
      }
      auto coefs = std::vector<TenElemT>(terms_num, TenElemT(1.0));
      auto site_block_op = Tensor();
      LinearCombine(coefs, psite_block_ops_res_s, TenElemT(0.0), &site_block_op);
      for (size_t j = 0; j < terms_num; j++) {
        delete psite_block_ops_res_s[j];
      }
      site_block_op.Transpose({0, 2, 1, 3});
      Tensor temp, res;
      Contract(&mps, &site_block_op, {{1, 2}, {0, 1}}, &temp);
      Contract(&temp, &mps_dag, {{1, 2}, {1, 2}}, &res);
      send_qlten(world_, kMasterRank, id_ - 1, res);
      //delete site_block_hamiltonian_term_group_ data
      for (size_t i = 0; i < terms_num; i++) {
        delete site_block_hamiltonian_term_group_[i][0];
        delete site_block_hamiltonian_term_group_[i][1];
      }
    }
  } else {//task_num > slave number
    size_t task_id;
    world_.recv(kMasterRank, id_, task_id);
    while (task_id < task_num) {
      RecvSiteBlockHamiltonianTermGroup_();
      const size_t terms_num = site_block_hamiltonian_term_group_.size();
      auto psite_block_ops_res_s = std::vector<Tensor *>(terms_num);
      for (size_t j = 0; j < terms_num; j++) {
        psite_block_ops_res_s[j] = new Tensor();
        Contract(site_block_hamiltonian_term_group_[j][0],
                 site_block_hamiltonian_term_group_[j][1],
                 {{}, {}},
                 psite_block_ops_res_s[j]);
      }
      auto coefs = std::vector<TenElemT>(terms_num, TenElemT(1.0));
      auto site_block_op = Tensor();
      LinearCombine(coefs, psite_block_ops_res_s, TenElemT(0.0), &site_block_op);
      for (size_t j = 0; j < terms_num; j++) {
        delete psite_block_ops_res_s[j];
      }
      site_block_op.Transpose({0, 2, 1, 3});
      Tensor temp, res;
      Contract(&mps, &site_block_op, {{1, 2}, {0, 1}}, &temp);
      Contract(&temp, &mps_dag, {{1, 2}, {1, 2}}, &res);
      send_qlten(world_, kMasterRank, task_id, res);
      //delete site_block_hamiltonian_term_group_ data
      for (size_t i = 0; i < terms_num; i++) {
        delete site_block_hamiltonian_term_group_[i][0];
        delete site_block_hamiltonian_term_group_[i][1];
      }
      world_.recv(kMasterRank, id_, task_id);
    }

  }
}

template<typename TenElemT, typename QNT>
mpi::status DMRGMPISlaveExecutor<TenElemT, QNT>::RecvBlockSiteHamiltonianTermGroup_() {
  size_t num_terms;
  world_.recv(kMasterRank, 2 * id_, num_terms);
  block_site_hamiltonian_term_group_.resize(num_terms);
  mpi::status recv_status;
  for (size_t i = 0; i < num_terms; i++) {
    block_site_hamiltonian_term_group_[i][0] = new Tensor();
    recv_qlten(world_, kMasterRank, i * id_, *block_site_hamiltonian_term_group_[i][0]);
    block_site_hamiltonian_term_group_[i][1] = new Tensor();
    recv_status = recv_qlten(world_, kMasterRank, i * id_, *block_site_hamiltonian_term_group_[i][1]);
  }
  return recv_status;
}

template<typename TenElemT, typename QNT>
mpi::status DMRGMPISlaveExecutor<TenElemT, QNT>::RecvSiteBlockHamiltonianTermGroup_() {
  size_t num_terms;
  mpi::status recv_status = world_.recv(kMasterRank, 2 * id_, num_terms);
  site_block_hamiltonian_term_group_.resize(num_terms);

  for (size_t i = 0; i < num_terms; i++) {
    site_block_hamiltonian_term_group_[i][0] = new Tensor();
    Tensor &h_site = *site_block_hamiltonian_term_group_[i][0];
    recv_qlten(world_, kMasterRank, i * id_, h_site);

    site_block_hamiltonian_term_group_[i][1] = new Tensor();
    Tensor &h_env = *site_block_hamiltonian_term_group_[i][1];
    recv_status = recv_qlten(world_, kMasterRank, i * id_, h_env);
  }
  return recv_status;
}

template<typename TenElemT, typename QNT>
void DMRGMPISlaveExecutor<TenElemT, QNT>::SlaveLanczosSolver_() {
  auto order = SlaveGetBroadcastOrder(world_);
  assert(order == lanczos_mat_vec_dynamic);
  WorkForDynamicHamiltonianMultiplyState_();
  order = SlaveGetBroadcastOrder(world_);
  while (order != lanczos_finish) {
    assert(order == lanczos_mat_vec_static);
    WorkForStaticHamiltonianMultiplyState_();
    order = SlaveGetBroadcastOrder(world_);
  }
}

template<typename TenElemT, typename QNT>
void DMRGMPISlaveExecutor<TenElemT, QNT>::WorkForDynamicHamiltonianMultiplyState_() {
#ifdef QLMPS_MPI_TIMING_MODE
  Timer slave_hamil_multiply_state_timer("node-" + std::to_string(id_) + "_hamiltonian_multiply_state_total_time");
  Timer slave_hamil_multiply_state_computation_timer
      ("node-" + std::to_string(id_) + "_hamiltonian_multiply_state_computation");
  slave_hamil_multiply_state_computation_timer.Suspend();
  Timer slave_hamil_multiply_state_delete_timer
      ("node-" + std::to_string(id_) + "_hamiltonian_multiply_state_delete");
  slave_hamil_multiply_state_delete_timer.Suspend();
  Timer slave_hamil_multiply_state_contract_timer
      ("node-" + std::to_string(id_) + "_hamiltonian_multiply_state_contraction");
  slave_hamil_multiply_state_contract_timer.Suspend();
  Timer slave_hamil_multiply_state_sum_timer("node-" + std::to_string(id_) + "_hamiltonian_multiply_state_summation");
  slave_hamil_multiply_state_sum_timer.Suspend();
  Timer slave_hamil_multiply_state_transpose_timer("node-" + std::to_string(id_) + "_hamiltonian_multiply_state_transpose");
  slave_hamil_multiply_state_transpose_timer.Suspend();
#endif

  size_t total_num_terms;
  mpi::broadcast(world_, total_num_terms, kMasterRank);
  Tensor state;
  RecvBroadCastQLTensor(world_, state, kMasterRank);
  block_site_ops_.clear();
  block_site_ops_.reserve(total_num_terms);
  site_block_ops_.clear();
  site_block_ops_.reserve(total_num_terms);
  ops_num_table_.clear();
  ops_num_table_.reserve(total_num_terms);

  Tensor multiplication_res;
  send_qlten(world_, kMasterRank, total_num_terms + 10086, multiplication_res);
  //tag > total_num_terms means invalid data
  bool terminated = false;

  size_t task_id;
  do {
    world_.recv(kMasterRank, id_, task_id);
    if (task_id > total_num_terms) {
      terminated = true;
    } else {
      RecvBlockSiteHamiltonianTermGroup_();
      RecvSiteBlockHamiltonianTermGroup_();
      //block site
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_computation_timer.Restart();
      slave_hamil_multiply_state_contract_timer.Restart();
#endif
      const size_t block_site_terms = block_site_hamiltonian_term_group_.size();
      auto pblock_site_ops_res_s = std::vector<Tensor *>(block_site_terms);

      for (size_t j = 0; j < block_site_terms; j++) {
        pblock_site_ops_res_s[j] = new Tensor();
        Contract(block_site_hamiltonian_term_group_[j][0],
                 block_site_hamiltonian_term_group_[j][1],
                 {{}, {}},
                 pblock_site_ops_res_s[j]);
      }
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_contract_timer.Suspend();
      slave_hamil_multiply_state_sum_timer.Restart();
#endif
      std::vector<TenElemT> coefs = std::vector<TenElemT>(block_site_terms, TenElemT(1.0));
      Tensor sum_op;

      LinearCombine(coefs, pblock_site_ops_res_s, TenElemT(0.0), &sum_op);
      for (size_t j = 0; j < block_site_terms; j++) {
        delete pblock_site_ops_res_s[j];
      }
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_sum_timer.Suspend();
      slave_hamil_multiply_state_transpose_timer.Restart();
#endif
      sum_op.Transpose({1, 3, 0, 2});
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_transpose_timer.Suspend();
#endif
      block_site_ops_.emplace_back(sum_op);

      //site block
      const size_t site_block_terms = site_block_hamiltonian_term_group_.size();
      auto psite_block_ops_res_s = std::vector<Tensor *>(site_block_terms);
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_contract_timer.Restart();
#endif
      for (size_t j = 0; j < site_block_terms; j++) {
        psite_block_ops_res_s[j] = new Tensor();
        Contract(site_block_hamiltonian_term_group_[j][0],
                 site_block_hamiltonian_term_group_[j][1],
                 {{}, {}},
                 psite_block_ops_res_s[j]);
      }
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_contract_timer.Suspend();
      slave_hamil_multiply_state_sum_timer.Restart();
#endif
      coefs = std::vector<TenElemT>(site_block_terms, TenElemT(1.0));
      sum_op = Tensor();
      LinearCombine(coefs, psite_block_ops_res_s, TenElemT(0.0), &sum_op);
      for (size_t j = 0; j < site_block_terms; j++) {
        delete psite_block_ops_res_s[j];
      }
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_sum_timer.Suspend();
      slave_hamil_multiply_state_transpose_timer.Restart();
#endif
      sum_op.Transpose({0, 2, 1, 3});
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_transpose_timer.Suspend();
#endif
      site_block_ops_.emplace_back(sum_op);

#ifdef QLMPS_MPI_TIMING_MODE
      //Hamiltonian * state
      slave_hamil_multiply_state_contract_timer.Restart();
#endif
      Tensor temp1, multiplication_res;
      Contract(&block_site_ops_.back(), &state, {{2, 3}, {0, 1}}, &temp1);
      Contract(&temp1, &site_block_ops_.back(), {{2, 3}, {0, 1}}, &multiplication_res);
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_contract_timer.Suspend();
#endif
      ops_num_table_.push_back(task_id);
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_delete_timer.Restart();
#endif
      for (size_t i = 0; i < block_site_terms; i++) {
        delete block_site_hamiltonian_term_group_[i][0];
        delete block_site_hamiltonian_term_group_[i][1];
      }
      for (size_t i = 0; i < site_block_terms; i++) {
        delete site_block_hamiltonian_term_group_[i][0];
        delete site_block_hamiltonian_term_group_[i][1];
      }
#ifdef QLMPS_MPI_TIMING_MODE
      slave_hamil_multiply_state_delete_timer.Suspend();
      slave_hamil_multiply_state_computation_timer.Suspend();
#endif
      send_qlten(world_, kMasterRank, task_id, multiplication_res);
    }
  } while (!terminated);
#ifdef QLMPS_MPI_TIMING_MODE
  slave_hamil_multiply_state_contract_timer.PrintElapsed();
  slave_hamil_multiply_state_computation_timer.PrintElapsed();
  slave_hamil_multiply_state_timer.PrintElapsed();
  slave_hamil_multiply_state_delete_timer.PrintElapsed();
  slave_hamil_multiply_state_sum_timer.PrintElapsed();
  slave_hamil_multiply_state_transpose_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT>
void DMRGMPISlaveExecutor<TenElemT, QNT>::WorkForStaticHamiltonianMultiplyState_() {
  const size_t num_terms = block_site_ops_.size();
#ifdef QLMPS_MPI_TIMING_MODE
  Timer slave_hamil_multiply_state_timer("node-" + std::to_string(id_) + "_hamiltonian_multiply_state_total_time");
  Timer slave_hamil_multiply_state_computation_timer
      ("node-" + std::to_string(id_) + "_hamiltonian_multiply_state_computation_time_task_num="
           + std::to_string(num_terms));
  slave_hamil_multiply_state_computation_timer.Suspend();
#endif
  Tensor state;
  RecvBroadCastQLTensor(world_, state, kMasterRank);

  assert(num_terms == site_block_ops_.size());
  Tensor sub_sum;
  if (num_terms > 0) {
#ifdef QLMPS_MPI_TIMING_MODE
    slave_hamil_multiply_state_computation_timer.Restart();
#endif
    auto multiplication_res = std::vector<Tensor>(num_terms);
    auto pmultiplication_res = std::vector<Tensor *>(num_terms);
    const std::vector<TenElemT> &coefs = std::vector<TenElemT>(num_terms, TenElemT(1.0));
    for (size_t i = 0; i < num_terms; i++) {
      Tensor temp1;
      Contract(&block_site_ops_[i], &state, {{2, 3}, {0, 1}}, &temp1);
      Contract(&temp1, &site_block_ops_[i], {{2, 3}, {0, 1}}, &multiplication_res[i]);
      pmultiplication_res[i] = &multiplication_res[i];
    }
    //TODO: optimize the summation
    LinearCombine(coefs, pmultiplication_res, TenElemT(0.0), &sub_sum);
#ifdef QLMPS_MPI_TIMING_MODE
    slave_hamil_multiply_state_computation_timer.Suspend();
#endif
  } else {
    sub_sum = Tensor(state.GetIndexes());
  }
  auto &bsdt = sub_sum.GetBlkSparDataTen();
  bsdt.MPISend(world_, kMasterRank, 10085);

  Tensor temp_scalar_ten;
  QLTEN_Double sub_overlap = 0.0;
  if (num_terms > 0) {
#ifdef QLMPS_MPI_TIMING_MODE
    slave_hamil_multiply_state_computation_timer.Restart();
#endif
    auto state_dag = Dag(state);
    Contract(
        &sub_sum,
        &state_dag,
        {{0, 1, 2, 3}, {0, 1, 2, 3}},
        &temp_scalar_ten
    );
    sub_overlap = Real(temp_scalar_ten());
#ifdef QLMPS_MPI_TIMING_MODE
    slave_hamil_multiply_state_computation_timer.Suspend();
#endif
  }
  world_.send(kMasterRank, 10087, sub_overlap);
#ifdef QLMPS_MPI_TIMING_MODE
  slave_hamil_multiply_state_computation_timer.PrintElapsed();
  slave_hamil_multiply_state_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT>
void DMRGMPISlaveExecutor<TenElemT, QNT>::WorkForGrowLeftBlockOps_() {
  Tensor mps;
  RecvBroadCastQLTensor(world_, mps, kMasterRank);
  const size_t local_num_ops = block_site_ops_.size();
  Tensor mps_dag = Dag(mps);
  for (size_t i = 0; i < local_num_ops; i++) {
    Tensor lop, temp;
    Contract(&block_site_ops_[i], &mps, {{2, 3}, {0, 1}}, &temp);
    Contract(&temp, &mps_dag, {{0, 1}, {0, 1}}, &lop);
    world_.send(kMasterRank, id_, ops_num_table_[i]);
    send_qlten(world_, kMasterRank, id_, lop);
  }
}

template<typename TenElemT, typename QNT>
void DMRGMPISlaveExecutor<TenElemT, QNT>::WorkForGrowRightBlockOps_() {
  Tensor mps;
  RecvBroadCastQLTensor(world_, mps, kMasterRank);
  const size_t local_num_ops = site_block_ops_.size();
  Tensor mps_dag = Dag(mps);
  for (size_t i = 0; i < local_num_ops; i++) {
    Tensor rop, temp;
    Contract(&mps, &site_block_ops_[i], {{1, 2}, {0, 1}}, &temp);
    Contract(&temp, &mps_dag, {{1, 2}, {1, 2}}, &rop);
    world_.send(kMasterRank, id_, ops_num_table_[i]);
    send_qlten(world_, kMasterRank, id_, rop);
  }
}

}//qlmps

#endif //QLMPS_ALGO_MPI_DMRG_DMRG_MPI_IMPL_SLAVE_H
