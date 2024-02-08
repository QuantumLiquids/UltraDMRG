// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-04
*
* Description: QuantumLiquids/UltraDMRG project. Finite state machine used by iMPO generator.
*/

#ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_IFSM
#define QLMPS_ONE_DIM_TN_MPO_MPOGEN_IFSM

#include "qlmps/one_dim_tn/mpo/mpogen/fsm.h"  //FSMNode, FSMPath

///< [front site fsm node, [ operator, latter site fsm node]]
using FSMNNLinks = std::map<FSMNode, std::map<OpRepr, FSMNode>>;

/**
 * Finite state machine for the generation of iMPO
 * The FSM sites are just before the physical sites
 *
 * possible todo: remove fsm_paths_
 */
class iFSM {
 public:
  iFSM(const size_t unit_cell_site_num,
       const std::vector<OpRepr> &id_ops) :
      unit_cell_site_num_(unit_cell_site_num),
      mid_stat_nums_(unit_cell_site_num, 0),
      fsm_nn_links_set_(unit_cell_site_num) {
    assert(id_ops.size() == unit_cell_site_num);
    for (size_t site = 0; site < unit_cell_site_num; site++) {
      auto &fsm_nn_links = fsm_nn_links_set_[site];
      FSMNode ready_node = FSMNode(site, kFSMReadyStatIdx);
      FSMNode next_ready_node = FSMNode(site + 1, kFSMReadyStatIdx);

      std::map<OpRepr, FSMNode> link_from_ready_to_ready;
      link_from_ready_to_ready.insert(std::make_pair(id_ops[site], next_ready_node));
      fsm_nn_links[ready_node] = link_from_ready_to_ready;

      FSMNode final_node = FSMNode(site, kFSMFinalStatIdx);
      FSMNode next_final_node = FSMNode(site + 1, kFSMFinalStatIdx);

      std::map<OpRepr, FSMNode> link_from_final_to_final;
      link_from_final_to_final.insert(std::make_pair(id_ops[site], next_final_node));
      fsm_nn_links[final_node] = link_from_final_to_final;
    }
  }
  size_t unit_cell_length(void) const { return unit_cell_site_num_; }
  size_t fsm_size(void) const { return unit_cell_site_num_; }
  void AddPath(const size_t head_site, const OpReprVec &);
  FSMPathVec GetFSMPaths(void) const { return fsm_paths_; }
  SparOpReprMatVec GenMatRepr(void) const;
  SparOpReprMatVec GenCompressedMatRepr(const bool show_matrix = false) const;

 private:
  SparOpReprMat CastToMatRepr_(const size_t site) const;

  std::vector<size_t> CalcFSMSiteDims_(void) const {
    std::vector<size_t> fsm_site_dim(mid_stat_nums_.size());
    std::transform(mid_stat_nums_.begin(), mid_stat_nums_.end(), fsm_site_dim.begin(),
                   [](size_t num) { return num + 2; });
    return fsm_site_dim;
  }

  std::vector<size_t> CalcFinalStatDimIdxs_(void) const {
    std::vector<size_t> final_stat_dim_idxs(mid_stat_nums_.size());
    std::transform(mid_stat_nums_.begin(), mid_stat_nums_.end(), final_stat_dim_idxs.begin(),
                   [](size_t num) { return num + 1; });
    return final_stat_dim_idxs;
  }

  size_t unit_cell_site_num_;  // fsm_site_num_ == unit_cell_site_num;
  std::vector<size_t> mid_stat_nums_;
  //all fsm sites include ready and final states
  FSMPathVec fsm_paths_;  // index 0 --> operator head
  std::vector<FSMNNLinks> fsm_nn_links_set_;
};

inline void iFSM::AddPath(const size_t head_site,
                          const OpReprVec &ntrvl_ops) {
  const size_t phys_site_num = ntrvl_ops.size();
  assert(phys_site_num > 1);
  FSMPath fsm_path(phys_site_num);
  for (size_t i = 0; i < phys_site_num; i++) {
    fsm_path.SetOpr(i, ntrvl_ops[i]);
  }
  // Set FSM nodes
  fsm_path.AddNode(0, FSMNode(head_site % unit_cell_site_num_, kFSMReadyStatIdx));
  bool open_new_path(false);
  /**
   * Generate the FSM nodes according the following rules:
   * 1. If FSM path is still on the old path,
   *    check if go along the old path to next node,
   *    but final stat is not allow, except the last operator
   * 2. If the FSM path can go along the old path,
   *    set the FSM path and move to next site
   * 3. If the FSM path cannot go along the old path,
   *    generate new node on next site,
   *    set the FSM path, and add mid_stat_num,
   *    insert the operator links
   * 4. If the FSM path has explore some new node(s),
   *    then the latter site can only follow the step 3.
   * 5. The last node should be the final status.
   */
  for (size_t i = 0; i < phys_site_num - 1; i++) {
    const OpRepr &opr = ntrvl_ops.at(i);
    const auto &last_node = fsm_path.Node(i);
    size_t last_site = last_node.Site();
    size_t target_site = (last_site + 1) % unit_cell_site_num_;
    auto &fsm_nn_links = fsm_nn_links_set_[last_site];
    if (!open_new_path) { //still go along the old path
      auto link_op2node_iter = fsm_nn_links.at(last_node).find(opr);
      if (link_op2node_iter != fsm_nn_links.at(last_node).cend()
          && (link_op2node_iter->second.Stat() != kFSMFinalStatIdx)) {
        //found, still go along the old way
        FSMNode next_node = link_op2node_iter->second;
        fsm_path.AddNode(i + 1, next_node);
      } else {
        //do not find, open new way
        mid_stat_nums_[target_site]++;
        FSMNode new_node(target_site, (long) mid_stat_nums_[target_site]);
        fsm_path.AddNode(i + 1, new_node);
        fsm_nn_links.at(last_node).insert(std::make_pair(opr, new_node));
        open_new_path = true;
      }
    } else {
      mid_stat_nums_[target_site]++;
      FSMNode new_node(target_site, (long) mid_stat_nums_[target_site]);
      fsm_path.AddNode(i + 1, new_node);
      fsm_nn_links.at(last_node).insert(std::make_pair(opr, new_node));
    }
  }
  //the last operator
  const auto &penultimate_node = fsm_path.Node(phys_site_num - 1);
  size_t tail_site = (phys_site_num + head_site) % unit_cell_site_num_; //tail fsm site
  assert(tail_site == (penultimate_node.Site() + 1) % unit_cell_site_num_);
  FSMNode end_node(tail_site, kFSMFinalStatIdx);
  fsm_path.AddNode(phys_site_num, end_node);
  if (open_new_path) {
    auto &fsm_nn_links = fsm_nn_links_set_[(tail_site - 1 + unit_cell_site_num_) % unit_cell_site_num_];
    fsm_nn_links.at(penultimate_node).insert(std::make_pair(ntrvl_ops.back(), end_node));
  } else {
    std::cout << "Same Operator Terms!" << std::endl;
    exit(1);
  }
  fsm_paths_.push_back(fsm_path);
}

SparOpReprMatVec iFSM::GenMatRepr(void) const {
  SparOpReprMatVec fsm_mat_repr;
  for (size_t site = 0; site < unit_cell_site_num_; site++) {
    fsm_mat_repr.push_back(CastToMatRepr_(site));
  }
  return fsm_mat_repr;
}

inline SparOpReprMatVec iFSM::GenCompressedMatRepr(const bool show_matrix) const {
  SparOpReprMatVec comp_mat_repr = GenMatRepr();
  for (size_t i = 0; i < unit_cell_site_num_; ++i) {
    SparOpReprMatColCompresser(comp_mat_repr[i], comp_mat_repr[(i + 1) % unit_cell_site_num_]);
  }
  for (size_t i = unit_cell_site_num_; i > 0; --i) {
    SparOpReprMatRowCompresser(comp_mat_repr[i % unit_cell_site_num_],
                               comp_mat_repr[(i + unit_cell_site_num_ - 1) % unit_cell_site_num_]);
  }

  if (show_matrix) {
    for (size_t i = 0; i < unit_cell_site_num_; ++i) {
      comp_mat_repr[i].Print();
    }
  }
  return comp_mat_repr;
}

SparOpReprMat iFSM::CastToMatRepr_(const size_t site) const {
  const auto &fsm_nn_links = fsm_nn_links_set_.at(site);
  size_t mat_rows = mid_stat_nums_[site] + 2;
  size_t mat_cols = mid_stat_nums_[(site + 1) % unit_cell_site_num_] + 2;
  SparOpReprMat mat(mat_rows, mat_cols);
  for (const auto &[row_node, links] : fsm_nn_links) {
    long row = row_node.Stat();
    if (row == kFSMFinalStatIdx) {
      row = mat_rows - 1;
    }
    for (const auto &[opr, col_node] : links) {
      long col = col_node.Stat();
      if (col == kFSMFinalStatIdx) {
        col = mat_cols - 1;
      }
      mat.SetElem(row, col, opr);
    }
  }
  return mat;
}

#endif //QLMPS_ONE_DIM_TN_MPO_MPOGEN_IFSM
