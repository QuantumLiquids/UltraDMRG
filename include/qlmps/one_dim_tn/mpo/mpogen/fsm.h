// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-11-18 18:28
* 
* Description: QuantumLiquids/UltraDMRG project. Finite state machine used by MPO generator.
*/
#ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_FSM
#define QLMPS_ONE_DIM_TN_MPO_MPOGEN_FSM

#include "qlmps/one_dim_tn/mpo/mpogen/symb_alg/coef_op_alg.h"

#include <vector>

#include <assert.h>

#ifdef Release
#define NDEBUG
#endif

struct FSMNode {
  size_t fsm_site_idx;
  long fsm_stat_idx;
};

using FSMNodeVec = std::vector<FSMNode>;

inline bool operator==(const FSMNode &lhs, const FSMNode &rhs) {
  return (lhs.fsm_site_idx == rhs.fsm_site_idx) &&
      (lhs.fsm_stat_idx == rhs.fsm_stat_idx);
}

inline bool operator!=(const FSMNode &lhs, const FSMNode &rhs) {
  return !(lhs == rhs);
}

struct FSMPath {
  FSMPath(const size_t phys_site_num, const size_t fsm_site_num) :
      fsm_nodes(fsm_site_num), op_reprs(phys_site_num) {
    assert(fsm_nodes.size() == op_reprs.size() + 1);
  }
  FSMNodeVec fsm_nodes;
  OpReprVec op_reprs;
};

using FSMPathVec = std::vector<FSMPath>;

class FSM {
 public:
  FSM(const size_t phys_site_num) :
      phys_site_num_(phys_site_num),
      fsm_site_num_(phys_site_num + 1),
      mid_stat_nums_(phys_site_num + 1, 0),
      has_readys_(phys_site_num + 1, false),
      has_finals_(phys_site_num + 1, false),
      id_op_labels_(phys_site_num, kIdOpLabel) {
    assert(fsm_site_num_ == phys_site_num_ + 1);
  }

  FSM(void) : FSM(0) {}

  size_t phys_size(void) const { return phys_site_num_; }

  size_t fsm_size(void) const { return fsm_site_num_; }

  void AddPath(const size_t, const size_t, const OpReprVec &);

  FSMPathVec GetFSMPaths(void) const { return fsm_paths_; }

  SparOpReprMatVec GenMatRepr(void) const;

  SparOpReprMatVec GenCompressedMatRepr(const bool show_matrix = false) const;

  void ReplaceIdOpLabels(std::vector<OpLabel> &);

 private:
  std::vector<size_t> CalcFSMSiteDims_(void) const;

  std::vector<long> CalcFinalStatDimIdxs_(const std::vector<size_t> &) const;

  void CastFSMPathToMatRepr_(
      const FSMPath &,
      const std::vector<long> &,
      SparOpReprMatVec &) const;

  size_t phys_site_num_;
  size_t fsm_site_num_;
  std::vector<size_t> mid_stat_nums_;
  std::vector<bool> has_readys_;
  std::vector<bool> has_finals_;
  FSMPathVec fsm_paths_;

  std::vector<OpLabel> id_op_labels_;
};

const long kFSMReadyStatIdx = 0;

const long kFSMFinalStatIdx = -1;

inline void FSM::AddPath(
    const size_t head_ntrvl_site_idx, const size_t tail_ntrvl_site_idx,
    const OpReprVec &ntrvl_ops) {
  assert(
      head_ntrvl_site_idx + ntrvl_ops.size() +
          (phys_site_num_ - tail_ntrvl_site_idx - 1) == phys_site_num_);
  FSMPath fsm_path(phys_site_num_, fsm_site_num_);
  // Set operator representations.
  for (size_t i = 0; i < phys_site_num_; ++i) {
    if (i < head_ntrvl_site_idx) {
      fsm_path.op_reprs[i] = OpRepr(id_op_labels_[i]);
    } else if (i > tail_ntrvl_site_idx) {
      fsm_path.op_reprs[i] = OpRepr(id_op_labels_[i]);
    } else {
      fsm_path.op_reprs[i] = ntrvl_ops[i - head_ntrvl_site_idx];
    }
  }
  // Set FSM nodes.
  fsm_path.fsm_nodes[0].fsm_site_idx = 0;
  fsm_path.fsm_nodes[0].fsm_stat_idx = kFSMReadyStatIdx;
  has_readys_[0] = true;
  for (size_t i = 0; i < phys_site_num_; ++i) {
    size_t tgt_fsm_site_idx = i + 1;
    if (i < head_ntrvl_site_idx) {
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_site_idx = tgt_fsm_site_idx;
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_stat_idx = kFSMReadyStatIdx;
      has_readys_[tgt_fsm_site_idx] = true;
    } else if (i >= tail_ntrvl_site_idx) {
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_site_idx = tgt_fsm_site_idx;
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_stat_idx = kFSMFinalStatIdx;
      has_finals_[tgt_fsm_site_idx] = true;
    } else {
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_site_idx = tgt_fsm_site_idx;
      mid_stat_nums_[tgt_fsm_site_idx]++;
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_stat_idx =
          mid_stat_nums_[tgt_fsm_site_idx];
    }
  }
  fsm_paths_.push_back(fsm_path);
}

inline SparOpReprMatVec FSM::GenMatRepr(void) const {
  auto fsm_site_dims = CalcFSMSiteDims_();
  auto final_stat_dim_idxs = CalcFinalStatDimIdxs_(fsm_site_dims);
  SparOpReprMatVec fsm_mat_repr;
  for (size_t i = 0; i < phys_site_num_; ++i) {
    auto mat_rows = fsm_site_dims[i];
    auto mat_cols = fsm_site_dims[i + 1];
    fsm_mat_repr.push_back(SparOpReprMat(mat_rows, mat_cols));
  }
  for (auto &fsm_path: fsm_paths_) {
    CastFSMPathToMatRepr_(fsm_path, final_stat_dim_idxs, fsm_mat_repr);
  }
  return fsm_mat_repr;
}

inline std::vector<size_t> FSM::CalcFSMSiteDims_(void) const {
  std::vector<size_t> fsm_site_dims(fsm_site_num_, 0);
  for (size_t i = 0; i < fsm_site_num_; ++i) {
    auto fsm_site_dim = mid_stat_nums_[i];
    if (has_readys_[i]) { fsm_site_dim++; }
    if (has_finals_[i]) { fsm_site_dim++; }
    fsm_site_dims[i] = fsm_site_dim;
  }
  return fsm_site_dims;
}

inline std::vector<long> FSM::CalcFinalStatDimIdxs_(
    const std::vector<size_t> &fsm_site_dims) const {
  std::vector<long> final_stat_dim_idxs(fsm_site_num_, -1);
  for (size_t i = 0; i < fsm_site_num_; ++i) {
    if (has_finals_[i]) {
      if (has_readys_[i]) {
        final_stat_dim_idxs[i] = fsm_site_dims[i] - 1;
      } else {
        final_stat_dim_idxs[i] = 0;
      }
    }
  }
  return final_stat_dim_idxs;
}

inline void FSM::CastFSMPathToMatRepr_(
    const FSMPath &fsm_path,
    const std::vector<long> &final_stat_dim_idxs,
    SparOpReprMatVec &fsm_mat_repr) const {
  for (size_t i = 0; i < phys_site_num_; ++i) {
    auto tgt_row_fsm_node = fsm_path.fsm_nodes[i];
    auto tgt_col_fsm_node = fsm_path.fsm_nodes[i + 1];
    auto tgt_op = fsm_path.op_reprs[i];

    size_t tgt_row_idx, tgt_col_idx;
    if (tgt_row_fsm_node.fsm_stat_idx == kFSMFinalStatIdx) {
      tgt_row_idx = final_stat_dim_idxs[i];
    } else if ((!has_readys_[i]) && (!has_finals_[i])) {
      tgt_row_idx = tgt_row_fsm_node.fsm_stat_idx - 1;
    } else {
      tgt_row_idx = tgt_row_fsm_node.fsm_stat_idx;
    }
    if (tgt_col_fsm_node.fsm_stat_idx == kFSMFinalStatIdx) {
      tgt_col_idx = final_stat_dim_idxs[i + 1];
    } else if ((!has_readys_[i + 1]) && (!has_finals_[i + 1])) {
      tgt_col_idx = tgt_col_fsm_node.fsm_stat_idx - 1;
    } else {
      tgt_col_idx = tgt_col_fsm_node.fsm_stat_idx;
    }

    if (fsm_mat_repr[i](tgt_row_idx, tgt_col_idx) == kNullOpRepr) {
      fsm_mat_repr[i].SetElem(tgt_row_idx, tgt_col_idx, tgt_op);
    } else if (fsm_mat_repr[i](tgt_row_idx, tgt_col_idx) != tgt_op) {
      auto new_op = fsm_mat_repr[i](tgt_row_idx, tgt_col_idx) + tgt_op;
      fsm_mat_repr[i].SetElem(tgt_row_idx, tgt_col_idx, new_op);
    } else {
      // Do nothing
    }
  }
}

inline SparOpReprMatVec FSM::GenCompressedMatRepr(const bool show_matrix) const {
  SparOpReprMatVec comp_mat_repr = GenMatRepr();
  //  for (size_t i = phys_site_num_-1; i >= phys_site_num_/2; --i) {
//    SparOpReprMatRowCompresser(comp_mat_repr[i], comp_mat_repr[i-1]);
//  }
//  for (size_t i = 0; i <= phys_site_num_/2; ++i) {
//    SparOpReprMatColCompresser(comp_mat_repr[i], comp_mat_repr[i+1]);
//  }
//  for (size_t i = phys_site_num_-1; i > 0; --i) {
//    SparOpReprMatRowCompresser(comp_mat_repr[i], comp_mat_repr[i-1]);
//  }
  for (size_t i = 0; i < phys_site_num_ - 1; ++i) {
    SparOpReprMatColCompresser(comp_mat_repr[i], comp_mat_repr[i + 1]);
  }
  for (size_t i = phys_site_num_ - 1; i > 0; --i) {
    SparOpReprMatRowCompresser(comp_mat_repr[i], comp_mat_repr[i - 1]);
  }

  if (show_matrix) {
    for (size_t i = 0; i < phys_site_num_; ++i) {
      comp_mat_repr[i].Print();
    }
  }

  return comp_mat_repr;
}

inline void FSM::ReplaceIdOpLabels(std::vector<OpLabel> &new_id_op_labels) {
  assert(new_id_op_labels.size() == id_op_labels_.size());
  id_op_labels_ = new_id_op_labels;
}

template<typename ConvObjT>
class LabelConvertor {
 public:
  LabelConvertor(void) = default;

  LabelConvertor(const ConvObjT &id) : conv_obj_hub_({id}) {}

  LabelConvertor<ConvObjT> &operator=(const LabelConvertor<ConvObjT> &rhs) {
    conv_obj_hub_ = rhs.conv_obj_hub_;
    return *this;
  }

  using ConvObjVec = std::vector<ConvObjT>;

  size_t Convert(const ConvObjT &conv_obj) {
    auto poss_it = std::find(
        conv_obj_hub_.cbegin(), conv_obj_hub_.cend(), conv_obj);
    if (poss_it == conv_obj_hub_.cend()) {
      conv_obj_hub_.push_back(conv_obj);
      size_t label = conv_obj_hub_.size() - 1;
#ifndef NDEBUG
      auto poss_it_minus = std::find(
          conv_obj_hub_.cbegin(), conv_obj_hub_.cend(), -conv_obj);
      if (poss_it_minus != conv_obj_hub_.cend()) {
        std::cout << "warning: label linear relative to previous: " << conv_obj << std::endl;
      }
#endif
      return label;
    } else {
      size_t label = poss_it - conv_obj_hub_.cbegin();
      return label;
    }
  }

  ConvObjVec GetLabelObjMapping(void) { return conv_obj_hub_; }
 private:
  ConvObjVec conv_obj_hub_;
};
#endif /* ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_FSM */
