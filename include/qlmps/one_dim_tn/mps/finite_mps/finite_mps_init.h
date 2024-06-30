// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-29 22:11
*
* Description: QuantumLiquids/UltraDMRG project. Finite MPS initializations.
*/
#ifndef QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_INIT_H
#define QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_INIT_H

#include "qlmps/one_dim_tn/mps/finite_mps/finite_mps.h"    // FiniteMPS
#include "qlten/qlten.h"

#ifdef Release
#define NDEBUG
#endif
#include <assert.h>

namespace qlmps {
using namespace qlten;


// Forward declarations
//// For random initialize MPS operation.
//Index GenHeadRightVirtBond(const Index &, const QN &, const long);

//Index GenBodyRightVirtBond(
//const Index &, const Index &, const QN &, const long);

//Index GenTailLeftVirtBond(const Index &, const QN &, const long);

//Index GenBodyLeftVirtBond(const Index &, const Index &, const QN &, const long);

//void DimCut(std::vector<QNSector> &, const long, const long);


//// Helpers
//inline bool GreaterQNSectorDim(const QNSector &qnsct1, const QNSector &qnsct2) {
//return qnsct1.dim > qnsct2.dim;
//}


// MPS initializations.
//template <typename TenType>
//void RandomInitMps(
//MPS<TenType> &mps,
//const QN &tot_div,
//const QN &zero_div,
//const long dmax
//) {
//MpsFree(mps);
//auto sites_info = mps.GetSitesInfo();
//auto pb_out_set = sites_info.sites;
//Index lvb, rvb;

//// Left to center.
//rvb = GenHeadRightVirtBond(pb_out_set[0], tot_div, dmax);
//mps(0) = new TenType({pb_out_set[0], rvb});
//mps(0)->Random(tot_div);
//assert(Div(mps[0]) == tot_div);
//auto N = mps.size();
//for (std::size_t i = 1; i < N/2; ++i) {
//lvb = InverseIndex(rvb);
//rvb = GenBodyRightVirtBond(lvb, pb_out_set[i], zero_div, dmax);
//mps(i) = new TenType({lvb, pb_out_set[i], rvb});
//mps(i)->Random(zero_div);
//assert(Div(mps[i]) == zero_div);
//}
//auto cent_bond = rvb;

//// Right to center.
//lvb = GenTailLeftVirtBond(pb_out_set[N-1], zero_div, dmax);
//mps(N-1) = new TenType({lvb, pb_out_set[N-1]});
//mps(N-1)->Random(zero_div);
//assert(Div(mps[N-1]) == zero_div);
//for (std::size_t i = N-2; i > N/2; --i) {
//rvb = InverseIndex(lvb);
//lvb = GenBodyLeftVirtBond(rvb, pb_out_set[i], zero_div, dmax);
//mps(i) = new TenType({lvb, pb_out_set[i], rvb});
//mps(i)->Random(zero_div);
//assert(Div(mps[i]) == zero_div);
//}

//rvb = InverseIndex(lvb);
//lvb = InverseIndex(cent_bond);
//mps(N/2) = new TenType({lvb, pb_out_set[N/2], rvb});
//mps(N/2)->Random(zero_div);
//assert(Div(mps[N/2]) == zero_div);

//// Centralize MPS.
//mps.Centralize(0);
//}


//inline Index GenHeadRightVirtBond(
//const Index &pb, const QN &tot_div, const long dmax) {
//std::vector<QNSector> new_qnscts;
//for (auto &qnsct : pb.qnscts) {
//auto new_qn = tot_div - qnsct.qn;
//auto has_qn = false;
//for (auto &new_qnsct : new_qnscts) {
//if (new_qnsct.qn == new_qn) {
//new_qnsct.dim += qnsct.dim;
//has_qn = true;
//break;
//}
//}
//if (!has_qn) {
//new_qnscts.push_back(QNSector(new_qn, qnsct.dim));
//}
//}
//DimCut(new_qnscts, dmax, pb.dim);
//return Index(new_qnscts, OUT);
//}


//inline Index GenBodyRightVirtBond(
//const Index &lvb, const Index &pb, const QN &zero_div, const long dmax) {
//std::vector<QNSector> new_qnscts;
//for (auto &lvqnsct : lvb.qnscts) {
//for (auto &pqnsct : pb.qnscts) {
//auto poss_rvb_qn = zero_div + lvqnsct.qn - pqnsct.qn;
//auto has_qn = false;
//for (auto &new_qnsct : new_qnscts) {
//if (poss_rvb_qn == new_qnsct.qn) {
//new_qnsct.dim += lvqnsct.dim;
//has_qn = true;
//break;
//}
//}
//if (!has_qn) {
//new_qnscts.push_back(QNSector(poss_rvb_qn, lvqnsct.dim));
//}
//}
//}
//DimCut(new_qnscts, dmax, pb.dim);
//return Index(new_qnscts, OUT);
//}


//inline Index GenTailLeftVirtBond(
//const Index &pb, const QN &zero_div, const long dmax) {
//std::vector<QNSector> new_qnscts;
//for (auto &qnsct : pb.qnscts) {
//auto new_qn = qnsct.qn - zero_div;
//auto has_qn = false;
//for (auto &new_qnsct : new_qnscts) {
//if (new_qnsct.qn == new_qn) {
//new_qnsct.dim += qnsct.dim;
//has_qn = true;
//break;
//}
//}
//if (!has_qn) {
//new_qnscts.push_back(QNSector(new_qn, qnsct.dim));
//}
//}
//DimCut(new_qnscts, dmax, pb.dim);
//return Index(new_qnscts, IN);
//}


//inline Index GenBodyLeftVirtBond(
//const Index &rvb, const Index &pb, const QN &zero_div, const long dmax) {
//std::vector<QNSector> new_qnscts;
//for (auto &rvqnsct : rvb.qnscts) {
//for (auto &pqnsct : pb.qnscts) {
//auto poss_lvb_qn = pqnsct.qn - zero_div + rvqnsct.qn;
//auto has_qn = false;
//for (auto &new_qnsct : new_qnscts) {
//if (poss_lvb_qn == new_qnsct.qn) {
//new_qnsct.dim += rvqnsct.dim;
//has_qn = true;
//break;
//}
//}
//if (!has_qn) {
//new_qnscts.push_back(QNSector(poss_lvb_qn, rvqnsct.dim));
//}
//}
//}
//DimCut(new_qnscts, dmax, pb.dim);
//return Index(new_qnscts, IN);
//}


//inline void DimCut(
//std::vector<QNSector> &qnscts, const long dmax, const long pdim) {
//std::sort(qnscts.begin(), qnscts.end(), GreaterQNSectorDim);
//auto kept_qn_cnt = 0;
//auto dim = 0;
//for (auto &qnsct : qnscts) {
//if (dim + qnsct.dim < dmax) {
//dim += qnsct.dim;
//kept_qn_cnt++;
//} else if (dim + qnsct.dim == dmax) {
//dim += qnsct.dim;
//kept_qn_cnt++;
//break;
//} else {
//qnsct.dim -= (dim + qnsct.dim - dmax);
//kept_qn_cnt++;
//break;
//}
//}
//qnscts.resize(kept_qn_cnt);
//}


/**
Initialize a finite MPS as a direct product state.
*/
template<typename TenElemT, typename QNT>
void DirectStateInitMps(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::vector<size_t> &stat_labs
) {
  using TenT = QLTensor<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;

  auto N = mps.size();
  assert(N == stat_labs.size());
  for (size_t i = 0; i < mps.size(); ++i) { mps.dealloc(i); }
  auto sites_info = mps.GetSitesInfo();
  auto pb_out_set = sites_info.sites;
  IndexT lvb, rvb;

  // Calculate total quantum number
  auto div = pb_out_set[0].GetQNSctFromActualCoor(stat_labs[0]).GetQn();
  for (size_t i = 1; i < N; ++i) {
    div += pb_out_set[i].GetQNSctFromActualCoor(stat_labs[i]).GetQn();
  }
  // Calculate zero quantum number
  QNT zero_qn = div - div;

  auto stat_lab = stat_labs[0];
  auto rvb_qn = div - pb_out_set[0].GetQNSctFromActualCoor(stat_lab).GetQn();
  lvb = IndexT({QNSctT(zero_qn, 1)}, TenIndexDirType::IN);
  rvb = IndexT({QNSctT(rvb_qn, 1)}, TenIndexDirType::OUT);
  mps(0) = new TenT({lvb, pb_out_set[0], rvb});
  (mps[0])({0, stat_lab, 0}) = 1;

  for (size_t i = 1; i < N - 1; ++i) {
    lvb = InverseIndex(rvb);
    stat_lab = stat_labs[i];
    rvb_qn = lvb.GetQNSctFromActualCoor(0).GetQn() -
        pb_out_set[i].GetQNSctFromActualCoor(stat_lab).GetQn();
    rvb = IndexT({QNSctT(rvb_qn, 1)}, TenIndexDirType::OUT);
    mps(i) = new TenT({lvb, pb_out_set[i], rvb});
    (mps[i])({0, stat_lab, 0}) = 1;
  }

  lvb = InverseIndex(rvb);
  rvb = IndexT({QNSctT(zero_qn, 1)}, TenIndexDirType::OUT);
  mps(N - 1) = new TenT({lvb, pb_out_set[N - 1], rvb});
  stat_lab = stat_labs[N - 1];
  (mps[N - 1])({0, stat_lab, 0}) = 1;

  // Centralize MPS.
  mps.Centralize(0);
}

/**
Initialize a finite MPS as a direct product state.
Every tensor in MPS has 0-flux.
*/
template<typename TenElemT, typename QNT>
void DirectStateInitZeroDivMps(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::vector<size_t> &stat_labs
) {
  using TenT = QLTensor<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;

  auto N = mps.size();
  assert(N == stat_labs.size());
  for (size_t i = 0; i < mps.size(); ++i) { mps.dealloc(i); }
  auto sites_info = mps.GetSitesInfo();
  auto pb_out_set = sites_info.sites;
  IndexT lvb, rvb;

  // Calculate total quantum number
  auto div = pb_out_set[0].GetQNSctFromActualCoor(stat_labs[0]).GetQn();
  for (size_t i = 1; i < N; ++i) {
    div += pb_out_set[i].GetQNSctFromActualCoor(stat_labs[i]).GetQn();
  }
  // Calculate zero quantum number
  QNT zero_qn = div - div;

  auto stat_lab = stat_labs[0];
  lvb = IndexT({QNSctT(zero_qn, 1)}, TenIndexDirType::IN);
  auto rvb_qn = zero_qn - pb_out_set[0].GetQNSctFromActualCoor(stat_lab).GetQn();
  rvb = IndexT({QNSctT(rvb_qn, 1)}, TenIndexDirType::OUT);
  mps(0) = new TenT({lvb, pb_out_set[0], rvb});
  (mps[0])({0, stat_lab, 0}) = 1;

  for (size_t i = 1; i < N - 1; ++i) {
    lvb = InverseIndex(rvb);
    stat_lab = stat_labs[i];
    rvb_qn = lvb.GetQNSctFromActualCoor(0).GetQn() -
        pb_out_set[i].GetQNSctFromActualCoor(stat_lab).GetQn();
    rvb = IndexT({QNSctT(rvb_qn, 1)}, TenIndexDirType::OUT);
    mps(i) = new TenT({lvb, pb_out_set[i], rvb});
    (mps[i])({0, stat_lab, 0}) = 1;
  }

  lvb = InverseIndex(rvb);
  rvb = IndexT({QNSctT(zero_qn, 1)}, TenIndexDirType::OUT);
  mps(N - 1) = new TenT({lvb, pb_out_set[N - 1], rvb});
  stat_lab = stat_labs[N - 1];
  (mps[N - 1])({0, stat_lab, 0}) = 1;

  // Centralize MPS.
  mps.Centralize(0);
}

/**
Initialize a finite MPS as a extended direct product state.
*/
template<typename TenElemT, typename QNT>
void ExtendDirectRandomInitMps(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::vector<std::vector<size_t>> &stat_labs_set,
    const size_t enlarged_dim
) {
  using TenT = QLTensor<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;

  auto fusion_stats_num = stat_labs_set.size();
  assert(fusion_stats_num >= 1);
  auto N = mps.size();
  assert(N == stat_labs_set[0].size());
  for (size_t i = 0; i < mps.size(); ++i) { mps.dealloc(i); }
  auto sites_info = mps.GetSitesInfo();
  auto pb_out_set = sites_info.sites;
  IndexT lvb, rvb;
  std::vector<QNSctT> rvb_qnscts;

  // Calculate total quantum number
  auto div = pb_out_set[0].GetQNSctFromActualCoor(stat_labs_set[0][0]).GetQn();
  for (size_t i = 1; i < N; ++i) {
    div += pb_out_set[i].GetQNSctFromActualCoor(stat_labs_set[0][i]).GetQn();
  }
  // Calculate zero quantum number
  auto zero_qn = div - div;

  // Deal with MPS head local tensor
  for (size_t i = 0; i < fusion_stats_num; ++i) {
    auto stat_lab = stat_labs_set[i][0];
    auto rvb_qn = div - pb_out_set[0].GetQNSctFromActualCoor(stat_lab).GetQn();
    rvb_qnscts.push_back(QNSctT(rvb_qn, enlarged_dim));
  }
  lvb = IndexT({QNSctT(zero_qn, 1)}, TenIndexDirType::IN);
  rvb = IndexT(rvb_qnscts, TenIndexDirType::OUT);
  rvb_qnscts.clear();
  mps(0) = new TenT({lvb, pb_out_set[0], rvb});
  mps[0].Random(div);

  // Deal with MPS middle local tensors
  for (size_t i = 1; i < N - 1; ++i) {
    lvb = InverseIndex(rvb);
    for (size_t j = 0; j < fusion_stats_num; ++j) {
      auto stat_lab = stat_labs_set[j][i];
      auto rvb_qn = lvb.GetQNSctFromActualCoor(j * enlarged_dim).GetQn() -
          pb_out_set[i].GetQNSctFromActualCoor(stat_lab).GetQn();
      rvb_qnscts.push_back(QNSctT(rvb_qn, enlarged_dim));
    }
    rvb = IndexT(rvb_qnscts, TenIndexDirType::OUT);
    mps(i) = new TenT({lvb, pb_out_set[i], rvb});
    rvb_qnscts.clear();
    mps[i].Random(zero_qn);
  }

  // Deal with MPS tail local tensor
  lvb = InverseIndex(rvb);
  rvb = IndexT({QNSctT(zero_qn, 1)}, TenIndexDirType::OUT);
  mps(N - 1) = new TenT({lvb, pb_out_set[N - 1], rvb});
  mps[N - 1].Random(zero_qn);

  // Centralize MPS.
  mps.Centralize(0);
  mps[0].Normalize();
}
} /* qlmps */
#endif /* ifndef QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_INIT_H */
