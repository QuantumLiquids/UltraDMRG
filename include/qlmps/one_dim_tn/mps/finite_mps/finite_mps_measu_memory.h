// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2019-10-08 22:18
*
* Description: QuantumLiquids/UltraDMRG project.
*              Finite MPS observation measurements. The MPS data should all be loaded in memory before the measurement.
*/

/**
@file finite_mps_measu.h
@brief Finite MPS observation measurements.
*/
#ifndef QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_MEASU_H
#define QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_MEASU_H

#include "qlmps/one_dim_tn/mps/finite_mps/finite_mps.h"    // FiniteMPS
#include "qlten/qlten.h"

#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>

namespace qlmps {
using namespace qlten;

template<typename TenElemT, typename QNT>
using TenVV = std::vector<std::vector<QLTensor<TenElemT, QNT>>>;

template<typename TenElemT, typename QNT>
using TenVVV = std::vector<std::vector<std::vector<QLTensor<TenElemT, QNT>>>>;

/**
Measurement result for a set specific operator(s).

@tparam AvgT Data type of the measurement, real or complex.
*/
template<typename AvgT>
struct MeasuResElem {
  MeasuResElem(void) = default;
  MeasuResElem(const std::vector<size_t> &sites, const AvgT avg) :
      sites(sites), avg(avg) {}

  std::vector<size_t> sites;  ///< Site indexes of the operators.
  AvgT avg;                 ///< average of the observation.
};

/**
A list of measurement results.
*/
template<typename AvgT>
using MeasuRes = std::vector<MeasuResElem<AvgT>>;

template<typename AvgT>
using MeasuResSet = std::vector<MeasuRes<AvgT>>;

template<typename TenElemT, typename QNT>
MeasuResElem<TenElemT> OneSiteOpAvg(
    const QLTensor<TenElemT, QNT> &, const QLTensor<TenElemT, QNT> &,
    const size_t, const size_t
);

template<typename TenElemT, typename QNT>
MeasuResElem<TenElemT> MultiSiteOpAvg(
    const FiniteMPS<TenElemT, QNT> &,
    const std::vector<QLTensor<TenElemT, QNT>> &,
    const std::vector<std::vector<QLTensor<TenElemT, QNT>>> &,
    const std::vector<size_t> &
);

template<typename TenElemT, typename QNT>
MeasuResElem<TenElemT> MultiSiteOpAvg(
    const FiniteMPS<TenElemT, QNT> &,
    const std::vector<QLTensor<TenElemT, QNT>> &,
    const std::vector<QLTensor<TenElemT, QNT>> &,
    const std::vector<size_t> &
);

template<typename TenElemT, typename QNT>
TenElemT OpsVecAvg(
    const FiniteMPS<TenElemT, QNT> &,
    const std::vector<QLTensor<TenElemT, QNT>> &,
    const size_t,
    const size_t
);

template<typename TenElemT, typename QNT>
void CtrctMidTen(
    const FiniteMPS<TenElemT, QNT> &, const size_t,
    const QLTensor<TenElemT, QNT> &, const QLTensor<TenElemT, QNT> &,
    QLTensor<TenElemT, QNT> *&
);

template<typename AvgT>
void DumpMeasuRes(const MeasuRes<AvgT> &, const std::string &);

// Helpers.
inline bool IsOrderKept(const std::vector<size_t> &sites) {
  for (size_t i = 0; i < sites.size() - 1; ++i) {
    if (sites[i] > sites[i + 1]) { return false; }
  }
  return true;
}

template<typename T>
inline void DumpSites(std::ofstream &ofs, const std::vector<T> &sites) {
  ofs << "[";
  for (auto it = sites.begin(); it != sites.end() - 1; ++it) {
    ofs << *it << ", ";
  }
  ofs << sites.back();
  ofs << "], ";
}

inline void DumpAvgVal(std::ofstream &ofs, const QLTEN_Double avg) {
  ofs << std::setw(14) << std::setprecision(12) << avg;
}

inline void DumpAvgVal(std::ofstream &ofs, const QLTEN_Complex avg) {
  ofs << "[";
  ofs << std::setw(14) << std::setprecision(12) << avg.real();
  ofs << ", ";
  ofs << std::setw(14) << std::setprecision(12) << avg.imag();
  ofs << "]";
}


// Measure one-site operator.
/**
Measure a single one-site operator on each sites of the finite MPS.

@tparam TenElemT Type of the tensor element.
@tparam QNT Quantum number type.

@param mps To-be-measured MPS.
@param op The single one-site operator.
@param res_file_basename The basename of the output file.
*/
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureOneSiteOp(
    FiniteMPS<TenElemT, QNT> &mps,
    const QLTensor<TenElemT, QNT> &op,
    const std::string &res_file_basename
) {
  auto N = mps.size();
  MeasuRes<TenElemT> measu_res(N);
  for (size_t i = 0; i < N; ++i) {
    mps.Centralize(i);
    measu_res[i] = OneSiteOpAvg(mps[i], op, i, N);
  }
  DumpMeasuRes(measu_res, res_file_basename);
  return measu_res;
}

/**
Measure a list of one-site operators on each sites of the finite MPS.

@tparam TenElemT Type of the tensor element.
@tparam QNT Quantum number type.

@param mps To-be-measured MPS.
@param ops A list of one-site operators.
@param res_file_basename The basename of the output file.
*/
template<typename TenElemT, typename QNT>
MeasuResSet<TenElemT> MeasureOneSiteOp(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::vector<QLTensor<TenElemT, QNT>> &ops,
    const std::vector<std::string> &res_file_basenames
) {
  auto op_num = ops.size();
  assert(op_num == res_file_basenames.size());
  auto N = mps.size();
  MeasuResSet<TenElemT> measu_res_set(op_num);
  for (auto &measu_res : measu_res_set) {
    measu_res = MeasuRes<TenElemT>(N);
  }
  for (size_t i = 0; i < N; ++i) {
    mps.Centralize(i);
    for (size_t j = 0; j < op_num; ++j) {
      measu_res_set[j][i] = OneSiteOpAvg(mps[i], ops[j], i, N);
    }
  }
  for (size_t i = 0; i < op_num; ++i) {
    DumpMeasuRes(measu_res_set[i], res_file_basenames[i]);
  }
  return measu_res_set;
}


// Measure two-site operator.
/**
Measure a two-site operator, for example, \f$\langle A_{i} O^{[1]}_{i+1} \cdots O^{[m]}_{j-1} B_{j} \rangle\f$.
The insert operators \f$O_{k}\f$ can be different for each measure event.

@tparam TenElemT Type of the tensor element, real or complex.
@param mps To-be-measured MPS.
@param phys_ops Physical operators \f$A\f$ and \f$B\f$.
@param inst_ops_set Insert operators \f$O^{[1]}, \cdots , O^{[m]} \f$ for each
       measure event. The size must equal to the number of measure events.
@param sites_set The indexes of the two physical operators with ascending order
       for each measure event. Its size defines the number of measure events.
@param res_file_basename The basename of the output file.
*/
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureTwoSiteOp(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::vector<QLTensor<TenElemT, QNT>> &phys_ops,
    const std::vector<std::vector<QLTensor<TenElemT, QNT>>> &inst_ops_set,
    const std::vector<std::vector<size_t>> &sites_set,
    const std::string &res_file_basename
) {
  // Deal with two physical operators
  assert(phys_ops.size() == 2);
  auto measu_event_num = sites_set.size();
  TenVV<TenElemT, QNT> phys_ops_set(measu_event_num, phys_ops);

  // Deal with inset operators for each measure event
  assert(inst_ops_set.size() == measu_event_num);
  TenVVV<TenElemT, QNT> inst_ops_set_set;
  for (size_t i = 0; i < measu_event_num; ++i) {
    assert(sites_set[i].size() == 2);
    assert((sites_set[i][1] - sites_set[i][0] - 1) == inst_ops_set[i].size());
    inst_ops_set_set.push_back({inst_ops_set[i]});
  }

  return MeasureMultiSiteOp(
      mps,
      phys_ops_set,
      inst_ops_set_set,
      sites_set,
      res_file_basename
  );
}

/**
Measure a two-site operator, for example, \f$\langle A_{i} O_{i+1} \cdots O_{j-1} B_{j} \rangle\f$.
The insert operators \f$O_{k}\f$ must be same at each sites and for each measure event.

@tparam TenElemT Type of the tensor element, real or complex.
@param mps To-be-measured MPS.
@param phys_ops Physical operators \f$A\f$ and \f$B\f$.
@param inst_op Insert operator \f$ O \f$.
@param sites_set The indexes of the two physical operators with ascending order
       for each measure event. Its size defines the number of measure events.
@param res_file_basename The basename of the output file.
*/
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureTwoSiteOp(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::vector<QLTensor<TenElemT, QNT>> &phys_ops,
    const QLTensor<TenElemT, QNT> &inst_op,
    const std::vector<std::vector<size_t>> &sites_set,
    const std::string &res_file_basename
) {
  assert(phys_ops.size() == 2);
  auto measu_event_num = sites_set.size();
  TenVV<TenElemT, QNT> phys_ops_set(measu_event_num, phys_ops);
  TenVV<TenElemT, QNT> inst_ops_set(measu_event_num, {inst_op});
  return MeasureMultiSiteOp(
      mps,
      phys_ops_set,
      inst_ops_set,
      sites_set,
      res_file_basename
  );
}


// Measure multi-site operator.
/**
Measure a multi-site operator, for example, \f$\langle A_{i} O^{[1]}_{i+1} 
\cdots O^{[m]}_{j-1} B_{j} O^{[n]}_{j+1} \cdots O^{[p]}_{k-1} C_{k} 
O^{[q]}_{k+1} \cdots\rangle\f$. All the physical operators and insert operators
should be defined by the user in each measure events.

@tparam TenElemT Type of the tensor element, real or complex.
@param mps To-be-measured MPS.
@param phys_ops_set Physical operators \f$A, B, C, \cdots\f$ for each measure
       events.
@param inst_ops_set_set Insert operators \f$\{O^{[1]}, \cdots , O^{[m]}\},
       \{O^{[n]}, \cdots , O^{[p]}\}, \cdots\f$ for each measure event. The size
       must equal to the number of measure events.
@param sites_set The indexes of the two physical operators with ascending order
       for each measure event. Its size defines the number of measure events.
@param res_file_basename The basename of the output file.
*/
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureMultiSiteOp(
    FiniteMPS<TenElemT, QNT> &mps,
    const TenVV<TenElemT, QNT> &phys_ops_set,
    const TenVVV<TenElemT, QNT> &inst_ops_set_set,
    const std::vector<std::vector<size_t>> &sites_set,
    const std::string &res_file_basename
) {
  auto measu_event_num = sites_set.size();
  MeasuRes<TenElemT> measu_res(measu_event_num);
  for (size_t i = 0; i < measu_event_num; ++i) {
    auto &phys_ops = phys_ops_set[i];
    auto &inst_ops_set = inst_ops_set_set[i];
    auto &sites = sites_set[i];
    assert(sites.size() > 1);
    assert(IsOrderKept(sites));
    mps.Centralize(sites[0]);
    measu_res[i] = MultiSiteOpAvg(mps, phys_ops, inst_ops_set, sites);
  }
  DumpMeasuRes(measu_res, res_file_basename);
  return measu_res;
}

/**
Measure a multi-site operator, for example, \f$\langle A_{i} O^{[m]}_{i+1} 
\cdots O^{[m]}_{j-1} B_{j} O^{[n]}_{j+1} \cdots O^{[n]}_{k-1} C_{k} 
O^{[p]}_{k+1} \cdots\rangle\f$. The insert operators between two given physical
operators are the same.

@tparam TenElemT Type of the tensor element, real or complex.
@param mps To-be-measured MPS.
@param phys_ops_set Physical operators \f$A, B, C, \cdots\f$ for each measure
       events.
@param inst_ops_set Insert operators \f$O^{[m]}, O^{[n]}, O^{[p]}, \cdots \f$
       for each measure event. The size must equal to the number of measure
       events.
@param sites_set The indexes of the two physical operators with ascending order
       for each measure event. Its size defines the number of measure events.
@param res_file_basename The basename of the output file.
*/
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureMultiSiteOp(
    FiniteMPS<TenElemT, QNT> &mps,
    const TenVV<TenElemT, QNT> &phys_ops_set,
    const TenVV<TenElemT, QNT> &inst_ops_set,
    const std::vector<std::vector<size_t>> &sites_set,
    const std::string &res_file_basename
) {
  auto measu_event_num = sites_set.size();
  MeasuRes<TenElemT> measu_res(measu_event_num);
  for (std::size_t i = 0; i < measu_event_num; ++i) {
    auto &phys_ops = phys_ops_set[i];
    auto &inst_ops = inst_ops_set[i];
    auto &sites = sites_set[i];
    assert(sites.size() > 1);
    assert(IsOrderKept(sites));
    mps.Centralize(sites[0]);
    measu_res[i] = MultiSiteOpAvg(mps, phys_ops, inst_ops, sites);
  }
  DumpMeasuRes(measu_res, res_file_basename);
  return measu_res;
}

// Averages.
template<typename TenElemT, typename QNT>
MeasuResElem<TenElemT> OneSiteOpAvg(
    const QLTensor<TenElemT, QNT> &cent_ten,
    const QLTensor<TenElemT, QNT> &op,
    const size_t site,
    const size_t N
) {
  std::vector<size_t> ta_ctrct_axes1{1};
  std::vector<size_t> tb_ctrct_axes1{0};
  std::vector<size_t> ta_ctrct_axes2{0, 2, 1};
  std::vector<size_t> tb_ctrct_axes2{0, 1, 2};
  QLTensor<TenElemT, QNT> temp_ten, res_ten;
  Contract(&cent_ten, &op, {ta_ctrct_axes1, tb_ctrct_axes1}, &temp_ten);
  auto cent_ten_dag = Dag(cent_ten);
  Contract(
      &temp_ten, &cent_ten_dag,
      {ta_ctrct_axes2, tb_ctrct_axes2},
      &res_ten
  );
  TenElemT avg = res_ten();
  return MeasuResElem<TenElemT>({site}, avg);
}

template<typename TenElemT, typename QNT>
MeasuResElem<TenElemT> MultiSiteOpAvg(
    const FiniteMPS<TenElemT, QNT> &mps,
    const std::vector<QLTensor<TenElemT, QNT>> &phys_ops,
    const TenVV<TenElemT, QNT> &inst_ops_set,
    const std::vector<size_t> &sites
) {
  auto inst_ops_num = inst_ops_set.size();
  auto phys_op_num = phys_ops.size();
  // All the insert operators are at the middle or
  // has a tail string behind the last physical operator.
  assert((phys_op_num == (inst_ops_num + 1)) || (phys_op_num == inst_ops_num));
  std::vector<QLTensor<TenElemT, QNT>> ops;
  auto middle_inst_ops_num = phys_op_num - 1;
  for (size_t i = 0; i < middle_inst_ops_num; ++i) {
    ops.push_back(phys_ops[i]);
    for (auto &inst_op : inst_ops_set[i]) {
      ops.push_back(inst_op);
    }
  }
  ops.push_back(phys_ops.back());
  if (inst_ops_num == phys_op_num) {    // Deal with tail insert operator string.
    for (auto &tail_inst_op : inst_ops_set.back()) {
      ops.push_back(tail_inst_op);
    }
  }
  auto head_site = sites.front();
  auto tail_site = head_site + ops.size() - 1;
  auto avg = OpsVecAvg(mps, ops, head_site, tail_site);

  return MeasuResElem<TenElemT>(sites, avg);
}

template<typename TenElemT, typename QNT>
MeasuResElem<TenElemT> MultiSiteOpAvg(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::vector<QLTensor<TenElemT, QNT>> &phys_ops,
    const std::vector<QLTensor<TenElemT, QNT>> &inst_ops,
    const std::vector<size_t> &sites
) {
  auto inst_op_num = inst_ops.size();
  auto phys_op_num = phys_ops.size();
  assert(phys_op_num == (inst_op_num + 1));
  TenVV<TenElemT, QNT> inst_ops_set;
  for (size_t i = 0; i < inst_op_num; ++i) {
    inst_ops_set.push_back(
        std::vector<QLTensor<TenElemT, QNT>>(
            sites[i + 1] - sites[i] - 1,
            inst_ops[i]
        )
    );
  }

  return MultiSiteOpAvg(mps, phys_ops, inst_ops_set, sites);
}

/**
 * @return
 *  temp_ten :
 *
 * |----Dag(mps_ten)---
 * |     |
 * |     op
 * |     |
 * |----mps_ten---
 *
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> ContractHeadSite(const QLTensor<TenElemT, QNT> &mps_ten,
                                         const QLTensor<TenElemT, QNT> &op) {
  QLTensor<TenElemT, QNT> temp_ten, temp_ten0;

  std::vector<size_t> head_mps_ten_ctrct_axes1{1};
  std::vector<size_t> head_mps_ten_ctrct_axes2{0, 2};
  std::vector<size_t> head_mps_ten_ctrct_axes3{0, 1};

  Contract(
      &mps_ten, &op,
      {head_mps_ten_ctrct_axes1, {0}},
      &temp_ten0
  );

  auto mps_ten_dag = Dag(mps_ten);

  Contract(
      &temp_ten0, &mps_ten_dag,
      {head_mps_ten_ctrct_axes2, head_mps_ten_ctrct_axes3},
      &temp_ten
  );
  return temp_ten;
}

/**
 * @return
 *  expectation value :
 *
 *   |-------Dag(mps_ten)
 *   |         |
 * temp_ten    op
 *   |         |
 *   |-------mps_ten
 *
 */
template<typename TenElemT, typename QNT>
TenElemT ContractTailSite(
    const QLTensor<TenElemT, QNT> &mps_ten,
    const QLTensor<TenElemT, QNT> &op,
    const QLTensor<TenElemT, QNT> &temp_ten
) {
  std::vector<size_t> tail_mps_ten_ctrct_axes1{0, 1, 2};
  std::vector<size_t> tail_mps_ten_ctrct_axes2{2, 0, 1};
  QLTensor<TenElemT, QNT> temp_ten2, temp_ten3, res_ten;
  Contract(&mps_ten, &temp_ten, {{0}, {0}}, &temp_ten2);
  Contract(&temp_ten2, &op, {{0}, {0}}, &temp_ten3);
  auto mps_ten_dag = Dag(mps_ten);
  Contract(
      &temp_ten3, &mps_ten_dag,
      {tail_mps_ten_ctrct_axes1, tail_mps_ten_ctrct_axes2},
      &res_ten
  );
  TenElemT avg = res_ten();
  return avg;
}

template<typename TenElemT, typename QNT>
TenElemT OpsVecAvg(
    const FiniteMPS<TenElemT, QNT> &mps,      // Has been centralized to head_site
    const std::vector<QLTensor<TenElemT, QNT>> &ops,
    const size_t head_site,
    const size_t tail_site
) {
  auto id_op_set = mps.GetSitesInfo().id_ops;
  // Deal with head tensor.
  auto ptemp_ten = new QLTensor<TenElemT, QNT>;
  *ptemp_ten = ContractHeadSite(mps[head_site], ops[0]);

  // Deal with middle tensors.
  assert(ops.size() == (tail_site - head_site + 1));
  for (size_t i = head_site + 1; i < tail_site; ++i) {
    CtrctMidTen(mps, i, ops[i - head_site], id_op_set[i], ptemp_ten);
  }

  // Deal with tail tensor.
  auto avg = ContractTailSite(mps[tail_site], ops.back(), *ptemp_ten);
  delete ptemp_ten;
  return avg;
}

template<typename TenElemT, typename QNT>
void CtrctMidTen(
    const FiniteMPS<TenElemT, QNT> &mps,
    const size_t site,
    const QLTensor<TenElemT, QNT> &op,
    const QLTensor<TenElemT, QNT> &id_op,
    QLTensor<TenElemT, QNT> *&t) {
  using Tensor = QLTensor<TenElemT, QNT>;
  if (op == id_op) {
    Tensor temp_ten;
    Contract(&mps[site], t, {{0}, {0}}, &temp_ten);
    delete t;
    t = new Tensor;
    auto mps_ten_dag = Dag(mps[site]);
    Contract(&temp_ten, &mps_ten_dag, {{0, 2}, {1, 0}}, t);
  } else {
    Tensor temp_ten1, temp_ten2;
    Contract(&mps[site], t, {{0}, {0}}, &temp_ten1);
    delete t;
    Contract(&temp_ten1, &op, {{0}, {0}}, &temp_ten2);
    t = new Tensor;
    auto mps_ten_dag = Dag(mps[site]);
    Contract(&temp_ten2, &mps_ten_dag, {{1, 2}, {0, 1}}, t);
  }
}

// Data dump.
template<typename AvgT>
void DumpMeasuRes(
    const MeasuRes<AvgT> &res,
    const std::string &basename
) {
  auto file = basename + ".json";
  std::ofstream ofs(file);

  ofs << "[\n";

  for (auto it = res.begin(); it != res.end(); ++it) {
    auto &measu_res_elem = *it;

    ofs << "  [";

    DumpSites(ofs, measu_res_elem.sites);
    DumpAvgVal(ofs, measu_res_elem.avg);

    if (it == res.end() - 1) {
      ofs << "]\n";
    } else {
      ofs << "],\n";
    }
  }

  ofs << "]";

  ofs.close();
}
} /* qlmps */
#endif /* ifndef QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_MEASU_H */
