// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-32
*
* Description: QuantumLiquids/UltraDMRG project. Finite MPS observation measurements with disk-stored MPS data.
*/

/**
@file finite_mps_measu_disk.h
@brief Finite MPS observation measurements with disk-stored MPS data.
*/
#ifndef QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_MEASU_DISK_H
#define QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_MEASU_DISK_H

#include "qlten/qlten.h"
#include "qlmps/one_dim_tn/mps/finite_mps/finite_mps_measu_memory.h"

namespace qlmps {

inline void PrintDiskMeasureHint() {
  std::cout << "HINT : The function MeasureOneSiteOp assume the MPS data in disk has canonical center 0!" << std::endl;
  return;
}

template<typename TenElemT, typename QNT>
void CtrctMidTen(
    FiniteMPS<TenElemT, QNT> &mps,
    const size_t site,
    const QLTensor<TenElemT, QNT> &op,
    const QLTensor<TenElemT, QNT> &id_op,
    QLTensor<TenElemT, QNT> *&t,
    const std::string &mps_path) {
  mps.LoadTen(mps_path, site);
  CtrctMidTen(mps, site, op, id_op, t);
  mps.dealloc(site);
}

/**
 * @brief Measures a single one-site operator across all sites of a finite MPS (Matrix Product State)
 * with a uniform Hilbert space.
 *
 * This function is optimized for memory efficiency. It requires the input MPS to be initialized as an
 * empty MPS, with the data stored on disk. The canonical center of the MPS is assumed to be at site 0.
 * During and after the measurement process, the data on disk remains unchanged.
 *
 *
 * @param mps The MPS to be measured. It must be initialized as an empty MPS before calling this function.
 * @param mps_path The file path to the stored MPS data on disk, which will be loaded for measurement.
 * @param op The single one-site operator to be measured.
 * @param res_file_basename The base name of the output file where the measurement results will be saved.
 *
 * @return A `MeasuRes` object containing the results of the measurement.
 */
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureOneSiteOp(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::string mps_path,
    const QLTensor<TenElemT, QNT> &op,
    const std::string &res_file_basename
) {
  PrintDiskMeasureHint();
  assert(mps.empty());
  size_t N = mps.size();
  size_t res_num = N;
  MeasuRes<TenElemT> measu_res;
  measu_res.reserve(res_num);

  mps.LoadTen(mps_path, 0);
  for (size_t site = 0; site < N; site++) {
    auto expt = OneSiteOpAvg(mps[site], op, site, N);
    measu_res.push_back(expt);
    std::cout << "measure site " << site << " exp :" << expt.avg << "\n";
    if (site < N - 1) {
      mps.LoadTen(mps_path, site + 1);
      mps.LeftCanonicalizeTen(site);
    }
    mps.dealloc(site);
  }
  DumpMeasuRes(measu_res, res_file_basename);
  return measu_res;
}

/**
 * @brief Measures one-site operators on selective sites of a finite MPS (Matrix Product State)
 *
 * This function is optimized for memory efficiency. It requires the input MPS to be initialized as an
 * empty MPS, with the data stored on disk. The canonical center of the MPS is assumed to be at site 0.
 * During and after the measurement process, the data on disk remains unchanged.
 *
 *
 * @param mps The MPS to be measured. It must be initialized as an empty MPS before calling this function.
 * @param mps_path The file path to the stored MPS data on disk, which will be loaded for measurement.
 * @param ops The set of one-site operators to be measured.
 * @param sites The selective sites to be measured.
 * @param res_file_basename The base name of the output file where the measurement results will be saved.
 *
 * @return A `MeasuRes` object containing the results of the measurement.
 */
template<typename TenElemT, typename QNT>
MeasuResSet<TenElemT> MeasureOneSiteOp(
    FiniteMPS<TenElemT, QNT> &mps,
    std::string mps_path,
    const std::vector<QLTensor<TenElemT, QNT>> &ops,
    const std::vector<size_t> &sites,//ascending order
    const std::vector<std::string> &res_file_basenames
) {
  PrintDiskMeasureHint();
  auto op_num = ops.size();
  assert(op_num == res_file_basenames.size());
  auto N = mps.size();
  size_t res_num = sites.size();
  MeasuResSet<TenElemT> measu_res_set(op_num);
  for (MeasuRes<TenElemT> &measu_res : measu_res_set) {
    measu_res.reserve(res_num);
  }

  mps.LoadTen(mps_path, 0);
  for (size_t i = 0; i < sites.size(); i++) {
    // move center
    for (size_t j = (i == 0 ? 0 : sites[i - 1]); j < sites[i]; j++) {
      mps.LoadTen(j + 1, GenMPSTenName(mps_path, j + 1));
      mps.LeftCanonicalizeTen(j);
      mps.dealloc(j);
    }
    for (size_t j = 0; j < op_num; ++j) {
      measu_res_set[j].push_back(OneSiteOpAvg(mps[sites[i]], ops[j], sites[i], N));
    }
    std::cout << "measured site " << sites[i] << std::endl;
  }
  mps.dealloc(sites.back());
  for (size_t i = 0; i < op_num; ++i) {
    DumpMeasuRes(measu_res_set[i], res_file_basenames[i]);
  }
  return measu_res_set;
}

template<typename TenElemT, typename QNT>
MeasuResSet<TenElemT> MeasureOneSiteOp(
    FiniteMPS<TenElemT, QNT> &mps,
    std::string mps_path,
    const std::vector<QLTensor<TenElemT, QNT>> &ops,
    const std::vector<std::string> &res_file_basenames
) {
  size_t N = mps.size();
  std::vector<size_t> sites(N);
  for (size_t i = 0; i < N; i++) {
    sites[i] = i;
  }
  return MeasureOneSiteOp(mps, mps_path, ops, sites, res_file_basenames);
}

/**
* @brief Measures two-site correlation function with the fixed reference site on a uniform Hilbert space.
*
* This function is optimized for memory efficiency. It requires the input MPS to be initialized as an
* empty MPS, with the data stored on disk. The canonical center of the MPS is assumed to be at site 0.
* During and after the measurement process, the data on disk remains unchanged.
*
*
* @param mps The MPS to be measured. It must be initialized as an empty MPS before calling this function.
* @param mps_path The file path to the stored MPS data on disk, which will be loaded for measurement.
* @param phys_ops1 operator on the reference site
* @param phys_ops2 operator on the target site
* @param site1 The reference site
* @param site2_set the set of target sites, ascending order with site2_set[0] > site1
* @param inst_op insertion operator.
*
* @return A `MeasuRes` object containing the results of the measurement.
*/
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureTwoSiteOpGroup(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::string mps_path,
    const QLTensor<TenElemT, QNT> &phys_ops1,
    const QLTensor<TenElemT, QNT> &phys_ops2,
    const size_t site1,
    const std::vector<size_t> &site2_set,
    const QLTensor<TenElemT, QNT> &inst_op = QLTensor<TenElemT, QNT>()
) {
  //move the center to site1
  mps.LoadTen(mps_path, 0);
  for (size_t j = 0; j < site1; j++) {
    mps.LoadTen(j + 1, GenMPSTenName(mps_path, j + 1));
    mps.LeftCanonicalizeTen(j);
    mps.dealloc(j);
  }

  //Contract mps[site1]*phys_ops1*dag(mps[site1])
  auto id_op_set = mps.GetSitesInfo().id_ops;
  auto ptemp_ten = new QLTensor<TenElemT, QNT>;
  *ptemp_ten = ContractHeadSite(mps[site1], phys_ops1);
  mps.dealloc(site1);

  size_t eated_site = site1; //the last site has been contracted
  MeasuRes<TenElemT> measure_res(site2_set.size());
  for (size_t event = 0; event < site2_set.size(); event++) {
    const size_t site2 = site2_set[event];
    while (eated_site < site2 - 1) {
      size_t eating_site = eated_site + 1;
      //Contract ptemp_ten*mps[eating_site]*dag(mps[eating_site])
      if (inst_op == QLTensor<TenElemT, QNT>()) {
        CtrctMidTen(mps, eating_site, id_op_set[eating_site], id_op_set[eating_site], ptemp_ten, mps_path);
      } else {
        CtrctMidTen(mps, eating_site, inst_op, id_op_set[eating_site], ptemp_ten, mps_path);
      }
      eated_site = eating_site;
    }

    //now site2-1 has been eaten.
    mps.LoadTen(site2, GenMPSTenName(mps_path, site2));
    //Contract ptemp_ten*mps[site2]*ops2*dag(mps[site2]) gives the expected value.
    auto avg = ContractTailSite(mps[site2], phys_ops2, *ptemp_ten);
    measure_res[event] = MeasuResElem<TenElemT>({site1, site2}, avg);
    mps.dealloc(site2);//according now code this site2 will load again in next loop. This may be optimized one day.
  }
  delete ptemp_ten;
  return measure_res;
}

/**
 * Measure the two-site correlations which specific physical operator pair,
 * the target sites are taken all the sites after the reference site.
 *
 * @return measured results
 */
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureTwoSiteOpGroup(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::string mps_path,
    const QLTensor<TenElemT, QNT> &phys_ops1,
    const QLTensor<TenElemT, QNT> &phys_ops2,
    const size_t ref_site,
    const QLTensor<TenElemT, QNT> &inst_op = QLTensor<TenElemT, QNT>()
) {
  size_t N = mps.size();
  std::vector<size_t> site2_set;
  site2_set.reserve(N);
  for (size_t site = ref_site + 1; site < N; site++) {
    site2_set.push_back(site);
  }
  return MeasureTwoSiteOpGroup(mps, mps_path, phys_ops1, phys_ops2, ref_site, site2_set, inst_op);
}

/**
 * Measure the two-site correlations which specific physical operator pair,
 * the target sites are taken all the sites after the reference site.
 * Dump the results to file based on the filename_base
 */
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureTwoSiteOpGroup(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::string mps_path,
    const QLTensor<TenElemT, QNT> &phys_ops1,
    const QLTensor<TenElemT, QNT> &phys_ops2,
    const size_t ref_site,
    const QLTensor<TenElemT, QNT> &inst_op,
    const std::string filename_base
) {
  MeasuRes<TenElemT> res = MeasureTwoSiteOpGroup(mps, mps_path, phys_ops1, phys_ops2, ref_site, inst_op);
  DumpMeasuRes(res, filename_base);
  return res;
}

///< Struct used to record the site of temp_ten
template<typename TenElemT, typename QNT>
struct LeftTempTen {
  using TenT = QLTensor<TenElemT, QNT>;
  size_t idx; // The last site been contracted
  TenT *ptemp_ten;

  LeftTempTen(const size_t &idx, const TenT &temp_ten) :
      idx(idx) {
    ptemp_ten = new TenT(temp_ten);
  }
  ~LeftTempTen() {
    delete ptemp_ten;
  }

  void MoveTo(FiniteMPS<TenElemT, QNT> &mps,
              size_t site,
              std::string mps_path) {
    assert(site >= idx);
    auto id_op_set = mps.GetSitesInfo().id_ops;
    for (size_t mid_site = idx + 1; mid_site <= site; mid_site++) {
      if (mps(mid_site) == nullptr) {// mps ten has possibly been loaded before in measuring the 4-point functions.
        mps.LoadTen(mps_path, mid_site);
      }
      CtrctMidTen(mps, mid_site, id_op_set[mid_site], id_op_set[mid_site], ptemp_ten);
      mps.dealloc(mid_site);
    }
    idx = site;
  }
};

/**
 * TODO: TEST
 * @brief Measures four-site correlation function with the fixed reference two sites on a uniform Hilbert space.
 *
 * The typical using scenario is the superconductor correlations (not on-site pair),
 * with the reference sites of the pairing-field are fixed and target site travel throughout the lattice.
 *
 * This function is optimized for memory efficiency. It requires the input MPS to be initialized as an
 * empty MPS, with the data stored on disk. The canonical center of the MPS is assumed to be at site 0.
 * During and after the measurement process, the data on disk remains unchanged.
 *
 * @param mps The MPS to be measured. It must be initialized as an empty MPS before calling this function.
 * @param mps_path The file path to the stored MPS data on disk, which will be loaded for measurement.
 * @param phys_ops
 * @param ref_sites
 * @param target_sites_set we assume the idx of target site > the idx reference site
 * @param inst_op insertion operator.
 *
 * @return A `MeasuRes` object containing the results of the measurement.
 */
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureFourSiteOpGroup(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::string &mps_path,
    const std::array<QLTensor<TenElemT, QNT>, 4> &phys_ops,
    const std::array<size_t, 2> &ref_sites,
    std::vector<std::array<size_t, 2>> target_sites_set,
    QLTensor<TenElemT, QNT> inst_op = QLTensor<TenElemT, QNT>()
) {
  assert(mps.empty());
  using Tensor = QLTensor<TenElemT, QNT>;
  std::sort(target_sites_set.begin(), target_sites_set.end(),
            [](const std::array<size_t, 2> &a, const std::array<size_t, 2> &b) {
              return a[0] < b[0];
            });
  MeasuRes<TenElemT> measure_res(target_sites_set.size());

  auto id_op_set = mps.GetSitesInfo().id_ops;
  if (inst_op == Tensor()) {
    inst_op = id_op_set[0]; // Uniform hilbert space
  }

  //Move center to ref_sites[0]
  mps.LoadTen(0, GenMPSTenName(mps_path, 0));
  for (size_t j = 0; j < ref_sites[0]; j++) {
    mps.LoadTen(j + 1, GenMPSTenName(mps_path, j + 1));
    mps.LeftCanonicalizeTen(j);
    mps.dealloc(j);
  }

  auto ptemp_ten = new QLTensor<TenElemT, QNT>;
  *ptemp_ten = ContractHeadSite(mps[ref_sites[0]], phys_ops[0]);
  mps.dealloc(ref_sites[0]);

  for (size_t i = ref_sites[0] + 1; i < ref_sites[1]; ++i) {
    CtrctMidTen(mps, i, inst_op, id_op_set[i], ptemp_ten, mps_path);
  }
  CtrctMidTen(mps, ref_sites[1], phys_ops[1], id_op_set[ref_sites[1]], ptemp_ten, mps_path);
  assert(mps.empty());

  LeftTempTen temp_struct = LeftTempTen(ref_sites[1], *ptemp_ten);
  delete ptemp_ten;
  for (size_t event = 0; event < target_sites_set.size(); event++) {
    size_t target_site0 = target_sites_set[event][0];
    temp_struct.MoveTo(mps, target_site0 - 1, mps_path);

    ptemp_ten = new Tensor(*(temp_struct.ptemp_ten));// deep copy
    mps.LoadTen(mps_path, target_site0);
    CtrctMidTen(mps, target_site0, phys_ops[2], id_op_set[target_site0], ptemp_ten);
    size_t target_site1 = target_sites_set[event][1];
    for (size_t i = target_site0 + 1; i < target_site1; ++i) {
      mps.LoadTen(mps_path, i);
      CtrctMidTen(mps, i, inst_op, id_op_set[i], ptemp_ten);
    }
    // Deal with tail tensor.
    mps.LoadTen(target_site1, GenMPSTenName(mps_path, target_site1));
    auto avg = ContractTailSite(mps[target_site1], phys_ops[3], *ptemp_ten);
    delete ptemp_ten;
    measure_res[event] = MeasuResElem<TenElemT>({ref_sites[0], ref_sites[1], target_site0, target_site1},
                                                avg);
  }
  //clean the data
  for (size_t i = target_sites_set.back()[0]; i <= target_sites_set.back()[1]; i++) {
    mps.dealloc(i);
  }
  assert(mps.empty());
  return measure_res;
}

}//qlmps
#endif /* ifndef QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_MEASU_DISK_H */
