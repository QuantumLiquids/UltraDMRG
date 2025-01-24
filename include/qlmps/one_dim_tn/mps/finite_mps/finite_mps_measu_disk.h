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

/**
* @brief Measures two-site correlation function with the fix reference site on a uniform Hilbert space.
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
  //Contract on site1
  std::vector<size_t> head_mps_ten_ctrct_axes1{1};
  std::vector<size_t> head_mps_ten_ctrct_axes2{0, 2};
  std::vector<size_t> head_mps_ten_ctrct_axes3{0, 1};
  QLTensor<TenElemT, QNT> temp_ten0;
  auto ptemp_ten = new QLTensor<TenElemT, QNT>;//TODO: delete
  Contract(
      &mps[site1], &phys_ops1,
      {{1}, {0}},
      &temp_ten0
  );
  QLTensor<TenElemT, QNT> mps_ten_dag = Dag(mps[site1]);
  Contract(
      &temp_ten0, &mps_ten_dag,
      {head_mps_ten_ctrct_axes2, head_mps_ten_ctrct_axes3},
      ptemp_ten
  );
  mps_ten_dag.GetBlkSparDataTen().Clear();//Save memory
  mps.dealloc(site1);

  size_t eated_site = site1; //the last site has been contracted
  MeasuRes<TenElemT> measure_res(site2_set.size());
  for (size_t event = 0; event < site2_set.size(); event++) {
    const size_t site2 = site2_set[event];
    while (eated_site < site2 - 1) {
      size_t eating_site = eated_site + 1;
      mps.LoadTen(eating_site, GenMPSTenName(mps_path, eating_site));
      //Contract ptemp_ten*mps[eating_site]*dag(mps[eating_site])
      if (inst_op == QLTensor<TenElemT, QNT>()) {
        CtrctMidTen(mps, eating_site, id_op_set[eating_site], id_op_set[eating_site], ptemp_ten);
      } else {
        CtrctMidTen(mps, eating_site, inst_op, id_op_set[eating_site], ptemp_ten);
      }
      eated_site = eating_site;
      mps.dealloc(eated_site);
    }

    //now site2-1 has been eaten.
    mps.LoadTen(site2, GenMPSTenName(mps_path, site2));
    //Contract ptemp_ten*mps[site2]*ops2*dag(mps[site2]) gives the expected value.
    std::vector<size_t> tail_mps_ten_ctrct_axes1{0, 1, 2};
    std::vector<size_t> tail_mps_ten_ctrct_axes2{2, 0, 1};
    QLTensor<TenElemT, QNT> temp_ten2, temp_ten3, res_ten;
    Contract(&mps[site2], ptemp_ten, {{0}, {0}}, &temp_ten2);
    Contract(&temp_ten2, &phys_ops2, {{0}, {0}}, &temp_ten3);
    mps_ten_dag = Dag(mps[site2]);
    Contract(
        &temp_ten3, &mps_ten_dag,
        {tail_mps_ten_ctrct_axes1, tail_mps_ten_ctrct_axes2},
        &res_ten
    );
    measure_res[event] = MeasuResElem<TenElemT>({site1, site2}, res_ten());

    mps.dealloc(site2);//according now code this site2 will load again in next loop. This may be optimized one day.
  }
  delete ptemp_ten;
  return measure_res;
}

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

}//qlmps
#endif /* ifndef QLMPS_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_MEASU_DISK_H */
