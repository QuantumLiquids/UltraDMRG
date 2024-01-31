// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-10 11:32
*
* Description: QuantumLiquids/UltraDMRG project. The generic matrix product state (MPS) class.
*/

/**
@file mps.h
@brief The generic matrix product state (MPS) class.
*/
#ifndef QLMPS_ONE_DIM_TN_MPS_MPS_H
#define QLMPS_ONE_DIM_TN_MPS_MPS_H


#include "qlmps/one_dim_tn/framework/ten_vec.h"    // TenVec
#include "qlmps/site_vec.h"    // SiteVec
#include "qlmps/consts.h"    // kMpsPath
#include "qlmps/utilities.h"     // IsPathExist, CreatPath

#include <vector>     // vector
#include <iomanip>    // fix, scientific, setw


namespace qlmps {
using namespace qlten;


// Helpers
inline std::string GenMPSTenName(const std::string &mps_path, const size_t idx) {
  return mps_path + "/" +
         kMpsTenBaseName + std::to_string(idx) + "." + kQLTenFileSuffix;
}


/**
The matrix product state (MPS) class.

@tparam TenElemT Element type of the local tensors.
@tparam QNT Quantum number type of the system.
*/
template <typename TenElemT, typename QNT>
class MPS : public TenVec<QLTensor<TenElemT, QNT>> {
public:
  using LocalTenT = QLTensor<TenElemT, QNT>;

  /**
  Create a empty MPS using system information.

  @param site_vec The sites information of the system.
  */
  MPS(const SiteVec<TenElemT, QNT> &site_vec) :
      TenVec<LocalTenT>(site_vec.size), site_vec_(site_vec) {}

  /**
  Get sites information.
  */
  const SiteVec<TenElemT, QNT> &GetSitesInfo(void) const {
    return site_vec_;
  }

  // HDD I/O
  /**
  Dump MPS to HDD.

  @param mps_path Path to the MPS directory.
  */
  void Dump(const std::string &mps_path = kMpsPath) const {
    if (!IsPathExist(mps_path)) { CreatPath(mps_path); }
    std::string file;
    for (size_t i = 0; i < this->size(); ++i) {
      file = GenMPSTenName(mps_path, i);
      this->DumpTen(i, file);
    }
  }

  /**
  Dump MPS to HDD.

  @param mps_path Path to the MPS directory.
  @param release_mem Whether release memory after dump.
  */
  void Dump(
      const std::string &mps_path = kMpsPath,
      const bool release_mem = false
  ) {
    if (!IsPathExist(mps_path)) { CreatPath(mps_path); }
    std::string file;
    for (size_t i = 0; i < this->size(); ++i) {
      file = GenMPSTenName(mps_path, i);
      this->DumpTen(i, file, release_mem);
    }
  }

  /**
  Load MPS from HDD.

  @param mps_path Path to the MPS directory.
  */
  void Load(const std::string &mps_path = kMpsPath) {
    std::string file;
    for (size_t i = 0; i < this->size(); ++i) {
      file = GenMPSTenName(mps_path, i);
      this->LoadTen(i, file);
    }
  }

private:
  SiteVec<TenElemT, QNT> site_vec_;
};
} /* qlmps */
#endif /* ifndef QLMPS_ONE_DIM_TN_MPS_MPS_H */
