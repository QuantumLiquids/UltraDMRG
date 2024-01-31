// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-08 16:02
* 
* Description: QuantumLiquids/UltraDMRG project. Utility functions used by QuantumLiquids/UltraDMRG.
*/
#ifndef QLMPS_UTILITIES_H
#define QLMPS_UTILITIES_H


#include "qlten/qlten.h"    // bfread, bfwrite

#include <iostream>
#include <fstream>          // ifstream, ofstream

#include <sys/stat.h>       // stat, mkdir, S_IRWXU, S_IRWXG, S_IROTH, S_IXOTH


namespace qlmps {
using namespace qlten;

inline double Real(const QLTEN_Double d) { return d; }

inline double Real(const QLTEN_Complex z) { return z.real(); }

/// The features in this namespace will be natively supported by GraceQ/tensor.
namespace mock_qlten {


template <typename TenElemT, typename QNT>
void SVD(
    const QLTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    QLTensor<TenElemT, QNT> *pu,
    QLTensor<QLTEN_Double, QNT> *ps,
    QLTensor<TenElemT, QNT> *pvt
) {
  auto t_shape = pt->GetShape();
  size_t lsize = 1;
  size_t rsize = 1;
  for (size_t i = 0; i < pt->Rank(); ++i) {
    if (i < ldims) {
      lsize *= t_shape[i];
    } else {
      rsize *= t_shape[i];
    }
  }
  auto D = ((lsize >= rsize) ? lsize : rsize);
  QLTEN_Double actual_trunc_err;
  size_t actual_bond_dim;
  SVD(
      pt,
      ldims, lqndiv, 0, D, D,
      pu, ps, pvt, &actual_trunc_err, &actual_bond_dim
  );
}
} /* mock_qlten */


template <typename TenT>
inline void WriteQLTensorTOFile(const TenT &t, const std::string &file) {
  std::ofstream ofs(file, std::ofstream::binary);
  ofs << t;
  ofs.close();
}


template <typename TenT>
inline bool ReadQLTensorFromFile(TenT &t, const std::string &file) {
  std::ifstream ifs(file, std::ifstream::binary);
  if(ifs.good()){
    ifs >> t;
    ifs.close();
    return true;
  } else {
    return false;
  }
}


inline bool IsPathExist(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}


inline void CreatPath(const std::string &path) {
  const int dir_err = mkdir(
                          path.c_str(),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH
                      );
  if (dir_err == -1) {
    std::cout << "error creating directory!" << std::endl;
    exit(1);
  }
}
} /* qlmps */
#endif /* ifndef QLMPS_UTILITIES_H */
