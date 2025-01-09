// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-22
*
* Description: QuantumLiquids/UltraDMRG project. Generate the file names.
*/

#ifndef QLMPS_ALGORITHM_DMRG_OPERATOR_IO_H
#define QLMPS_ALGORITHM_DMRG_OPERATOR_IO_H

#include "qlten/qlten.h"
#include "qlmps/consts.h"                //kOpFileBaseName
#include "qlmps/algorithm/dmrg/dmrg.h"   //RightBlockOperatorGroup
#include "qlmps/utilities.h"             //WriteQLTensorTOFile

namespace qlmps {
using namespace qlten;

inline std::string GenOpFileName(
    const std::string &dir,
    const size_t blk_len,
    const size_t component,
    const std::string &temp_path
) {
  return temp_path + "/" + dir
      + kOpFileBaseName + std::to_string(blk_len)
      + "Comp" + std::to_string(component)
      + "." + kQLTenFileSuffix;
}

template<typename TenT>
void WriteOperatorGroup(
    const std::string &dir,
    const size_t blk_len,
    BlockOperatorGroup<TenT> &op_gp,
    const std::string &temp_path
) {
  for (size_t comp = 0; comp < op_gp.size(); comp++) {
    std::string file_name = GenOpFileName(dir, blk_len, comp, temp_path);
    assert(!op_gp[comp].IsDefault());
    WriteQLTensorTOFile(op_gp[comp], file_name);
  }
}

///< elements in op_gp are assumed as TenT(), with correct number of elements.
template<typename TenT>
bool ReadOperatorGroup(
    const std::string &dir,
    const size_t blk_len,
    BlockOperatorGroup<TenT> &op_gp,
    const std::string &temp_path
) {
  bool read_success(true);
  for (size_t comp = 0; comp < op_gp.size(); comp++) {
    std::string file_name = GenOpFileName(dir, blk_len, comp, temp_path);
    read_success &= ReadQLTensorFromFile(op_gp[comp], file_name);
  }
  return read_success;
}

template<typename TenT>
bool ReadAndRemoveOperatorGroup(
    const std::string &dir,
    const size_t blk_len,
    BlockOperatorGroup<TenT> &op_gp,
    const std::string &temp_path
) {
  bool read_success(true);
  for (size_t comp = 0; comp < op_gp.size(); comp++) {
    std::string file_name = GenOpFileName(dir, blk_len, comp, temp_path);
    read_success &= ReadQLTensorFromFile(op_gp[comp], file_name);
    RemoveFile(file_name);
  }
  return read_success;
}

}

#endif //QLMPS_ALGORITHM_DMRG_OPERATOR_IO_H
