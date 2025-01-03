// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-31
*
* Description: QuantumLiquids/UltraDMRG project. initialisation for two-site update finite size vMPS.
*/

/**
 @file   two_site_update_finite_vmps_init.h
 @brief  Initilization for two-site update finite size vMPS.
         0. include an overall initial function cover all the functions in this file;
         1. Find the left/right boundaries, only between which the tensors need to be update.
            Also make sure the bond dimensions of tensors out of boundaries are sufficient large.
            Move the centre on the left_boundary+1 site (Assuming the before the centre <= left_boundary+1)
         2. Check if .temp exsits, if exsits, check if temp tensors are complete.
            if one of above if is not, regenerate the environment.
         3. Generate the environment of boundary tensors
*/

#ifndef QLMPS_ALGORITHM_VMPS_VMPS_INIT_H
#define QLMPS_ALGORITHM_VMPS_VMPS_INIT_H

#include <map>
#include "qlten/qlten.h"
#include "qlmps/one_dim_tn/mps_all.h"                          //FiniteMPS
#include "qlmps/algorithm/finite_vmps_sweep_params.h"  //FiniteVMPSSweepParams

namespace qlmps {
using namespace qlten;

//forward declaration
template<typename TenElemT, typename QNT>
std::pair<size_t, size_t> CheckAndUpdateBoundaryMPSTensors(
    FiniteMPS<TenElemT, QNT> &,
    const std::string &,
    const size_t
);

template<typename TenElemT, typename QNT>
void UpdateBoundaryEnvs(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const std::string mps_path,
    const std::string temp_path,
    const size_t left_boundary,
    const size_t right_boundary,
    const size_t update_site_num = 2 //e.g., two site update or single site update
);

inline bool NeedGenerateRightEnvs(
    const size_t N, //mps size
    const size_t left_boundary,
    const size_t right_boundary,
    const std::string &temp_path
);

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> UpdateSiteRenvs(
    const QLTensor<TenElemT, QNT> &,
    const QLTensor<TenElemT, QNT> &,
    const QLTensor<TenElemT, QNT> &
);

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> UpdateSiteLenvs(
    const QLTensor<TenElemT, QNT> &,
    const QLTensor<TenElemT, QNT> &,
    const QLTensor<TenElemT, QNT> &
);

template<typename TenElemT, typename QNT>
std::pair<size_t, size_t> FiniteVMPSInit(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  std::cout << "\n";
  std::cout << "=====> Sweep Parameters <=====" << "\n";
  std::cout << "MPS/MPO size: \t " << mpo.size() << "\n";
  std::cout << "Sweep times: \t " << sweep_params.sweeps << "\n";
  std::cout << "Bond dimension: \t " << sweep_params.Dmin << "/" << sweep_params.Dmax << "\n";
  std::cout << "Truncation error: \t " << sweep_params.trunc_err << "\n";
  std::cout << "Lanczos max iterations \t" << sweep_params.lancz_params.max_iterations << "\n";
  std::cout << "MPS path: \t" << sweep_params.mps_path << "\n";
  std::cout << "Temp path: \t" << sweep_params.temp_path << std::endl;

  std::cout << "=====> Technical Parameters <=====" << "\n";
#ifndef USE_GPU
  std::cout << "The number of threads: \t" << hp_numeric::GetTensorManipulationThreads() << "\n";
#endif
  std::cout << "=====> Checking and updating boundary tensors =====>" << std::endl;
  auto [left_boundary, right_boundary] = CheckAndUpdateBoundaryMPSTensors(
      mps,
      sweep_params.mps_path,
      sweep_params.Dmax
  );

  if (NeedGenerateRightEnvs(
      mpo.size(),
      left_boundary,
      right_boundary,
      sweep_params.temp_path)
      ) {
    std::cout << "=====> Creating the environment tensors =====>" << std::endl;
    InitEnvs(mps, mpo, sweep_params.mps_path, sweep_params.temp_path, left_boundary + 2);
  } else {
    std::cout << "The environment tensors exist." << std::endl;
  }

  //update the left env of left_boundary site and right env of right_boundary site
  UpdateBoundaryEnvs(mps, mpo, sweep_params.mps_path,
                     sweep_params.temp_path, left_boundary, right_boundary, 2);
  return std::make_pair(left_boundary, right_boundary);
}

/** CheckAndUpdateBoundaryMPSTensors
 *
 * This function makes sure the bond dimension
 * of tensors near ends are sufficiently large. If the bond dimension is not sufficient,
 * the tensor will replaced by combiner, and one more contract to make sure the mps is
 * not changed. Left/right canonical condition of tensors on each sides
 * are also promised in this procedure, so that the later vmps only need doing between
 * left and right boundary.
 * The first tensors that need to be truncate gives the left boundary and
 * right boundary.
 *
 * Thus a design is for compatibility with other vmps function's results. (2021-08-27)
 *
 * @return left_boundary    the most left site needs to update in VMPS
 * @return right_boundary   the most right site needs to update in VMPS
 * @note we suppose the centre of mps <= left_boundary+1 before call this function,
 *       and the centre will be moved to left_boundary+1 when return;
 *
 * TODO: QR decomposition replaces SVD
*/
template<typename TenElemT, typename QNT>
std::pair<size_t, size_t> CheckAndUpdateBoundaryMPSTensors(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::string &mps_path,
    const size_t Dmax
) {
  assert(mps.empty());
  //TODO: check if central file, add this function to the friend of FiniteMPS
  using TenT = QLTensor<TenElemT, QNT>;

  using std::cout;
  using std::endl;
  size_t N = mps.size();
  size_t left_boundary(0);  //the most left site which needs to update.
  size_t right_boundary(0); //the most right site which needs to update

  size_t left_middle_site, right_middle_site;
  if (N % 2 == 0) {
    left_middle_site = N / 2 - 1;
    right_middle_site = N / 2;
    //make sure at least four sites are used to sweep
  } else {
    left_middle_site = N / 2;
    right_middle_site = N / 2;
    //make sure at least three sites are used to sweep
  }

  //Assume the central of MPS at zero

  //Left Side
  mps.LoadTen(0, GenMPSTenName(mps_path, 0));
  for (size_t i = 0; i < left_middle_site; i++) {
    mps.LoadTen(i + 1, GenMPSTenName(mps_path, i + 1));
    mps.LeftCanonicalizeTen(i);

    TenT &mps_ten = mps[i];
    ShapeT mps_ten_shape = mps_ten.GetShape();
    if (mps_ten_shape[0] * mps_ten_shape[1] > Dmax) {
      left_boundary = i;
      break;
    } else if (mps_ten_shape[0] * mps_ten_shape[1] > mps_ten_shape[2]) {
      TenIndexDirType new_dir = mps_ten.GetIndexes()[2].GetDir();
      Index<QNT> index_0 = mps_ten.GetIndexes()[0];
      Index<QNT> index_1 = mps_ten.GetIndexes()[1];

      TenT index_combiner_for_fuse = IndexCombine<TenElemT, QNT>(
          InverseIndex(index_0),
          InverseIndex(index_1),
          IN
      );
      TenT ten_tmp;
      Contract(&index_combiner_for_fuse, &mps_ten, {{0, 1}, {0, 1}}, &ten_tmp);
      TenT mps_next_tmp;
      Contract(&ten_tmp, mps(i + 1), {{1}, {0}}, &mps_next_tmp);
      mps[i + 1] = std::move(mps_next_tmp);

      mps[i] = std::move(IndexCombine<TenElemT, QNT>(
          index_0,
          index_1,
          new_dir
      ));
      assert(ten_tmp.GetIndexes()[0] == InverseIndex(mps[i].GetIndexes()[2]));
    }
    if (i == left_middle_site - 1) {
      left_boundary = i;
    }
  }

  for (size_t i = 0; i <= left_boundary + 1; i++) {
    mps.DumpTen(i, GenMPSTenName(mps_path, i), true);
  }

  //Right Side
  mps.LoadTen(N - 1, GenMPSTenName(mps_path, N - 1));
  for (size_t i = N - 1; i > right_middle_site; i--) {
    mps.LoadTen(i - 1, GenMPSTenName(mps_path, i - 1));
    mps.RightCanonicalizeTen(i);

    TenT &mps_ten = mps[i];
    ShapeT mps_ten_shape = mps_ten.GetShape();
    if (mps_ten_shape[1] * mps_ten_shape[2] > Dmax) {
      right_boundary = i;
      break;
    } else if (mps_ten_shape[1] * mps_ten_shape[2] > mps_ten_shape[0]) {
      TenT index_combiner = IndexCombine<TenElemT, QNT>(
          mps[i].GetIndexes()[1],
          mps[i].GetIndexes()[2],
          mps[i].GetIndexes()[0].GetDir()
      );
      index_combiner.Transpose({2, 0, 1});
      mps[i].FuseIndex(1, 2);
      assert(mps[i].GetIndexes()[0] == InverseIndex(index_combiner.GetIndexes()[0]));
      InplaceContract(mps(i - 1), mps(i), {{2}, {1}});
      mps[i] = std::move(index_combiner);
    }

    if (i == right_middle_site + 1) {
      right_boundary = i;
    }
  }
  for (size_t i = N - 1; i >= right_boundary - 1; i--) {
    mps.DumpTen(i, GenMPSTenName(mps_path, i), true);
  }

  assert(mps.empty());
  return std::make_pair(left_boundary, right_boundary);
}

/**
  If need to generate right environment tensors,
  checked by if right environment tensors are enough.
  If no temp_path, it will be generate by the way.
*/
inline bool NeedGenerateRightEnvs(
    const size_t N, //mps size
    const size_t left_boundary,
    const size_t right_boundary,
    const std::string &temp_path
) {
  if (IsPathExist(temp_path)) {
    for (size_t env_num = (N - 1) - right_boundary; env_num <= (N - 1) - (left_boundary + 1); env_num++) {
      std::string file = GenEnvTenName("r", env_num, temp_path);
      if (access(file.c_str(), 4) != 0) {
        std::cout << "Lost file" << file << "." << "\n";
        return true;
      }
    }
    return false;
  } else {
    std::cout << "No temp path " << temp_path << "\n";
    CreatPath(temp_path);
    return true;
  }
}

/** UpdateBoundaryEnvs
 *  regenerate and rewrite environment tensors including:
 *    - left env of site left_boundary
 *    - right env of site right_boundary
 *    - right env of site right_boundary-1
 *
*/
template<typename TenElemT, typename QNT>
void UpdateBoundaryEnvs(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const std::string mps_path,
    const std::string temp_path,
    const size_t left_boundary,
    const size_t right_boundary,
    const size_t update_site_num //e.g., two site update or single site update
) {
  assert(mps.empty());

  using TenT = QLTensor<TenElemT, QNT>;
  auto N = mps.size();

  //Write a trivial right environment tensor to disk
  mps.LoadTen(N - 1, GenMPSTenName(mps_path, N - 1));
  auto mps_trivial_index = mps.back().GetIndexes()[2];
  auto mpo_trivial_index_inv = InverseIndex(mpo.back().GetIndexes()[3]);
  auto mps_trivial_index_inv = InverseIndex(mps_trivial_index);
  TenT renv = TenT({mps_trivial_index, mpo_trivial_index_inv, mps_trivial_index_inv});
  renv({0, 0, 0}) = 1;
  mps.dealloc(N - 1);

  //bulk right environment tensors
  for (size_t i = 1; i <= N - right_boundary - 1; ++i) {
    mps.LoadTen(N - i, GenMPSTenName(mps_path, N - i));
    renv = std::move(UpdateSiteRenvs(renv, mps[N - i], mpo[N - i])); //question: if it's efficient?
    mps.dealloc(N - i);
  }
  std::string file = GenEnvTenName("r", N - right_boundary - 1, temp_path);
  WriteQLTensorTOFile(renv, file);

  //right env of site right_boundary-1
  mps.LoadTen(right_boundary, GenMPSTenName(mps_path, right_boundary));
  renv = std::move(UpdateSiteRenvs(renv, mps[right_boundary], mpo[right_boundary]));
  mps.dealloc(right_boundary);
  file = GenEnvTenName("r", N - right_boundary, temp_path);
  WriteQLTensorTOFile(renv, file);



  //Write a trivial left environment tensor to disk
  mps.LoadTen(0, GenMPSTenName(mps_path, 0));
  mps_trivial_index = mps.front().GetIndexes()[0];
  mpo_trivial_index_inv = InverseIndex(mpo.front().GetIndexes()[0]);
  mps_trivial_index_inv = InverseIndex(mps_trivial_index);
  TenT lenv = TenT({mps_trivial_index, mpo_trivial_index_inv, mps_trivial_index_inv});
  lenv({0, 0, 0}) = 1;
  mps.dealloc(0);
  std::cout << "left boundary = " << left_boundary << std::endl;
  for (size_t i = 0; i < left_boundary; ++i) {
    mps.LoadTen(i, GenMPSTenName(mps_path, i));
    lenv = std::move(UpdateSiteLenvs(lenv, mps[i], mpo[i]));
    mps.dealloc(i);
  }
  file = GenEnvTenName("l", left_boundary, temp_path);
  WriteQLTensorTOFile(lenv, file);
  assert(mps.empty());
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> UpdateSiteRenvs(
    const QLTensor<TenElemT, QNT> &renv,
    const QLTensor<TenElemT, QNT> &mps_ten,
    const QLTensor<TenElemT, QNT> &mpo_ten
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT mps_ten_dag = Dag(mps_ten);
  TenT renv_next, temp_ten, temp_ten2;
  Contract<TenElemT, QNT, true, true>(mps_ten_dag, renv, 2, 0, 1, temp_ten);
  Contract<TenElemT, QNT, true, false>(temp_ten, mpo_ten, 1, 2, 2, temp_ten2);
  temp_ten.GetBlkSparDataTen().Clear();
  Contract<TenElemT, QNT, true, false>(temp_ten2, mps_ten, 3, 1, 2, renv_next);
  return renv_next;
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> UpdateSiteLenvs(
    const QLTensor<TenElemT, QNT> &lenv,
    const QLTensor<TenElemT, QNT> &mps_ten,
    const QLTensor<TenElemT, QNT> &mpo_ten
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT mps_ten_dag = Dag(mps_ten);
  TenT lenv_next, temp_ten, temp_ten2;
  Contract<TenElemT, QNT, true, true>(lenv, mps_ten, 2, 0, 1, temp_ten);
  Contract<TenElemT, QNT, true, true>(temp_ten, mpo_ten, 1, 0, 2, temp_ten2);
  temp_ten.GetBlkSparDataTen().Clear();
  Contract<TenElemT, QNT, false, true>(mps_ten_dag, temp_ten2, 0, 1, 2, lenv_next);
  return lenv_next;
}

}//qlmps
#endif
