// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2022/5/11
*
* Description: QuantumLiquids/UltraDMRG project. Finite MPO Class.
*/

/**
@file finite_mpo.h
@brief Finite MPO Class.
*/

#ifndef QLMPS_ONE_DIM_TN_MPO_FINITE_MPO_FINITE_MPO_H
#define QLMPS_ONE_DIM_TN_MPO_FINITE_MPO_FINITE_MPO_H

#include "qlten/qlten.h"
#include "qlmps/one_dim_tn/mps_all.h"

namespace qlmps {
using namespace qlten;

const size_t kMaxVariationSweeps = 10;
const double kVariationConvergeTolerance = 1e-13;

using MPOTenCanoType = MPSTenCanoType;

//forward declaration
template<typename TenElemT, typename QNT>
class FiniteMPO;

// MPO variation optimize params
struct MpoVOptimizeParams {
  MpoVOptimizeParams(
      const std::string &initial_mpo_path,
      const size_t Dmin, const size_t Dmax,
      const double trunc_err,
      const size_t sweeps = kMaxVariationSweeps,
      const double converge_tolerance = kVariationConvergeTolerance,
      const std::string &temp_path = kRuntimeTempPath
  ) :
      Dmin(Dmin), Dmax(Dmax), trunc_err(trunc_err),
      sweeps(sweeps), converge_tolerance(converge_tolerance),
      initial_mpo_path(initial_mpo_path),
      temp_path(temp_path) {}

  size_t Dmin;
  size_t Dmax;
  double trunc_err;

  size_t sweeps;
  double converge_tolerance;

  std::string initial_mpo_path;
  std::string temp_path;
};

template<typename TenElemT, typename QNT>
void MpoProduct(
    const FiniteMPO<TenElemT, QNT> &mpo1,
    const FiniteMPO<TenElemT, QNT> &mpo2,
    FiniteMPO<TenElemT, QNT> &output_mpo,
    const size_t Dmin,
    const size_t Dmax, // the bond dimension
    const double trunc_err, // truncation error when svd decomposition
    const size_t sweep_time_max,   // max sweep time when variational sweep
    const double sweep_converge_tolerance,
    const std::string temp_path,
    const bool output_info = false
);

inline std::string GenMPOTenName(const std::string &mpo_path, const size_t idx) {
  return mpo_path + "/" +
      kMpoTenBaseName + std::to_string(idx) + "." + kQLTenFileSuffix;
}

template<typename TenElemT, typename QNT>
class FiniteMPO : public TenVec<QLTensor<TenElemT, QNT>> {
 public:
  using LocalTenT = QLTensor<TenElemT, QNT>;

  FiniteMPO(const size_t size) : TenVec<LocalTenT>(size),
                                 center_(kUncentralizedCenterIdx),
                                 tens_cano_type_(size) {}

  FiniteMPO(const MPO<LocalTenT> &mpo) : TenVec<LocalTenT>(mpo), center_(kUncentralizedCenterIdx),
                                         tens_cano_type_(mpo.size()) {}

  operator MPO<LocalTenT>() {
    return MPO<LocalTenT>(*this);
  }

  FiniteMPO(const FiniteMPO &other) = default;

  FiniteMPO &operator=(const FiniteMPO &) = default;

  FiniteMPO &operator=(FiniteMPO &&rhs) {
    TenVec<QLTensor<TenElemT, QNT>>::operator=(std::move(rhs));
    center_ = rhs.center_;
    tens_cano_type_ = std::move(rhs.tens_cano_type_);
    rhs.center_ = -1;
    rhs.tens_cano_type_ = std::vector<MPOTenCanoType>(tens_cano_type_.size(), MPOTenCanoType::NONE);
    return *this;
  }

  /**
   * Access to local tensor
   */
  LocalTenT &operator[](const size_t idx) {
    tens_cano_type_[idx] = MPSTenCanoType::NONE;
    center_ = kUncentralizedCenterIdx;
    return DuoVector<LocalTenT>::operator[](idx);
  }

  /**
   * Read-only access to local tensor.
   */
  const LocalTenT &operator[](const size_t idx) const {
    return DuoVector<LocalTenT>::operator[](idx);
  }

  /**
   * Access to the pointer to local tensor.
   */
  LocalTenT *&operator()(const size_t idx) {
    tens_cano_type_[idx] = MPOTenCanoType::NONE;
    center_ = kUncentralizedCenterIdx;
    return DuoVector<LocalTenT>::operator()(idx);
  }

  /**
   * Read-only access to the pointer to local tensor.
   * @param idx
   * @return
   */
  const LocalTenT *operator()(const size_t idx) const {
    return DuoVector<LocalTenT>::operator()(idx);
  }

  FiniteMPO &operator*=(const TenElemT s) {
    this->Scale(s);
    return (*this);
  }

  FiniteMPO &operator+=(const FiniteMPO &rhs) {
    const size_t N = this->size();
    assert(rhs.size() == N);

    for (size_t i = 0; i < N; i++) {
      LocalTenT *ptemp = new LocalTenT();
      assert((*this)(i) != nullptr);
      if (i == 0) {
        Expand((*this)(i), rhs(i), {3}, ptemp);
      } else if (i == N - 1) {
        Expand((*this)(i), rhs(i), {0}, ptemp);
      } else {
        Expand((*this)(i), rhs(i), {0, 3}, ptemp);
      }
      delete (*this)(i);
      (*this)(i) = ptemp;
//      std::cout << "(summation) site = " << i << std::endl;
    }
    return (*this);
  }

  FiniteMPO operator+(const FiniteMPO &rhs) const {
    auto res = *this;
    res += rhs;
    return res;
  }

  ///<  *this * rhs,  *this is at below, represent the front matrix
  FiniteMPO SimpleProduct(const FiniteMPO &rhs) const {

  }

  /**
   * square the MPO without normalization
   *
   * @param optimize_params
   * @return
   */
  void Square(const MpoVOptimizeParams &optimize_params) {
    const size_t N = this->size();

    FiniteMPO result(N);
    result.Load(optimize_params.initial_mpo_path);
    MpoProduct(*this, *this, result,
               optimize_params.Dmin, optimize_params.Dmax, optimize_params.trunc_err,
               optimize_params.sweeps, optimize_params.converge_tolerance,
               optimize_params.temp_path);
    size_t result_center = result.center_;
    for (size_t i = 0; i < N; i++) {
      assert((*this)(i) != nullptr);
      delete (*this)(i);
      (*this)(i) = result(i);
      result(i) = nullptr;
    }
    center_ = result_center;
    assert(center_ != kUncentralizedCenterIdx);
    SetCenter_();
    return;
  }

  double SquareAndNormlize(const MpoVOptimizeParams &optimize_params) {
    double norm2 = this->Square(optimize_params);
    assert(center_ != kUncentralizedCenterIdx);
    this->Scale(1.0 / norm2);
    return norm2;
  }

  void Centralize(const int);

  void LeftCanonicalize(const size_t);

  void RightCanonicalize(const size_t);

  void LeftCanonicalizeTen(const size_t);

  void RightCanonicalizeTen(const size_t);

  void Truncate(const QLTEN_Double, const size_t, const size_t);

  void Dump(const std::string &mpo_path = kMpoPath) const {
    if (!IsPathExist(mpo_path)) { CreatPath(mpo_path); }
    std::string file;
    for (size_t i = 0; i < this->size(); ++i) {
      file = GenMPOTenName(mpo_path, i);
      this->DumpTen(i, file);
    }

    file = mpo_path + "/center";
    std::ofstream ofs(file, std::ofstream::binary);
    ofs << center_;
    ofs.close();
  }

  bool Load(const std::string &mpo_path = kMpoPath) {
    if (!IsPathExist(mpo_path)) {
      return false;
//      std::cout << "DONOT FIND THE PATH " << mpo_path <<". Please check the mpo data directory" << std::endl;
    }
    std::string file;
    for (size_t i = 0; i < this->size(); ++i) {
      file = GenMPOTenName(mpo_path, i);
      this->LoadTen(i, file);
    }

    file = mpo_path + "/center";
    std::ifstream ifs(file, std::ifstream::binary);
    ifs >> center_;
    ifs.close();
    SetCenter_();
    return true;
  }

  void Scale(const TenElemT s) {
    if (center_ != kUncentralizedCenterIdx) {
      const size_t old_center = center_;
      (*this)[old_center] *= s;
      center_ = old_center;
    } else {
      (*this)[0] *= s;
    }
  }

  double Normalize() {
    double norm2;
    if (center_ != kUncentralizedCenterIdx) {
      const size_t old_center = center_;
      norm2 = (*this)(old_center)->Normalize();
      center_ = old_center;
    } else {
      norm2 = (*this)(0)->Normalize();
    }
    return norm2;
  }

  TenElemT Trace();

  size_t GetMaxBondDim(void) const {
    size_t D = 1;
    for (size_t i = 0; i < (*this).size() - 1; i++) {
      D = std::max(D, (*this)[i].GetShape()[3]);
    }
    return D;
  }

  int GetCenter(void) const { return center_; }

  std::vector<MPSTenCanoType> GetTensCanoType(void) const {
    return tens_cano_type_;
  }

  MPSTenCanoType GetTenCanoType(const size_t idx) const {
    return tens_cano_type_[idx];
  }

 private:
  //? SiteVec<TenElemT, QNT> site_vec_;
  int center_;
  std::vector<MPOTenCanoType> tens_cano_type_;

  /// with truncate version
  void RightCanonicalizeTen_(const size_t, const QLTEN_Double,
                             const size_t, const size_t,
                             QLTEN_Double &, size_t &);

  void SetCenter_() {
    if (center_ != kUncentralizedCenterIdx) {
      for (size_t i = 0; i < center_; i++) {
        tens_cano_type_[i] = MPOTenCanoType::LEFT;
      }
      tens_cano_type_[center_] = MPOTenCanoType::NONE;
      for (size_t i = center_ + 1; i < this->size(); i++) {
        tens_cano_type_[i] = MPOTenCanoType::RIGHT;
      }
    }
  }

  template<typename TenElemT2, typename QNT2>
  friend void MpoProduct(
      const FiniteMPO<TenElemT2, QNT2> &mpo1,
      const FiniteMPO<TenElemT2, QNT2> &mpo2,
      FiniteMPO<TenElemT2, QNT2> &output_mpo,
      const size_t Dmin,
      const size_t Dmax, // the bond dimension
      const double trunc_err, // truncation error when svd decomposition
      const size_t sweep_time_max,   // max sweep time when variational sweep
      const double sweep_converge_tolerance,
      const std::string temp_path,
      const bool
  );
};

template<typename TenElemT, typename QNT>
void FiniteMPO<TenElemT, QNT>::Centralize(const int target_center) {
  assert(target_center >= 0);
  auto mpo_tail_idx = this->size() - 1;
  if (target_center != 0) { LeftCanonicalize(target_center - 1); }
  if (target_center != mpo_tail_idx) {
    RightCanonicalize(target_center + 1);
  }
  center_ = target_center;
}

template<typename TenElemT, typename QNT>
void FiniteMPO<TenElemT, QNT>::LeftCanonicalize(const size_t stop_idx) {
  size_t start_idx;
  for (size_t i = 0; i <= stop_idx; ++i) {
    start_idx = i;
    if (tens_cano_type_[i] != MPOTenCanoType::LEFT) { break; }
    if (i == stop_idx) { return; }    // All related tensors are left canonical, do nothing.
  }
  for (size_t i = start_idx; i <= stop_idx; ++i) { LeftCanonicalizeTen(i); }
}

template<typename TenElemT, typename QNT>
void FiniteMPO<TenElemT, QNT>::LeftCanonicalizeTen(const size_t site_idx) {
  assert(site_idx < this->size() - 1);
  size_t ldims(3);
  auto pq = new LocalTenT;
  LocalTenT r;
  QR((*this)(site_idx), ldims, Div((*this)[site_idx]), pq, &r);
  delete (*this)(site_idx);
  (*this)(site_idx) = pq;

  auto pnext_ten = new LocalTenT;
  Contract(&r, (*this)(site_idx + 1), {{1},
                                       {0}}, pnext_ten);
  delete (*this)(site_idx + 1);
  (*this)(site_idx + 1) = pnext_ten;

  tens_cano_type_[site_idx] = MPSTenCanoType::LEFT;
  tens_cano_type_[site_idx + 1] = MPSTenCanoType::NONE;
}

template<typename TenElemT, typename QNT>
void FiniteMPO<TenElemT, QNT>::RightCanonicalize(const size_t stop_idx) {
  auto mpo_tail_idx = this->size() - 1;
  size_t start_idx;
  for (size_t i = mpo_tail_idx; i >= stop_idx; --i) {
    start_idx = i;
    if (tens_cano_type_[i] != MPSTenCanoType::RIGHT) { break; }
    if (i == stop_idx) { return; }    // All related tensors are right canonical, do nothing.
  }
  for (size_t i = start_idx; i >= stop_idx; --i) { RightCanonicalizeTen(i); }
}

template<typename TenElemT, typename QNT>
void FiniteMPO<TenElemT, QNT>::RightCanonicalizeTen(const size_t site_idx) {
  ///< TODO: using LU decomposition
  assert(site_idx > 0);
  size_t ldims = 1;
  LocalTenT u;
  QLTensor<QLTEN_Double, QNT> s;
  auto pvt = new LocalTenT;
  auto qndiv = Div((*this)[site_idx]);
  mock_qlten::SVD((*this)(site_idx), ldims, qndiv - qndiv, &u, &s, pvt);
  delete (*this)(site_idx);
  (*this)(site_idx) = pvt;

  LocalTenT temp_ten;
  Contract(&u, &s, {{1},
                    {0}}, &temp_ten);
  std::vector<std::vector<size_t>> ctrct_axes = {{3},
                                                 {0}};
  auto pprev_ten = new LocalTenT;
  Contract((*this)(site_idx - 1), &temp_ten, ctrct_axes, pprev_ten);
  delete (*this)(site_idx - 1);
  (*this)(site_idx - 1) = pprev_ten;

  tens_cano_type_[site_idx] = MPSTenCanoType::RIGHT;
  tens_cano_type_[site_idx - 1] = MPSTenCanoType::NONE;
}

template<typename TenElemT, typename QNT>
void FiniteMPO<TenElemT, QNT>::Truncate(const QLTEN_Double trunc_err,
                                        const size_t Dmin,
                                        const size_t Dmax) {
  auto N = this->size();
  assert(N >= 2);
  this->Centralize(N - 1);

  QLTEN_Double actual_trunc_err;
  size_t D;

  for (size_t i = N - 1; i > 0; i--) {
    this->RightCanonicalizeTen_(i, trunc_err, Dmin, Dmax, actual_trunc_err, D);
    std::cout << "Truncate FiniteMPO bond " << std::setw(4) << i
              << ", TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
              << ", D = " << std::setw(5) << D
              << std::endl;
  }
  return;
}

/** the trace of the operator
 *  todo: (not important) using trace(summation) to replace contract with identity one day.
 * @tparam TenElemT
 * @tparam QNT
 */
template<typename TenElemT, typename QNT>
TenElemT FiniteMPO<TenElemT, QNT>::Trace() {
  LocalTenT left_vector_ten;
  Index<QNT> left_trivial_index = (*this)[0].GetIndexes()[0];
  left_vector_ten = LocalTenT({InverseIndex(left_trivial_index)});
  left_vector_ten({0}) = 1.0;
//  LocalTenT right_vector_ten;
//  Index<QNT> right_trivial_index = (*this).back().GetIndexes()[3];
//  right_vector_ten = LocalTenT({InverseIndex( right_trivial_index });

  for (size_t i = 0; i < (*this).size(); i++) {
    LocalTenT temp_ten;
    const Index<QNT> &idx_out = (*this)[i].GetIndexes()[2];
    LocalTenT id = GenIdOp<TenElemT, QNT>(idx_out);//assume non-uniform lattice site
    Contract(&left_vector_ten, (*this)(i), {{0},
                                            {0}}, &temp_ten);
    left_vector_ten = LocalTenT();
    Contract(&temp_ten, &id, {{0, 1},
                              {1, 0}}, &left_vector_ten);
  }
  TenElemT t = left_vector_ten({0});
  return t;
}

/** Truncate when right canonicalize the tensor on site `site_idx`.
 *
 *
 * @param site_idx
 * @param trunc_err
 * @param Dmin
 * @param Dmax
 * @param actrual_trunc_err  output, the actrual truncation error.
 * @param D                  output, the bond dimension after canonicalizing.
 */
template<typename TenElemT, typename QNT>
void FiniteMPO<TenElemT, QNT>::RightCanonicalizeTen_(const size_t site_idx,
                                                     const QLTEN_Double trunc_err,
                                                     const size_t Dmin,
                                                     const size_t Dmax,
                                                     QLTEN_Double &actrual_trunc_err,
                                                     size_t &D) {
  assert(site_idx > 0);
  size_t ldims = 1;
  LocalTenT u;
  QLTensor<QLTEN_Double, QNT> s;
  auto pvt = new LocalTenT;
  auto qndiv = Div((*this)[site_idx]);
  qlten::SVD((*this)(site_idx), ldims, qndiv - qndiv, trunc_err, Dmin, Dmax, &u, &s, pvt, &actrual_trunc_err, &D);
  delete (*this)(site_idx);
  (*this)(site_idx) = pvt;

  LocalTenT temp_ten;
  Contract(&u, &s, {{1},
                    {0}}, &temp_ten);
  std::vector<std::vector<size_t>> ctrct_axes = {{3},
                                                 {0}};
  auto pprev_ten = new LocalTenT;
  Contract((*this)(site_idx - 1), &temp_ten, ctrct_axes, pprev_ten);
  delete (*this)(site_idx - 1);
  (*this)(site_idx - 1) = pprev_ten;

  tens_cano_type_[site_idx] = MPSTenCanoType::RIGHT;
  tens_cano_type_[site_idx - 1] = MPSTenCanoType::NONE;
}

}

#endif //QLMPS_ONE_DIM_TN_MPO_FINITE_MPO_FINITE_MPO_H
