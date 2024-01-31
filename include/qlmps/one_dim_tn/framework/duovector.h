// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-19 16:40
*
* Description: QuantumLiquids/UltraDMRG project. A fix size vector which supports maintaining
*              elements using a reference (the memory managed by this class) or
*              a pointer (the memory managed by user themselves).
*/

/**
@file duovector.h
@brief A fix size vector supporting elements maintaining by reference or pointer.
*/
#ifndef QLMPS_ONE_DIM_TN_FRAMEWORK_DUOVECTOR_H
#define QLMPS_ONE_DIM_TN_FRAMEWORK_DUOVECTOR_H


#include <vector>     // vector
#include <utility>    // move
#include <cstddef>    // size_t


namespace qlmps {


/**
A fix size vector supporting elements maintaining by reference or pointer.

@tparam ElemT Type of the elements.
*/
template <typename ElemT>
class DuoVector {
public:
  /**
  Default constructor.
  */
  DuoVector(void) = default;

  /**
  Create a DuoVector using its size.

  @param size The size of the DuoVector.
  */
  DuoVector(const size_t size) : raw_data_(size, nullptr) {}

  /**
  Create a DuoVector by copying another DuoVector.

  @param duovec A DuoVector instance.
  */
  DuoVector(const DuoVector<ElemT> &duovec) : raw_data_(duovec.size(), nullptr) {
    for (size_t i = 0; i < duovec.size(); ++i) {
      if (duovec(i) != nullptr) {
        raw_data_[i] = new ElemT(duovec[i]);
      }
    }
  }

  /**
  Copy a DuoVector.

  @param rhs A DuoVector instance.
  */
  DuoVector<ElemT> &operator=(const DuoVector<ElemT> &rhs) {
//    this->DuoVector::~DuoVector();
//    I'm no sure why above line do not work if used in the derived FiniteMPS class
    for (auto &rpelem : raw_data_) {
      if (rpelem != nullptr) {
        delete rpelem;
      }
    }
    const size_t N = rhs.size();
    raw_data_ = std::vector<ElemT *>(N, nullptr);
    for (size_t i = 0; i < N; ++i) {
      if (rhs(i) != nullptr) {
        raw_data_[i] = new ElemT(rhs[i]);
      }
    }
    return *this;
  }

  /**
  Create a DuoVector by moving raw data from another DuoVector instance.

  @param duovec A DuoVector instance.
  */
  DuoVector(
      DuoVector<ElemT> &&duovec
  ) noexcept : raw_data_(std::move(duovec.raw_data_)) {
    duovec.raw_data_ = std::vector<ElemT *>(duovec.size(), nullptr);
  }

  /**
  Move a DuoVector.

  @param rhs A DuoVector instance.
  */
  DuoVector<ElemT> &operator=(DuoVector<ElemT> &&rhs) noexcept {
//    this->DuoVector::~DuoVector();
//    I have no idea why above line do not work if used in the derived FiniteMPS class
    for (auto &rpelem : raw_data_) {
      if (rpelem != nullptr) {
        delete rpelem;
      }
    }
    raw_data_ = std::move(rhs.raw_data_);
    rhs.raw_data_ = std::vector<ElemT *>(rhs.size(), nullptr);
    return *this;
  }

  /**
  Destruct a DuoVector. Release memory it maintained.
  */
  virtual ~DuoVector(void) {
    for (auto &rpelem : raw_data_) {
      if (rpelem != nullptr) {
        delete rpelem;
      }
    }
  }

  // Data access methods.
  /**
  Element getter.

  @param idx The index of the element.
  */
  const ElemT &operator[](const size_t idx) const { return *raw_data_[idx]; }

  /**
  Element setter. If the corresponding memory has not been allocated, allocate
  it first.

  @param idx The index of the element.
  */
  ElemT &operator[](const size_t idx) {
    if (raw_data_[idx] == nullptr) {
      raw_data_[idx] = new ElemT;
    }
    return *raw_data_[idx];
  }

  /**
  Pointer-of-element getter.

  @param idx The index of the element.
  */
  const ElemT *operator()(const size_t idx) const {return raw_data_[idx]; }

  /**
  Pointer-of-element setter.

  @param idx The index of the element.
  */
  ElemT * &operator()(const size_t idx) { return raw_data_[idx]; }

  /**
  Access the first element.
  */
  ElemT &front(void) { return *raw_data_.front(); }

  /// @copydoc DuoVector::front()
  const ElemT &front(void) const { return *raw_data_.front(); }

  /**
  Access the last element.
  */
  ElemT &back(void) { return *raw_data_.back(); }

  /// @copydoc DuoVector::back()
  const ElemT &back(void) const { return *raw_data_.back(); }

  /**
  Read-only raw data access.
  */
  const std::vector<const ElemT *> cdata(void) const {
    std::vector<const ElemT *> craw_data;
    for (auto &rpelem : raw_data_) {
      craw_data.push_back(rpelem);
    }
    return craw_data;
  }

  // Memory management methods
  /**
  Allocate memory of the element at given index. If the given place has a
  non-nullptr, release the memory which point to first.

  @param idx The index of the element.
  */
  void alloc(const size_t idx) {
    if (raw_data_[idx] != nullptr) {
      delete raw_data_[idx];
    }
    raw_data_[idx] = new ElemT;
  }

  /**
  Deallocate memory of the element at given index.

  @param idx The index of the element.
  */
  void dealloc(const size_t idx) {
    if (raw_data_[idx] != nullptr) {
      delete raw_data_[idx];
      raw_data_[idx] = nullptr;
    }
  }

  /**
  Deallocate all elements.
  */
  void clear(void) {
    for (size_t i = 0; i < size(); ++i) { dealloc(i); }
  }

  // Property methods
  /**
  Get the size of the DuoVector.
  */
  size_t size(void) const { return raw_data_.size(); }

  /**
  Check whether the vector is empty.
  */
  bool empty(void) const {
    for (auto &pelem : raw_data_) {
      if (pelem != nullptr) {
        return false;
      }
    }
    return true;
  }

private:
  std::vector<ElemT *> raw_data_;
};
} /* qlmps */
#endif /* ifndef QLMPS_ONE_DIM_TN_FRAMEWORK_DUOVECTOR_H */
