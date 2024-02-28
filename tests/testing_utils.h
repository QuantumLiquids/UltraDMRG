// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-12 11:10
* 
* Description: GraceQ/mps2 project. Testing utilities.
*/
#ifndef QLMPS_TESTING_UTILS_H
#define QLMPS_TESTING_UTILS_H

#include "qlten/qlten.h"

#include <iostream>
#include <vector>
#include <cstdlib>

#include "gtest/gtest.h"
#ifndef USE_OPENBLAS
#include "mkl.h"
#else
#include "cblas.h"
#endif

using namespace qlten;

inline double Rand(void) { return double(rand()) / RAND_MAX; }

inline void RandRealSymMat(double *mat, long dim) {
  srand(0);
  for (long i = 0; i < dim; ++i) {
    for (long j = 0; j < dim; ++j) {
      mat[(i * dim + j)] = Rand();
    }
  }
  srand(0);
  for (long i = 0; i < dim; ++i) {
    for (long j = 0; j < dim; ++j) {
      mat[(j * dim + i)] += Rand();
    }
  }
}

inline void RandCplxHerMat(QLTEN_Complex *mat, long dim) {
  for (long i = 0; i < dim; ++i) {
    for (long j = 0; j < i; ++j) {
      QLTEN_Complex elem(Rand(), Rand());
      mat[(i * dim + j)] = elem;
      mat[(j * dim + i)] = std::conj(elem);
    }
  }
  for (long i = 0; i < dim; ++i) {
    mat[i * dim + i] = Rand();
  }
}

inline void LapackeSyev(
    int matrix_layout, char jobz, char uplo,
    size_t n, double *a, size_t lda, double *w) {
  LAPACKE_dsyev(matrix_layout, jobz, uplo, n, a, lda, w);
}

inline void LapackeSyev(
    int matrix_layout, char jobz, char uplo,
    size_t n, QLTEN_Complex *a, size_t lda, double *w) {
  LAPACKE_zheev(matrix_layout, jobz, uplo, n,
                reinterpret_cast<lapack_complex_double *>(a), lda, w);
}

inline void EXPECT_COMPLEX_EQ(
    const QLTEN_Complex &lhs,
    const QLTEN_Complex &rhs) {
  EXPECT_DOUBLE_EQ(lhs.real(), rhs.real());
  EXPECT_DOUBLE_EQ(lhs.imag(), rhs.imag());
}

inline void RemoveFolder(const std::string &folder_path) {
  std::string command = "rm -rf " + folder_path;
  system(command.c_str());
}

// Helpers
inline void KeepOrder(size_t &x, size_t &y) {
  if (x > y) {
    auto temp = y;
    y = x;
    x = temp;
  }
}

inline size_t coors2idx(
    const size_t x, const size_t y, const size_t Nx, const size_t Ny) {
  return x * Ny + y;
}

inline size_t coors2idxSquare(
    const int x, const int y, const size_t Nx, const size_t Ny) {
  return x * Ny + y;
}

inline size_t coors2idxHoneycomb(
    const int x, const int y, const size_t Nx, const size_t Ny) {
  return Ny * (x % Nx) + y % Ny;
}

#endif /* ifndef QLMPS_TESTING_UTILS_H */
