// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-12 10:19
*
* Description: GraceQ/mps2 project. Lanczos algorithm unittests.
*/

#include "gtest/gtest.h"
#include "qlmps/algorithm/vmps/lanczos_vmps_solver_impl.h"
#include "../testing_utils.h"

#ifdef Release
#define NDEBUG
#endif

using namespace qlmps;
using namespace qlten;

using special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

size_t d = 2;
size_t D = 20;
size_t dh = 2;

struct TestLanczos : public testing::Test {
  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
  IndexT idx_din = IndexT({QNSctT(qn0, d)}, TenIndexDirType::IN);
  IndexT idx_dout = InverseIndex(idx_din);
  IndexT idx_Din = IndexT({QNSctT(qn0, D)}, TenIndexDirType::IN);
  IndexT idx_Dout = InverseIndex(idx_Din);
  IndexT idx_vin = IndexT({QNSctT(qn0, dh)}, TenIndexDirType::IN);
  IndexT idx_vout = InverseIndex(idx_vin);
};

template<typename TenElemT, typename QNT>
void RunTestTwoSiteLanczosSolverCase(
    const std::vector<QLTensor<TenElemT, QNT> *
    > &eff_ham,
    QLTensor<TenElemT, QNT> *pinit_state,
    const LanczosParams &lanczos_params
) {
  using TenT = QLTensor<TenElemT, QNT>;

  Timer timer("two_site_lancz");
  auto lancz_res = LanczosSolver(
      eff_ham, pinit_state,
      &eff_ham_mul_two_site_state,
      lanczos_params
  );
  timer.
      PrintElapsed();

  std::vector<size_t> ta_ctrct_axes1 = {1};
  std::vector<size_t> ta_ctrct_axes2 = {4};
  std::vector<size_t> ta_ctrct_axes3 = {6};
  std::vector<size_t> tb_ctrct_axes1 = {0};
  std::vector<size_t> tb_ctrct_axes2 = {0};
  std::vector<size_t> tb_ctrct_axes3 = {1};
  auto eff_ham_ten = new TenT;
  Contract(eff_ham[0], eff_ham[1], {{1}, {0}}, eff_ham_ten);
  InplaceContract(eff_ham_ten, eff_ham[2], {{4}, {0}});
  InplaceContract(eff_ham_ten, eff_ham[3], {{6}, {1}});
  eff_ham_ten->Transpose({1, 2, 4, 7, 0, 3, 5, 6});

  auto dense_mat_dim = D * d * d * D;
  auto dense_mat_size = dense_mat_dim * dense_mat_dim;
  auto dense_mat = (TenElemT *) malloc(
      dense_mat_size * sizeof(TenElemT)
  );
  std::vector<std::pair<size_t, size_t>> eff_mat_coors_list;
  eff_mat_coors_list.
      reserve(dense_mat_size);
  for (size_t i = 0; i < dense_mat_dim; ++i) {
    for (size_t j = 0; j < dense_mat_dim; ++j) {
      eff_mat_coors_list.emplace_back(std::make_pair(i, j));
    }
  }
  size_t idx = 0;
  for (auto &coors : GenAllCoors(eff_ham_ten->GetShape())) {
    auto eff_mat_coors = eff_mat_coors_list[idx];
    if (eff_mat_coors.first > eff_mat_coors.second) {
      dense_mat[idx] = 0.0;
    } else {
      dense_mat[idx] = (*eff_ham_ten)(coors);
    }
    idx++;
  }

  auto w = new double[dense_mat_dim];
  LapackeSyev(
      LAPACK_ROW_MAJOR,
      'N', 'U',
      dense_mat_dim, dense_mat, dense_mat_dim, w);

  EXPECT_NEAR(lancz_res
                  .gs_eng, w[0], 1.0E-8);

  delete lancz_res.
      gs_vec;
  delete
      eff_ham_ten;
  delete[]
      w;
  free(dense_mat);
#ifndef USE_OPENBLAS
  mkl_free_buffers();
#endif
}

TEST_F(TestLanczos, TestTwoSiteLanczosSolver
) {
// Tensor with double elements.
  auto dlblock = DQLTensor({idx_Din, idx_vout, idx_Dout});
  auto dlsite = DQLTensor({idx_vin, idx_din, idx_dout, idx_vout});
  auto drblock = DQLTensor({idx_Dout, idx_vin, idx_Din});
  auto dblock_random_mat = new double[D * D];
  RandRealSymMat(dblock_random_mat, D
  );
  for (
      size_t i = 0;
      i < D;
      ++i) {
    for (
        size_t j = 0;
        j < D;
        ++j) {
      for (
          size_t k = 0;
          k < dh;
          ++k) {
        dlblock({
                    i, k, j}) = dblock_random_mat[(
            i * D
                + j)];
        drblock({
                    j, k, i}) = dblock_random_mat[(
            i * D
                + j)];
      }
    }
  }
  delete[]
      dblock_random_mat;
  auto dsite_random_mat = new double[d * d];
  RandRealSymMat(dsite_random_mat, d
  );
  for (
      size_t i = 0;
      i < d;
      ++i) {
    for (
        size_t j = 0;
        j < d;
        ++j) {
      for (
          size_t k = 0;
          k < dh;
          ++k) {
        dlsite({
                   k, i, j, k}) = dsite_random_mat[(
            i * d
                + j)];
      }
    }
  }
  delete[]
      dsite_random_mat;
  auto drsite = DQLTensor(dlsite);
  auto pdinit_state = new DQLTensor({idx_Din, idx_dout, idx_dout, idx_Dout});

// Finish iteration when Lanczos error targeted.
  srand(0);
  pdinit_state->
      Random(qn0);
  LanczosParams lanczos_params(1.0E-9);
  RunTestTwoSiteLanczosSolverCase(
      {
          &dlblock, &dlsite, &drsite, &drblock},
      pdinit_state,
      lanczos_params
  );

// Finish iteration when maximal Lanczos iteration number targeted.
  pdinit_state = new DQLTensor({idx_Din, idx_dout, idx_dout, idx_Dout});
  srand(0);
  pdinit_state->
      Random(qn0);
  LanczosParams lanczos_params2(1.0E-16, 20);
  RunTestTwoSiteLanczosSolverCase(
      {
          &dlblock, &dlsite, &drsite, &drblock},
      pdinit_state,
      lanczos_params2);

// Tensor with complex elements.
  auto zlblock = ZQLTensor({idx_Din, idx_vout, idx_Dout});
  auto zlsite = ZQLTensor({idx_vin, idx_din, idx_dout, idx_vout});
  auto zrblock = ZQLTensor({idx_Dout, idx_vin, idx_Din});
  auto zblock_random_mat = new QLTEN_Complex[D * D];
  RandCplxHerMat(zblock_random_mat, D
  );
  for (
      size_t i = 0;
      i < D;
      ++i) {
    for (
        size_t j = 0;
        j < D;
        ++j) {
      for (
          size_t k = 0;
          k < dh;
          ++k) {
        zlblock({
                    i, k, j}) = zblock_random_mat[(
            i * D
                + j)];
        zrblock({
                    j, k, i}) = zblock_random_mat[(
            i * D
                + j)];
      }
    }
  }
  delete[]
      zblock_random_mat;
  auto zsite_random_mat = new QLTEN_Complex[d * d];
  RandCplxHerMat(zsite_random_mat, d
  );
  for (
      size_t i = 0;
      i < d;
      ++i) {
    for (
        size_t j = 0;
        j < d;
        ++j) {
      for (
          size_t k = 0;
          k < dh;
          ++k) {
        zlsite({
                   k, i, j, k}) = zsite_random_mat[(
            i * d
                + j)];
      }
    }
  }
  delete[]
      zsite_random_mat;
  auto zrsite = ZQLTensor(zlsite);
  auto pzinit_state = new ZQLTensor({idx_Din, idx_dout, idx_dout, idx_Dout});

// Finish iteration when Lanczos error targeted.
  srand(0);
  pzinit_state->
      Random(qn0);
  RunTestTwoSiteLanczosSolverCase(
      {
          &zlblock, &zlsite, &zrsite, &zrblock},
      pzinit_state,
      lanczos_params
  );
}

template<typename TenElemT, typename QNT>
void RunTestSingleSiteLanczosSolverCase(
    const std::vector<QLTensor<TenElemT, QNT> *
    > &eff_ham,
    QLTensor<TenElemT, QNT> *pinit_state,
    const LanczosParams &lanczos_params
) {
  using TenT = QLTensor<TenElemT, QNT>;

  Timer timer("single_site_lancz");
  auto lancz_res = LanczosSolver(
      eff_ham, pinit_state,
      &eff_ham_mul_single_site_state,
      lanczos_params
  );
  timer.
      PrintElapsed();

  std::vector<size_t> ta_ctrct_axes1 = {1};
  std::vector<size_t> ta_ctrct_axes2 = {4};
  std::vector<size_t> tb_ctrct_axes1 = {0};
  std::vector<size_t> tb_ctrct_axes2 = {1};
  auto eff_ham_ten = new TenT;
  Contract(eff_ham[0], eff_ham[1],
           {
               {
                   1},
               {
                   0}}, eff_ham_ten);
  InplaceContract(eff_ham_ten, eff_ham[2],
                  {
                      {
                          4},
                      {
                          1}});
  eff_ham_ten->Transpose({
                             1, 2, 5, 0, 3, 4});

  auto dense_mat_dim = D * d * D;
  auto dense_mat_size = dense_mat_dim * dense_mat_dim;
  auto dense_mat = (TenElemT *) malloc(
      dense_mat_size * sizeof(TenElemT)
  );
  std::vector<std::pair<size_t, size_t>> eff_mat_coors_list;
  eff_mat_coors_list.
      reserve(dense_mat_size);
  for (
      size_t i = 0;
      i < dense_mat_dim;
      ++i) {
    for (
        size_t j = 0;
        j < dense_mat_dim;
        ++j) {
      eff_mat_coors_list.
          emplace_back(std::make_pair(i, j)
      );
    }
  }
  size_t idx = 0;
  for (
    auto &coors
      :
      GenAllCoors(eff_ham_ten
                      ->
                          GetShape()
      )) {
    auto eff_mat_coors = eff_mat_coors_list[idx];
    if (eff_mat_coors.first > eff_mat_coors.second) {
      dense_mat[idx] = 0.0;
    } else {
      dense_mat[idx] = (*eff_ham_ten)(coors);
    }
    idx++;
  }
  auto w = new double[dense_mat_dim];
  LapackeSyev(
      LAPACK_ROW_MAJOR,
      'N', 'U',
      dense_mat_dim, dense_mat, dense_mat_dim, w
  );

  EXPECT_NEAR(lancz_res
                  .gs_eng, w[0], 1.0E-8);

  delete lancz_res.
      gs_vec;
  delete
      eff_ham_ten;
  delete[]
      w;
  free(dense_mat);
#ifndef USE_OPENBLAS
  mkl_free_buffers();
#endif
}

TEST_F(TestLanczos, TestSingleSiteLanczosSolver
) {
  auto dlblock = DQLTensor({idx_Din, idx_vout, idx_Dout});
  auto dlsite = DQLTensor({idx_vin, idx_din, idx_dout, idx_vout});
  auto drblock = DQLTensor({idx_Dout, idx_vin, idx_Din});
  auto dblock_random_mat = new double[D * D];
  RandRealSymMat(dblock_random_mat, D
  );
  for (
      size_t i = 0;
      i < D;
      ++i) {
    for (
        size_t j = 0;
        j < D;
        ++j) {
      for (
          size_t k = 0;
          k < dh;
          ++k) {
        dlblock({
                    i, k, j}) = dblock_random_mat[(
            i * D
                + j)];
        drblock({
                    j, k, i}) = dblock_random_mat[(
            i * D
                + j)];
      }
    }
  }
  delete[]
      dblock_random_mat;
  auto dsite_random_mat = new double[d * d];
  RandRealSymMat(dsite_random_mat, d
  );
  for (
      size_t i = 0;
      i < d;
      ++i) {
    for (
        size_t j = 0;
        j < d;
        ++j) {
      for (
          size_t k = 0;
          k < dh;
          ++k) {
        dlsite({
                   k, i, j, k}) = dsite_random_mat[(
            i * d
                + j)];
      }
    }
  }
  delete[]
      dsite_random_mat;
  auto pdinit_state = new DQLTensor({idx_Din, idx_dout, idx_Dout});

// Finish iteration when Lanczos error targeted.
  srand(0);
  pdinit_state->
      Random(qn0);
  LanczosParams lanczos_params(1.0E-9);
  RunTestSingleSiteLanczosSolverCase(
      {
          &dlblock, &dlsite, &drblock},
      pdinit_state,
      lanczos_params
  );

// Tensor with complex element.
  auto zlblock = ZQLTensor({idx_Din, idx_vout, idx_Dout});
  auto zlsite = ZQLTensor({idx_vin, idx_din, idx_dout, idx_vout});
  auto zrblock = ZQLTensor({idx_Dout, idx_vin, idx_Din});
  auto zblock_random_mat = new QLTEN_Complex[D * D];
  RandCplxHerMat(zblock_random_mat, D
  );
  for (
      size_t i = 0;
      i < D;
      ++i) {
    for (
        size_t j = 0;
        j < D;
        ++j) {
      for (
          size_t k = 0;
          k < dh;
          ++k) {
        zlblock({
                    i, k, j}) = zblock_random_mat[(
            i * D
                + j)];
        zrblock({
                    j, k, i}) = zblock_random_mat[(
            i * D
                + j)];
      }
    }
  }
  delete[]
      zblock_random_mat;
  auto zsite_random_mat = new QLTEN_Complex[d * d];
  RandCplxHerMat(zsite_random_mat, d
  );
  for (
      size_t i = 0;
      i < d;
      ++i) {
    for (
        size_t j = 0;
        j < d;
        ++j) {
      for (
          size_t k = 0;
          k < dh;
          ++k) {
        zlsite({
                   k, i, j, k}) = zsite_random_mat[(
            i * d
                + j)];
      }
    }
  }
  delete[]
      zsite_random_mat;
  auto pzinit_state = new ZQLTensor({idx_Din, idx_dout, idx_Dout});

// Finish iteration when Lanczos error targeted.
  srand(0);
  pzinit_state->
      Random(qn0);
  RunTestSingleSiteLanczosSolverCase(
      {
          &zlblock, &zlsite, &zrblock},
      pzinit_state,
      lanczos_params
  );
}
