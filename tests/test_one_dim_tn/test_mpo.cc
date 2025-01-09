// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-06 16:40
*
* Description: QuantumLiquids/UltraDMRG project. Unittests for MPO .
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/one_dim_tn/mpo/mpo.h"

using namespace qlmps;
using namespace qlten;

using U1QN = QN<U1QNVal>;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using Tensor = DQLTensor;

struct TestMPO : public testing::Test {
  QNT qn = QNT({QNCard("N", U1QNVal(0))});
  IndexT idx_out = IndexT({QNSctT(qn, 3)}, OUT);
  IndexT idx_in = IndexT({QNSctT(qn, 4)}, IN);
  Tensor ten1 = Tensor({idx_out});
  Tensor ten2 = Tensor({idx_in, idx_out});
  Tensor ten3 = Tensor({idx_in, idx_out, idx_out});

  void SetUp(void) {
    ten1.Random(qn);
    ten2.Random(qn);
    ten3.Random(qn);
  }
};

template<typename TenT>
void RunTestMPOConstructor1Case(const int N) {
  MPO<TenT> mpo(N);
  EXPECT_EQ(mpo.size(), N);
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(mpo[i], TenT());
  }
}

TEST_F(TestMPO, TestConstructors) {
  RunTestMPOConstructor1Case<Tensor>(0);
  RunTestMPOConstructor1Case<Tensor>(5);
}

TEST_F(TestMPO, TestElemAccess) {
  MPO<Tensor> mpo(3);
  mpo[0] = ten1;
  EXPECT_EQ(mpo[0], ten1);
  mpo[1] = ten2;
  EXPECT_EQ(mpo[1], ten2);
  mpo[2] = ten3;
  EXPECT_EQ(mpo[2], ten3);

  mpo[1] = ten3;
  EXPECT_EQ(mpo[1], ten3);
}
