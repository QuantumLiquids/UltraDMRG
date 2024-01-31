// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-20 17:29
* 
* Description: QuantumLiquids/UltraDMRG project. Unittests for TenVec .
*/
#include "qlmps/one_dim_tn/framework/ten_vec.h"
#include "qlten/qlten.h"

#include "gtest/gtest.h"

using namespace qlmps;
using namespace qlten;

using special_qn::U1QN;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using Tensor = DQLTensor;

struct TestTenVec : public testing::Test {
  QNT qn0;
  QNT qn1;
  QNT qnm1;
  IndexT idx_out;
  IndexT idx_in;
  Tensor ten0, ten1, ten2;

  void SetUp(void) {
    qn0 = QNT({QNCard("N", U1QNVal(0))});
    qn1 = QNT({QNCard("N", U1QNVal(1))});
    qnm1 = QNT({QNCard("N", U1QNVal(-1))});
    idx_out = IndexT(
        {QNSctT(qn0, 2), QNSctT(qn1, 2)},
        TenIndexDirType::OUT
    );
    idx_in = InverseIndex(idx_out);
    ten0 = Tensor({idx_in, idx_out});
    ten1 = Tensor({idx_in, idx_out});
    ten2 = Tensor({idx_in, idx_out});
    ten0.Random(qn0);
    ten1.Random(qn1);
    ten2.Random(qnm1);
  }
};

TEST_F(TestTenVec, TestConstructor) {
  TenVec<Tensor> tenvec(3);
  tenvec[0] = ten0;
  tenvec[1] = ten1;
  tenvec[2] = ten2;

  TenVec<Tensor> tenvec_cp(tenvec);
  EXPECT_EQ(tenvec_cp.size() , 3);
  EXPECT_EQ(tenvec_cp[0], ten0);
  EXPECT_EQ(tenvec_cp[1], ten1);
  EXPECT_EQ(tenvec_cp[2], ten2);
}

TEST_F(TestTenVec, TestIO) {
  TenVec<Tensor> tenvec(3);
  tenvec[0] = ten0;
  tenvec[1] = ten1;
  tenvec[2] = ten2;
  tenvec.DumpTen(0, "ten0." + kQLTenFileSuffix);
  tenvec.DumpTen(1, "ten1." + kQLTenFileSuffix, true);
  tenvec.DumpTen(2, "ten2." + kQLTenFileSuffix, false);
  tenvec.dealloc(0);
  tenvec.dealloc(2);
  EXPECT_TRUE(tenvec.empty());

  tenvec.LoadTen(0, "ten2." + kQLTenFileSuffix);
  tenvec.LoadTen(1, "ten0." + kQLTenFileSuffix);
  tenvec.LoadTen(2, "ten1." + kQLTenFileSuffix);
  EXPECT_EQ(tenvec[0], ten2);
  EXPECT_EQ(tenvec[1], ten0);
  EXPECT_EQ(tenvec[2], ten1);
}
