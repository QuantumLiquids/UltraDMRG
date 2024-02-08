// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-14 10:26
*
* Description: QuantumLiquids/mps project. Unittests for MPO generation, focus on the most generic AddTerm API.
*/

#include "gtest/gtest.h"
#include "qlmps/qlmps.h"

using namespace qlmps;
using namespace qlten;

using U1QN = QN<U1QNVal>;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

using DSiteVec = SiteVec<QLTEN_Double, QNT>;
using ZSiteVec = SiteVec<QLTEN_Complex, QNT>;
using DMPOGenerator = MPOGenerator<QLTEN_Double, QNT>;
using ZMPOGenerator = MPOGenerator<QLTEN_Complex, QNT>;

const BasisOpLabel kIdOpLabel = 0;                 // Coefficient label for identity id.
const OpRepr kIdOpRepr = OpRepr(kIdOpLabel);  // Operator representation for identity operator.

struct TestMpoGenerator : public testing::Test {
  IndexT phys_idx_out = IndexT({
                                   QNSctT(QNT({QNCard("Sz", U1QNVal(-1))}), 1),
                                   QNSctT(QNT({QNCard("Sz", U1QNVal(1))}), 1)},
                               TenIndexDirType::OUT
  );
  IndexT phys_idx_in = InverseIndex(phys_idx_out);
  DSiteVec dsite_vec_2 = DSiteVec(2, phys_idx_out);
  DSiteVec dsite_vec_3 = DSiteVec(3, phys_idx_out);
  DSiteVec dsite_vec_4 = DSiteVec(4, phys_idx_out);
  DSiteVec dsite_vec_5 = DSiteVec(5, phys_idx_out);
  ZSiteVec zsite_vec_2 = ZSiteVec(2, phys_idx_out);
  ZSiteVec zsite_vec_3 = ZSiteVec(3, phys_idx_out);
  ZSiteVec zsite_vec_4 = ZSiteVec(4, phys_idx_out);
  ZSiteVec zsite_vec_5 = ZSiteVec(5, phys_idx_out);
  DQLTensor dsz = DQLTensor({phys_idx_in, phys_idx_out});
  ZQLTensor zsz = ZQLTensor({phys_idx_in, phys_idx_out});
  ZQLTensor zsx = ZQLTensor({phys_idx_in, phys_idx_out});
  ZQLTensor zsy = ZQLTensor({phys_idx_in, phys_idx_out});
  DQLTensor did = DQLTensor({phys_idx_in, phys_idx_out});
  ZQLTensor zid = ZQLTensor({phys_idx_in, phys_idx_out});
  QNT qn0 = QNT({QNCard("Sz", U1QNVal(0))});

  void SetUp(void) {
    dsz({0, 0}) = -0.5;
    dsz({1, 1}) = 0.5;
    zsz({0, 0}) = -0.5;
    zsz({1, 1}) = 0.5;
    did({0, 0}) = 1;
    did({1, 1}) = 1;
    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsx({0, 1}) = 1;
    zsx({1, 0}) = 1;
    zsy({0, 1}) = QLTEN_Complex(0, -1);
    zsy({1, 0}) = QLTEN_Complex(0, 1);
  }
};

TEST_F(TestMpoGenerator, TestInitialization) {
  DMPOGenerator mpo_generator1(dsite_vec_2);
  DMPOGenerator mpo_generator2(dsite_vec_2, qn0);
}

TEST_F(TestMpoGenerator, TestAddTermCase1) {
  SparOpReprMat bchmk_m0(1, 1), bchmk_m1(1, 1);
  bchmk_m0.SetElem(0, 0, OpRepr(1));
  bchmk_m1.SetElem(0, 0, kIdOpRepr);

  DMPOGenerator mpo_generator1(dsite_vec_2, qn0);
  mpo_generator1.AddTerm(1., {-dsz}, {0});
  FSM fsm1 = mpo_generator1.GetFSM();
  SparOpReprMatVec fsm_comp_mat_repr1 = fsm1.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr1[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr1[1], bchmk_m1);

  DMPOGenerator mpo_generator3(dsite_vec_2, qn0);
  mpo_generator3.AddTerm(1., {-dsz}, {0});
  mpo_generator3.AddTerm(0.0, {-dsz}, {0});
  auto fsm3 = mpo_generator3.GetFSM();
  auto fsm_comp_mat_repr3 = fsm3.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr3[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr3[1], bchmk_m1);
}

TEST_F(TestMpoGenerator, TestAddTermCase2) {
  auto s = OpRepr(1);
  SparOpReprMat bchmk_m0(1, 1), bchmk_m1(1, 1);
  bchmk_m0.SetElem(0, 0, s);
  bchmk_m1.SetElem(0, 0, s + s);

  DMPOGenerator mpo_generator1(dsite_vec_2, qn0);
  mpo_generator1.AddTerm(1., {dsz, dsz}, {0, 1});
  mpo_generator1.AddTerm(1., {dsz, dsz}, {0, 1});
  auto fsm1 = mpo_generator1.GetFSM();
  auto fsm_comp_mat_repr1 = fsm1.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr1[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr1[1], bchmk_m1);
}

TEST_F(TestMpoGenerator, TestAddTermCase3) {
  auto s = OpRepr(1);
  SparOpReprMat bchmk_m0(1, 2), bchmk_m1(2, 3), bchmk_m2(3, 2), bchmk_m3(2, 1);
  bchmk_m0.SetElem(0, 0, s);
  bchmk_m0.SetElem(0, 1, kIdOpRepr);
  bchmk_m1.SetElem(0, 2, s);
  bchmk_m1.SetElem(1, 0, kIdOpRepr);
  bchmk_m1.SetElem(1, 1, s);
  bchmk_m2.SetElem(0, 0, s);
  bchmk_m2.SetElem(1, 1, s);
  bchmk_m2.SetElem(2, 1, kIdOpRepr);
  bchmk_m3.SetElem(0, 0, s);
  bchmk_m3.SetElem(1, 0, kIdOpRepr);

  DMPOGenerator mpo_generator1(dsite_vec_4, qn0);
  mpo_generator1.AddTerm(1., {dsz, dsz}, {0, 1});
  mpo_generator1.AddTerm(1., {dsz, dsz}, {1, 2});
  mpo_generator1.AddTerm(1., {dsz, dsz}, {2, 3});
  auto fsm1 = mpo_generator1.GetFSM();
  auto fsm_comp_mat_repr1 = fsm1.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr1[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr1[1], bchmk_m1);
  EXPECT_EQ(fsm_comp_mat_repr1[2], bchmk_m2);
  EXPECT_EQ(fsm_comp_mat_repr1[3], bchmk_m3);
}

TEST_F(TestMpoGenerator, TestAddTermCase4) {
  auto s = OpRepr(1);
  SparOpReprMat bchmk_m0(1, 2), bchmk_m1(2, 2), bchmk_m2(2, 2), bchmk_m3(2, 2),
      bchmk_m4(2, 1);
  bchmk_m0.SetElem(0, 0, kIdOpRepr);
  bchmk_m0.SetElem(0, 1, s);
  bchmk_m1.SetElem(0, 1, s);
  bchmk_m1.SetElem(1, 0, kIdOpRepr);
  bchmk_m2.SetElem(0, 0, kIdOpRepr);
  bchmk_m2.SetElem(1, 1, s);
  bchmk_m3.SetElem(0, 1, kIdOpRepr);
  bchmk_m3.SetElem(1, 0, s);
  bchmk_m3.SetElem(1, 1, kIdOpRepr);
  bchmk_m4.SetElem(0, 0, kIdOpRepr);
  bchmk_m4.SetElem(0, 1, s);

  DMPOGenerator mpo_generator1(dsite_vec_5, qn0);
  mpo_generator1.AddTerm(1., {dsz, dsz}, {0, 4});
  mpo_generator1.AddTerm(1., {-dsz, -dsz, -dsz}, {1, 2, 4});
  mpo_generator1.AddTerm(1., {-dsz, -dsz, -dsz}, {1, 2, 3});
  auto fsm1 = mpo_generator1.GetFSM();
  auto fsm_comp_mat_repr1 = fsm1.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr1[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr1[1], bchmk_m1);
  EXPECT_EQ(fsm_comp_mat_repr1[2], bchmk_m2);
  EXPECT_EQ(fsm_comp_mat_repr1[3], bchmk_m3);
  EXPECT_EQ(fsm_comp_mat_repr1[4], bchmk_m4);
}

TEST_F(TestMpoGenerator, TestAddTermCase5) {
  CNumberLabel j1 = 1, j2 = 2;
  BasisOpLabel s = 1;
  OpRepr op_s(s);
  SparOpReprMat bchmk_m0(1, 2), bchmk_m1(2, 4), bchmk_m2(4, 2), bchmk_m3(2, 1);
  bchmk_m0.SetElem(0, 0, kIdOpRepr);
  bchmk_m0.SetElem(0, 1, op_s);
  bchmk_m1.SetElem(0, 0, OpRepr(j1, kIdOpLabel));
  bchmk_m1.SetElem(0, 3, op_s);
  bchmk_m1.SetElem(1, 1, OpRepr(j1, s));
  bchmk_m1.SetElem(1, 2, kIdOpRepr);
  bchmk_m2.SetElem(0, 0, op_s);
  bchmk_m2.SetElem(1, 1, kIdOpRepr);
  bchmk_m2.SetElem(2, 0, OpRepr(j2, kIdOpLabel));
  bchmk_m2.SetElem(2, 1, OpRepr(j1, s));
  bchmk_m2.SetElem(3, 0, OpRepr(j1, kIdOpLabel));
  bchmk_m2.SetElem(3, 1, OpRepr(j2, s));
  bchmk_m3.SetElem(0, 0, op_s);
  bchmk_m3.SetElem(1, 0, kIdOpRepr);

  QLTEN_Double ja = 0.5, jb = 2.0;

  DMPOGenerator mpo_generator1(dsite_vec_4, qn0);
  mpo_generator1.AddTerm(ja, {dsz, dsz}, {0, 1});
  mpo_generator1.AddTerm(ja, {dsz, dsz}, {0, 2});
  mpo_generator1.AddTerm(ja, {dsz, dsz}, {1, 3});
  mpo_generator1.AddTerm(ja, {dsz, dsz}, {2, 3});
  mpo_generator1.AddTerm(jb, {dsz, dsz}, {0, 3});
  mpo_generator1.AddTerm(jb, {dsz, dsz}, {1, 2});
  auto fsm1 = mpo_generator1.GetFSM();
  auto fsm_comp_mat_repr1 = fsm1.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr1[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr1[1], bchmk_m1);
  EXPECT_EQ(fsm_comp_mat_repr1[2], bchmk_m2);
  EXPECT_EQ(fsm_comp_mat_repr1[3], bchmk_m3);
}

TEST_F(TestMpoGenerator, TestAddTermCase6) {
  ZMPOGenerator mpo_generator(zsite_vec_3, qn0);
  QLTEN_Complex J = 0.5, K = 2.0;
  mpo_generator.AddTerm(J, {zsx, zsx}, {0, 1});
  mpo_generator.AddTerm(J, {zsy, zsy}, {0, 1});
  mpo_generator.AddTerm(J, {zsz, zsz}, {0, 1});
  mpo_generator.AddTerm(J, {zsx, zsx}, {1, 2});
  mpo_generator.AddTerm(J, {zsy, zsy}, {1, 2});
  mpo_generator.AddTerm(J, {zsz, zsz}, {1, 2});
  mpo_generator.AddTerm(K, {zsx, zsx}, {0, 1});
  mpo_generator.AddTerm(K, {zsz, zsz}, {1, 2});
  auto fsm = mpo_generator.GetFSM();

  CNumberLabel j = 1, k = 2;
  BasisOpLabel sx = 1, sy = 2, sz = 3;
  SparOpReprMat bchmk_m0(1, 4), bchmk_m1(4, 4), bchmk_m2(4, 1);
  bchmk_m0.SetElem(0, 0, OpRepr({k, j}, {sx, sx}));
  bchmk_m0.SetElem(0, 1, OpRepr(j, sy));
  bchmk_m0.SetElem(0, 2, OpRepr(j, sz));
  bchmk_m0.SetElem(0, 3, kIdOpRepr);
  bchmk_m1.SetElem(0, 3, OpRepr(sx));
  bchmk_m1.SetElem(1, 3, OpRepr(sy));
  bchmk_m1.SetElem(2, 3, OpRepr(sz));
  bchmk_m1.SetElem(3, 0, OpRepr(j, sx));
  bchmk_m1.SetElem(3, 1, OpRepr(j, sy));
  bchmk_m1.SetElem(3, 2, OpRepr({j, k}, {sz, sz}));
  bchmk_m2.SetElem(0, 0, OpRepr(sx));
  bchmk_m2.SetElem(1, 0, OpRepr(sy));
  bchmk_m2.SetElem(2, 0, OpRepr(sz));
  bchmk_m2.SetElem(3, 0, kIdOpRepr);
  auto fsm_comp_mat_repr = fsm.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr[1], bchmk_m1);
  EXPECT_EQ(fsm_comp_mat_repr[2], bchmk_m2);
}
