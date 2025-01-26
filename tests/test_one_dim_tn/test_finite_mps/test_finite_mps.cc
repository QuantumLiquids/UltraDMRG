// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-21 09:13
*
* Description: QuantumLiquids/UltraDMRG project. Unittests for MPS .
*/

#include <utility>    // move
#include "gtest/gtest.h"

#include "qlten/qlten.h"
#include "qlmps/one_dim_tn/mpo/mpo.h"                              // MPO
#include "qlmps/one_dim_tn/mpo/mpogen/mpogen.h"                    // MPOGenerator
#include "qlmps/algorithm/vmps/vmps_all.h"
#include "../../testing_utils.h"                                       //RemoveFolder

using namespace qlmps;
using namespace qlten;

using U1QN = special_qn::U1QN;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;
using Tensor = DQLTensor;

using SiteVecT = SiteVec<QLTEN_Double, U1QN>;
using MPST = FiniteMPS<QLTEN_Double, U1QN>;

struct TestMPS : public testing::Test {
  QNT qn0 = QNT({QNCard("N", U1QNVal(0))});
  QNT qn1 = QNT({QNCard("N", U1QNVal(1))});
  QNT qn2 = QNT({QNCard("N", U1QNVal(2))});
  IndexT vb0_in = IndexT({QNSctT(qn0, 1)}, IN);
  IndexT pb_out = IndexT({QNSctT(qn0, 1), QNSctT(qn1, 1)}, OUT);
  IndexT vb01_out = IndexT({QNSctT(qn0, 1), QNSctT(qn1, 1)}, OUT);
  IndexT vb01_in = InverseIndex(vb01_out);
  IndexT vb012_out = IndexT(
      {QNSctT(qn0, 1), QNSctT(qn1, 2), QNSctT(qn2, 1)},
      OUT
  );
  IndexT vb012_in = InverseIndex(vb012_out);
  IndexT vb0_out = InverseIndex(vb0_in);
  Tensor t0 = Tensor({vb0_in, pb_out, vb01_out});
  Tensor t1 = Tensor({vb01_in, pb_out, vb012_out});
  Tensor t2 = Tensor({vb012_in, pb_out, vb012_out});
  Tensor t3 = Tensor({vb012_in, pb_out, vb01_out});
  Tensor t4 = Tensor({vb01_in, pb_out, vb0_out});

  SiteVecT site_vec = SiteVecT(5, pb_out);

  MPST mps = MPST(site_vec);

  void SetUp(void) {
    t0.Random(qn1);
    t1.Random(qn1);
    t2.Random(qn0);
    t3.Random(qn0);
    t4.Random(qn0);
    mps[0] = t0;
    mps[1] = t1;
    mps[2] = t2;
    mps[3] = t3;
    mps[4] = t4;
  }
};

// Helpers for testing MPS centralization.
template<typename TenT>
void CheckIsIdTen(const TenT &t) {
  auto shape = t.GetShape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], shape[1]);
  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      if (i == j) {
        EXPECT_NEAR(t(i, j), 1.0, 1.2E-15);
      } else {
        EXPECT_NEAR(t(i, j), 0.0, 1E-15);
      }
    }
  }
}

void CheckMPSTenCanonical(
    const MPST &mps,
    const size_t i,
    const int center
) {
  std::vector<std::vector<size_t>> ctrct_leg_idxs;
  if (i < center) {
    ctrct_leg_idxs = {{0, 1}, {0, 1}};
  } else if (i > center) {
    ctrct_leg_idxs = {{1, 2}, {1, 2}};
  }

  Tensor res;
  auto ten = mps[i];
  auto ten_dag = Dag(ten);
  Contract(&ten, &ten_dag, ctrct_leg_idxs, &res);
  CheckIsIdTen(res);
}

void CheckMPSCenter(const MPST &mps, const int center) {
  EXPECT_EQ(mps.GetCenter(), center);

  auto mps_size = mps.size();
  auto tens_cano_type = mps.GetTensCanoType();
  for (size_t i = 0; i < mps_size; ++i) {
    if (i < center) {
      EXPECT_EQ(tens_cano_type[i], MPSTenCanoType::LEFT);
      CheckMPSTenCanonical(mps, i, center);
    }
    if (i > center) {
      EXPECT_EQ(tens_cano_type[i], MPSTenCanoType::RIGHT);
      CheckMPSTenCanonical(mps, i, center);
    }
    if (i == center) {
      EXPECT_EQ(tens_cano_type[i], MPSTenCanoType::NONE);
    }
  }
}

void RunTestMPSCentralizeCase(MPST &mps, const int center) {
  mps.Centralize(center);
  CheckMPSCenter(mps, center);
}

void RunTestMPSCentralizeCase(MPST &mps) {
  for (size_t i = 0; i < mps.size(); ++i) {
    RunTestMPSCentralizeCase(mps, i);
  }
}

TEST_F(TestMPS, TestCentralize) {
  RunTestMPSCentralizeCase(mps, 0);
  RunTestMPSCentralizeCase(mps, 2);
  RunTestMPSCentralizeCase(mps, 4);
  RunTestMPSCentralizeCase(mps);

  mps[0].Random(qn0);
  RunTestMPSCentralizeCase(mps, 0);
  RunTestMPSCentralizeCase(mps, 2);
  RunTestMPSCentralizeCase(mps, 4);
  RunTestMPSCentralizeCase(mps);

  mps[1].Random(qn0);
  mps[2].Random(qn0);
  mps[4].Random(qn0);
  RunTestMPSCentralizeCase(mps, 0);
  RunTestMPSCentralizeCase(mps, 2);
  RunTestMPSCentralizeCase(mps, 4);
  RunTestMPSCentralizeCase(mps);
}

TEST_F(TestMPS, TestCopyAndMove) {
  mps.Centralize(2);
  const MPST &crmps = mps;

  MPST mps_copy(mps);
  const MPST &crmps_copy = mps_copy;
  EXPECT_EQ(mps_copy.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_copy.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_copy[i], crmps[i]);
    EXPECT_NE(crmps_copy(i), crmps(i));
  }

  MPST mps_copy2(mps.GetSitesInfo());
  mps_copy2 = mps;
  const MPST &crmps_copy2 = mps_copy2;
  EXPECT_EQ(mps_copy2.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_copy2.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_copy2[i], crmps[i]);
    EXPECT_NE(crmps_copy2(i), crmps(i));
  }

  auto craw_data_copy = mps_copy.cdata();
  MPST mps_move(std::move(mps_copy));
  const MPST &crmps_move = mps_move;
  EXPECT_EQ(mps_move.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_move.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_move[i], crmps[i]);
    EXPECT_EQ(crmps_move(i), craw_data_copy[i]);
  }

  auto craw_data_copy2 = mps_copy2.cdata();
  MPST mps_move2(mps_copy2.GetSitesInfo());
  mps_move2 = std::move(mps_copy2);
  const MPST &crmps_move2 = mps_move2;
  EXPECT_EQ(mps_move2.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_move2.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_move2[i], crmps[i]);
    EXPECT_EQ(crmps_move2(i), craw_data_copy2[i]);
  }
}

TEST_F(TestMPS, TestElemAccess) {
  mps.Centralize(2);

  const MPST &crmps = mps;
  Tensor ten = crmps[1];
  EXPECT_EQ(mps.GetTenCanoType(1), MPSTenCanoType::LEFT);
  ten = mps[1];
  EXPECT_EQ(mps.GetTenCanoType(1), MPSTenCanoType::NONE);
  EXPECT_EQ(crmps.GetTenCanoType(1), MPSTenCanoType::NONE);

  const MPST *cpmps = &mps;
  ten = (*cpmps)[3];
  EXPECT_EQ(mps.GetTenCanoType(3), MPSTenCanoType::RIGHT);
  MPST *pmps = &mps;
  ten = (*pmps)[3];
  EXPECT_EQ(mps.GetTenCanoType(3), MPSTenCanoType::NONE);
  EXPECT_EQ(crmps.GetTenCanoType(3), MPSTenCanoType::NONE);
}

TEST_F(TestMPS, TestIO) {
  MPST mps2(SiteVecT(5, pb_out));
  mps.Dump();
  mps2.Load();
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(mps2[i], mps[i]);
  }

  mps.Dump("mps2");
  mps2.Load("mps2");
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(mps2[i], mps[i]);
  }

  mps.Dump("mps3", true);
  EXPECT_TRUE(mps.empty());
}

TEST_F(TestMPS, TestTruncate) {
  TruncateMPS(mps, 0, 1, 3);

  TruncateMPS(mps, 0, 2, 2);

}

TEST_F(TestMPS, TestOperation) {
  mps.Centralize(0);
  mps(0)->Normalize();
  EXPECT_NEAR(FiniteMPSInnerProd(mps, mps), 1.0, 1e-14);
}

using DSiteVec = SiteVec<QLTEN_Double, U1QN>;
using ZSiteVec = SiteVec<QLTEN_Complex, U1QN>;
using DMPS = FiniteMPS<QLTEN_Double, U1QN>;
using ZMPS = FiniteMPS<QLTEN_Complex, U1QN>;

// Test spin systems
struct TestTwoSiteAlgorithmSpinSystem : public testing::Test {
  size_t N = 6;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);
  DSiteVec dsite_vec_6 = DSiteVec(N, pb_out);
  ZSiteVec zsite_vec_6 = ZSiteVec(N, pb_out);

  DQLTensor did = DQLTensor({pb_in, pb_out});
  DQLTensor dsz = DQLTensor({pb_in, pb_out});
  DQLTensor dsp = DQLTensor({pb_in, pb_out});
  DQLTensor dsm = DQLTensor({pb_in, pb_out});
  DMPS dmps = DMPS(dsite_vec_6);

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});
  ZMPS zmps = ZMPS(zsite_vec_6);

  void SetUp(void) {
    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;
  }
};

TEST_F(TestTwoSiteAlgorithmSpinSystem, Test1DHeisenbergEntanglementEntropy) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec_6, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    dmpo_gen.AddTerm(1, {dsz, dsz}, {i, i + 1});
    dmpo_gen.AddTerm(0.5, {dsp, dsm}, {i, i + 1});
    dmpo_gen.AddTerm(0.5, {dsm, dsp}, {i, i + 1});
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      16, 16, 1.0E-9,
      LanczosParams(1.0E-9)
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(dmps, stat_labs);
  dmps.Dump(sweep_params.mps_path, true);
  auto e0 = TwoSiteFiniteVMPS(dmps, dmpo, sweep_params);

  dmps.Load(sweep_params.mps_path);
  std::vector<double> ee_list_std = {0.6931472, 0.3756033, 0.7113730, 0.3756033, 0.6931472};
  std::vector<double> ee_list = dmps.GetEntanglementEntropy(1);
  for (size_t i = 0; i < ee_list.size(); i++) {
    EXPECT_NEAR(ee_list[i], ee_list_std[i], 1e-6);
  }

  dmps.clear();
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec_6, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    zmpo_gen.AddTerm(1, {zsz, zsz}, {i, i + 1});
    zmpo_gen.AddTerm(0.5, {zsp, zsm}, {i, i + 1});
    zmpo_gen.AddTerm(0.5, {zsm, zsp}, {i, i + 1});
  }
  auto zmpo = zmpo_gen.Gen();
  DirectStateInitMps(zmps, stat_labs);
  zmps.Dump(sweep_params.mps_path, true);
  e0 = TwoSiteFiniteVMPS(zmps, zmpo, sweep_params);

  zmps.Load(sweep_params.mps_path);
  ee_list = zmps.GetEntanglementEntropy(1);
  for (size_t i = 0; i < ee_list.size(); i++) {
    EXPECT_NEAR(ee_list[i], ee_list_std[i], 1e-6);
  }

  zmps.clear();

  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}
