// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
*         Haoxin Wang < wanghaoxin1996@gmail.com>
* Creation Date: 2019-10-08 09:38
* 
* Description: QuantumLiquids/UltraDMRG project. Unittest for finite MPS measurements.
*/
#include "qlmps/one_dim_tn/mps_all.h"
#include "qlten/qlten.h"

#include "gtest/gtest.h"
#include <stdlib.h>

using namespace qlmps;
using namespace qlten;

using U1QN = QN<U1QNVal>;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

using DSiteVec = SiteVec<QLTEN_Double, U1QN>;
using ZSiteVec = SiteVec<QLTEN_Complex, U1QN>;
using DMPS = FiniteMPS<QLTEN_Double, U1QN>;
using ZMPS = FiniteMPS<QLTEN_Complex, U1QN>;

inline void ExpectDoubleEq(const double lhs, const double rhs) {
  EXPECT_DOUBLE_EQ(lhs, rhs);
}

inline void ExpectDoubleEq(const QLTEN_Complex lhs, const QLTEN_Complex rhs) {
  EXPECT_DOUBLE_EQ(lhs.real(), rhs.real());
  EXPECT_DOUBLE_EQ(lhs.imag(), rhs.imag());
}

struct TestMpsMeasurement : public testing::Test {
  long N = 6;

  QNT qn0 = QNT({QNCard("N", U1QNVal(0))});
  IndexT pb_out = IndexT({
                             QNSctT(QNT({QNCard("N", U1QNVal(0))}), 1),
                             QNSctT(QNT({QNCard("N", U1QNVal(1))}), 1)},
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);

  DQLTensor dntot = DQLTensor({pb_in, pb_out});
  ZQLTensor zntot = ZQLTensor({pb_in, pb_out});
  DQLTensor did = DQLTensor({pb_in, pb_out});
  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  DMPS dmps = DMPS(DSiteVec(N, pb_out));
  ZMPS zmps = ZMPS(ZSiteVec(N, pb_out));

  std::vector<size_t> stat_labs1;
  std::vector<size_t> stat_labs2;

  void SetUp(void) {
    dntot({0, 0}) = 0;
    dntot({1, 1}) = 1;
    zntot({0, 0}) = 0;
    zntot({1, 1}) = 1;
    did({0, 0}) = 1;
    did({1, 1}) = 1;
    zid({0, 0}) = 1;
    zid({1, 1}) = 1;

    for (long i = 0; i < N; ++i) {
      stat_labs1.push_back(1);
      stat_labs2.push_back(i % 2);
    }
  }
};

template<typename TenElemT, typename QNT>
void RunTestMeasureOneSiteOpCase(
    FiniteMPS<TenElemT, QNT> &mps,
    const QLTensor<TenElemT, QNT> &op,
    const std::vector<TenElemT> &res
) {
  auto measu_res = MeasureOneSiteOp(mps, op, "local_op1_memory");
  assert(measu_res.size() == res.size());
  for (size_t i = 0; i < res.size(); ++i) {
    ExpectDoubleEq(measu_res[i].avg, res[i]);
  }

  mps.Centralize(0);
  std::string mps_path = "test_measure_mps";
  mps.Dump(mps_path, true);
  measu_res = MeasureOneSiteOp(mps, mps_path, op, "local_op1_disk");
  for (size_t i = 0; i < res.size(); ++i) {
    ExpectDoubleEq(measu_res[i].avg, res[i]);
  }
}
//TODO: test for disk version of MeasureOneSiteOp on multiple operators.
TEST_F(TestMpsMeasurement, TestMeasureOneSiteOp) {
  // Double case 1
  std::vector<QLTEN_Double> dres1;
  for (long i = 0; i < N; ++i) { dres1.push_back(stat_labs1[i]); }
  DirectStateInitMps(dmps, stat_labs1);
  RunTestMeasureOneSiteOpCase(dmps, dntot, dres1);
  // Double case 2
  std::vector<QLTEN_Double> dres2;
  for (long i = 0; i < N; ++i) { dres2.push_back(stat_labs2[i]); }
  DirectStateInitMps(dmps, stat_labs2);
  RunTestMeasureOneSiteOpCase(dmps, dntot, dres2);
  // Complex case 1
  std::vector<QLTEN_Complex> zres1;
  for (auto d : dres1) { zres1.push_back(d); }
  DirectStateInitMps(zmps, stat_labs1);
  RunTestMeasureOneSiteOpCase(zmps, zntot, zres1);
  // Complex case 2
  std::vector<QLTEN_Complex> zres2;
  for (auto d : dres2) { zres2.push_back(d); }
  DirectStateInitMps(zmps, stat_labs2);
  RunTestMeasureOneSiteOpCase(zmps, zntot, zres2);
}

template<typename TenElemT, typename QNT>
void RunTestMeasureTwoSiteOpCase(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::vector<QLTensor<TenElemT, QNT>> &phys_ops,
    const QLTensor<TenElemT, QNT> &inst_op,
    const std::vector<std::vector<size_t>> &sites_set,
    const std::vector<TenElemT> &res
) {
  auto measu_res = MeasureTwoSiteOp(
      mps, phys_ops, inst_op, sites_set, "op1op2"
  );
  assert(measu_res.size() == res.size());
  for (size_t i = 0; i < res.size(); ++i) {
    ExpectDoubleEq(measu_res[i].avg, res[i]);
  }
}

TEST_F(TestMpsMeasurement, TestMeasureTwoSiteOp) {
  std::vector<std::vector<size_t>> sites_set = {
      {0, 1}, {0, 2}, {0, 5},
      {1, 2}, {1, 3},
      {4, 5}
  };

  // Double case 1
  std::vector<QLTEN_Double> dres1(sites_set.size(), 1.0);
  DirectStateInitMps(dmps, stat_labs1);
  RunTestMeasureTwoSiteOpCase(dmps, {did, did}, did, sites_set, dres1);
  RunTestMeasureTwoSiteOpCase(dmps, {dntot, dntot}, did, sites_set, dres1);
  // Double case 2
  DirectStateInitMps(dmps, stat_labs2);
  RunTestMeasureTwoSiteOpCase(dmps, {did, did}, did, sites_set, dres1);
  std::vector<QLTEN_Double> dres2 = {0, 0, 0, 0, 1, 0};
  RunTestMeasureTwoSiteOpCase(dmps, {dntot, dntot}, did, sites_set, dres2);

  // Complex case 1
  std::vector<QLTEN_Complex> zres1(sites_set.size(), 1.0);
  DirectStateInitMps(zmps, stat_labs1);
  RunTestMeasureTwoSiteOpCase(zmps, {zid, zid}, zid, sites_set, zres1);
  RunTestMeasureTwoSiteOpCase(zmps, {zntot, zntot}, zid, sites_set, zres1);
  // Double case 2
  DirectStateInitMps(zmps, stat_labs2);
  RunTestMeasureTwoSiteOpCase(zmps, {zid, zid}, zid, sites_set, zres1);
  std::vector<QLTEN_Complex> zres2 = {0, 0, 0, 0, 1, 0};
  RunTestMeasureTwoSiteOpCase(zmps, {zntot, zntot}, zid, sites_set, zres2);
}

template<typename TenElemT, typename QNT>
void RunTestMeasureTwoSiteOpGroupCase(
    FiniteMPS<TenElemT, QNT> &mps,
    const QLTensor<TenElemT, QNT> &op1,
    const QLTensor<TenElemT, QNT> &op2,
    const QLTensor<TenElemT, QNT> &inst_op,
    const size_t site1,
    const std::vector<size_t> &site2_set,
    const std::vector<TenElemT> &res
) {
  mps.Centralize(0);
  std::string mps_path = "test_measure_mps";
  mps.Dump(mps_path, true);

  auto measu_res = MeasureTwoSiteOpGroup(
      mps, mps_path, op1, op2, site1, site2_set,
      inst_op);

  assert(measu_res.size() == res.size());
  for (size_t i = 0; i < res.size(); ++i) {
    ExpectDoubleEq(measu_res[i].avg, res[i]);
  }
  mps.Load(mps_path);
}

//TODO: write distinguishable test cases
TEST_F(TestMpsMeasurement, TestMeasureTwoSiteOpGroup) {
  size_t site1 = 0;
  std::vector<size_t> site2_set = {1, 2, 5};

  // Double case 1
  std::vector<QLTEN_Double> dres1(site2_set.size(), 1.0);
  DirectStateInitMps(dmps, stat_labs1);
  RunTestMeasureTwoSiteOpGroupCase(dmps, did, did, did, site1, site2_set, dres1);
  RunTestMeasureTwoSiteOpGroupCase(dmps, dntot, dntot, did, site1, site2_set, dres1);
  // Double case 2
  DirectStateInitMps(dmps, stat_labs2);
  RunTestMeasureTwoSiteOpGroupCase(dmps, did, did, did, site1, site2_set, dres1);
  std::vector<QLTEN_Double> dres2 = {0, 0, 0};
  RunTestMeasureTwoSiteOpGroupCase(dmps, dntot, dntot, did, site1, site2_set, dres2);

  // Complex case 1
  std::vector<QLTEN_Complex> zres1(site2_set.size(), 1.0);
  DirectStateInitMps(zmps, stat_labs1);
  RunTestMeasureTwoSiteOpGroupCase(zmps, zid, zid, zid, site1, site2_set, zres1);
  RunTestMeasureTwoSiteOpGroupCase(zmps, zntot, zntot, zid, site1, site2_set, zres1);
  // Double case 2
  DirectStateInitMps(zmps, stat_labs2);
  RunTestMeasureTwoSiteOpGroupCase(zmps, zid, zid, zid, site1, site2_set, zres1);
  std::vector<QLTEN_Complex> zres2 = {0, 0, 0};
  RunTestMeasureTwoSiteOpGroupCase(zmps, zntot, zntot, zid, site1, site2_set, zres2);
}

//struct TestNonUniformMpsMeasurement : public testing::Test {
//long N = 4; //unit cell number, A-B-A-B-A-B-A-B

//QN qn0 = QN({QNNameVal("N", 0)});
//Index pb_outA = Index({QNSector(QN({QNNameVal("N", 0)}), 1),
//QNSector(QN({QNNameVal("N", 1)}), 1)}, OUT);
//Index pb_inA = InverseIndex(pb_outA);
//Index pb_outB = Index({QNSector(QN({QNNameVal("N", 0)}), 2)}, OUT);
//Index pb_inB = InverseIndex(pb_outB);

//DQLTensor dntotA = DQLTensor({pb_inA, pb_outA});
//DQLTensor didA = DQLTensor({pb_inA, pb_outA});
//DQLTensor didB = DQLTensor({pb_inB, pb_outB});
//DTenPtrVec dmps = DTenPtrVec(2*N);
//std::vector<Index> pb_out_set = std::vector<Index>(2*N);

//std::vector<long> stat_labs1 = std::vector<long>(2*N,1);
//std::vector<long> A_sublattice=std::vector<long>(N);

//void SetUp(void) {
//dntotA({0, 0}) = 0;
//dntotA({1, 1}) = 1;
//didA({0, 0}) = 1;
//didA({1, 1}) = 1;

//didB({0, 0}) = 1;
//didB({1, 1}) = 1;
//srand((unsigned)time(NULL));
//for (long i = 0; i < 2*N; ++i) {
//if(i%2==0) pb_out_set[i]=pb_outA;
//else pb_out_set[i]=pb_outB;
//}
//for (long i=0;i<N;++i){
//A_sublattice[i]=2*i;
//stat_labs1[2*i]= rand()%2;
//}
//}
//};


//TEST_F(TestNonUniformMpsMeasurement, TestNoUniformMeasureOneSiteOp) {
//std::vector<QLTEN_Double> dres1(N);
//for (long i = 0; i < N; ++i) { dres1[i]=stat_labs1[2*i]; }
//auto dmps1 = dmps;
//DirectStateInitMps(dmps1, stat_labs1, pb_out_set, qn0);
//auto dmps_for_measu1 = MPS<DQLTensor>(dmps1, -1);
//RunTestMeasureOneSiteOpCase(dmps_for_measu1, dntotA,A_sublattice, dres1);
//MpsFree(dmps1);
//}

/////No insertion operator case, can used for uniform or non-uniform indices
//template <typename MpsType, typename TenElemType>
//void RunTestMeasureTwoSiteOpCase(
//MpsType &mps,
//const std::vector<QLTensor<TenElemType>> &phys_ops,
//const std::vector<std::vector<long>> &sites_set,
//const std::vector<TenElemType> &res) {
//auto measu_res = MeasureTwoSiteOp(
//mps, phys_ops, sites_set, "op1op2");
//assert(measu_res.size() == res.size());
//for (size_t i = 0; i < res.size(); ++i) {
//ExpectDoubleEq(measu_res[i].avg, res[i]);
//}
//}

//TEST_F(TestNonUniformMpsMeasurement, TestNoUniformMeasureTwoSiteOp) {
//std::vector<std::vector<long>> sites_set = {
//{0, 2}, {0, 4},{2, 4}
//};
//// Double case 1
//std::vector<QLTEN_Double> dres1(sites_set.size());
//for(int i = 0;i<sites_set.size();i++){
//long site1 = sites_set[i][0];
//long site2 = sites_set[i][1];
//dres1[i]=stat_labs1[site1]*stat_labs1[site2];
//}
//auto dmps1 = dmps;
//DirectStateInitMps(dmps1, stat_labs1, pb_out_set, qn0);
//auto dmps_for_measu1 = MPS<DQLTensor>(dmps1, -1);
//RunTestMeasureTwoSiteOpCase(
//dmps_for_measu1, {dntotA, dntotA}, sites_set, dres1);
//MpsFree(dmps1);
//}
