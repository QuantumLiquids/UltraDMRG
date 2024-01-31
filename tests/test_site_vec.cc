// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-07-29 16:48
* 
* Description: QuantumLiquids/UltraDMRG project. Unittests for SiteVec .
*/
#include "qlmps/site_vec.h"
#include "qlten/qlten.h"

#include "gtest/gtest.h"


using namespace qlmps;
using namespace qlten;


using QNT = QN<U1QNVal>;
using QNSctT = QNSector<QNT>;
using IndexT = Index<QNT>;
using Tensor = QLTensor<QLTEN_Double, QNT>;


template <typename TenT>
void TestIsIdOp(const TenT &ten) {
  EXPECT_EQ(ten.GetIndexes().size(), 2);
  EXPECT_EQ(ten.GetShape()[0], ten.GetShape()[1]);
  EXPECT_EQ(ten.GetIndexes()[0].GetDir(), IN);
  EXPECT_EQ(InverseIndex(ten.GetIndexes()[0]), ten.GetIndexes()[1]);

  auto dim = ten.GetShape()[0];
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      if (i == j) {
        EXPECT_EQ(ten(i, j), 1.0);
      } else {
        EXPECT_EQ(ten(i, j), 0.0);
      }
    }
  }
}


template <typename TenElemT, typename QNT>
void RunTestSiteVecBasicFeatures(
    const int N,
    const Index<QNT> &local_hilbert_space
) {
  SiteVec<TenElemT, QNT> site_vec(N, local_hilbert_space);
  EXPECT_EQ(site_vec.size, N);
  Index<QNT> site;
  if (local_hilbert_space.GetDir() == OUT) {
    site = local_hilbert_space;
  } else {
    site = InverseIndex(local_hilbert_space);
  }
  EXPECT_EQ(site_vec.sites, IndexVec<QNT>(N, site));
  for (int i = 0; i < site_vec.size; ++i) {
    TestIsIdOp(site_vec.id_ops[i]);
  }

  EXPECT_EQ(site_vec, site_vec);
}


template <typename TenElemT, typename QNT>
void RunTestSiteVecBasicFeatures(const IndexVec<QNT> &local_hilbert_spaces) {
  SiteVec<TenElemT, QNT> site_vec(local_hilbert_spaces);
  EXPECT_EQ(site_vec.size, local_hilbert_spaces.size());
  for (int i = 0; i < site_vec.size; ++i) {
    if (local_hilbert_spaces[i].GetDir() == OUT) {
      EXPECT_EQ(site_vec.sites[i], local_hilbert_spaces[i]);
    } else {
      EXPECT_EQ(site_vec.sites[i], InverseIndex(local_hilbert_spaces[i]));
    }
    TestIsIdOp(site_vec.id_ops[i]);
  }
}


TEST(TestSiteVec, TestBasicFeatures) {
  IndexT pb_out1 = IndexT({
                          QNSctT(QNT({QNCard("N", U1QNVal(0))}), 1),
                          QNSctT(QNT({QNCard("N", U1QNVal(1))}), 1)
                      },
                      OUT
                   );
  IndexT pb_in1 = InverseIndex(pb_out1);
  IndexT pb_out2 = IndexT({
                          QNSctT(QNT({QNCard("N", U1QNVal(1))}), 3)
                      },
                      OUT
                  );
  IndexT pb_in2 = InverseIndex(pb_out2);
  RunTestSiteVecBasicFeatures<QLTEN_Double, QNT>(1, pb_out1);
  RunTestSiteVecBasicFeatures<QLTEN_Double, QNT>(1, pb_in1);
  RunTestSiteVecBasicFeatures<QLTEN_Double, QNT>(3, pb_out1);
  RunTestSiteVecBasicFeatures<QLTEN_Double, QNT>(3, pb_in1);

  RunTestSiteVecBasicFeatures<QLTEN_Double, QNT>({pb_out1});
  RunTestSiteVecBasicFeatures<QLTEN_Double, QNT>({pb_out2, pb_out2});
  RunTestSiteVecBasicFeatures<QLTEN_Double, QNT>({pb_out1, pb_out2, pb_out1});
  RunTestSiteVecBasicFeatures<QLTEN_Double, QNT>({pb_in2, pb_out1, pb_out1, pb_in1, pb_out2});
}
