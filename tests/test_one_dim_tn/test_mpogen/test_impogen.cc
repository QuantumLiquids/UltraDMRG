// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-05
*
* Description: QuantumLiquids/mps project. Unittests for infinite MPO generation.
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
using DMPOGenerator = iMPOGenerator<QLTEN_Double, QNT>;
using ZMPOGenerator = iMPOGenerator<QLTEN_Complex, QNT>;

struct TestIMpoGenerator : public testing::Test {
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
