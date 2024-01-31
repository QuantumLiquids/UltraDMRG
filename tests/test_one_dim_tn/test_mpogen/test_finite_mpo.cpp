/*
* Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-12-13
*
* Description: QuantumLiquids/UltraDMRG project. Unittests for FiniteMPO
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/qlmps.h"

using namespace qlmps;
using namespace qlten;

using special_qn::U1QN;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;;

struct TestFiniteMPO : public testing::Test {




};
