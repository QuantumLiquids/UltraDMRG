// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-19 17:33
*
* Description: QuantumLiquids/UltraDMRG project. Unittests for DuoVector .
*/
#include "qlmps/one_dim_tn/framework/duovector.h"
#include "gtest/gtest.h"

#include <utility>    // move


using namespace qlmps;


template <typename ElemT>
void RunTestDuoVectorConstructorsCase(const size_t size) {
  DuoVector<ElemT> duovec(size);
  EXPECT_EQ(duovec.size(), size);
  auto craw_data = duovec.cdata();
  for (auto &rpelem : craw_data) {
    EXPECT_EQ(rpelem, nullptr);
  }

  for (size_t i = 0; i < size; ++i) {
    duovec[i] = i + 5;
  }

  DuoVector<ElemT> duovec_copy(duovec);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(duovec_copy[i], duovec[i]);
    EXPECT_NE(duovec_copy(i), duovec(i));
  }

  auto craw_data_copy = duovec_copy.cdata();
  DuoVector<ElemT> duovec_moved(std::move(duovec_copy));
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(duovec_moved(i), craw_data_copy[i]);
    EXPECT_EQ(duovec_moved[i], duovec[i]);
  }

  DuoVector<ElemT> duovec_copy2;
  duovec_copy2 = duovec;
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(duovec_copy2[i], duovec[i]);
    EXPECT_NE(duovec_copy2(i), duovec(i));
  }

  auto craw_data_copy2 = duovec_copy2.cdata();
  DuoVector<ElemT> duovec_moved2;
  duovec_moved2 = std::move(duovec_copy2);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(duovec_moved2(i), craw_data_copy2[i]);
    EXPECT_EQ(duovec_moved2[i], duovec[i]);
  }
}


TEST(TestDuoVector, TestConstructors) {
  DuoVector<int> default_duovec;
  EXPECT_EQ(default_duovec.size(), 0);

  RunTestDuoVectorConstructorsCase<int>(1);
  RunTestDuoVectorConstructorsCase<int>(3);

  RunTestDuoVectorConstructorsCase<double>(1);
  RunTestDuoVectorConstructorsCase<double>(3);
}


TEST(TestDuoVector, TestElemAccess) {
  DuoVector<int> intduovec(1);

  intduovec[0] = 3;
  EXPECT_EQ(intduovec[0], 3);

  EXPECT_EQ(intduovec.front(), 3);
  EXPECT_EQ(intduovec.back(), 3);
  intduovec.front() = 6;
  EXPECT_EQ(intduovec.front(), 6);
  EXPECT_EQ(intduovec.back(), 6);

  auto pelem = intduovec.cdata()[0];
  intduovec[0] = 5;
  EXPECT_EQ(intduovec.cdata()[0], pelem);
  EXPECT_EQ(intduovec[0], 5);

  auto pelem2 = new int(4);
  delete intduovec(0);
  intduovec(0) = pelem2;
  EXPECT_EQ(intduovec[0], 4);
  EXPECT_NE(intduovec.cdata()[0], pelem);
  EXPECT_EQ(intduovec.cdata()[0], pelem2);
}


TEST(TestDuoVector, TestElemAllocDealloc) {
  DuoVector<int> intduovec(2);

  intduovec.alloc(0);
  EXPECT_NE(intduovec.cdata()[0], nullptr);
  EXPECT_EQ(intduovec.cdata()[1], nullptr);

  intduovec[0] = 3;
  EXPECT_EQ(intduovec[0], 3);

  intduovec.dealloc(0);
  EXPECT_EQ(intduovec.cdata()[0], nullptr);

  intduovec.dealloc(1);
  EXPECT_EQ(intduovec.cdata()[1], nullptr);
}
