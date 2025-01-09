// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021/11/2
*
* Description: GraceQ/mps2 project. Unittest for two site finite TDVP algorithm.
*/

#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../testing_utils.h"                                  //RemoveFolder
#include "gtest/gtest.h"

using namespace qlmps;
using namespace qlten;

using U1QN = special_qn::U1QN;
using U1U1QN = special_qn::U1U1QN;

using IndexT = Index<U1QN>;
using IndexT2 = Index<U1U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctT2 = QNSector<U1U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;
using QNSctVecT2 = QNSectorVec<U1U1QN>;
using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using DQLTensor2 = QLTensor<QLTEN_Double, U1U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;
using ZQLTensor2 = QLTensor<QLTEN_Complex, U1U1QN>;
using DSiteVec = SiteVec<QLTEN_Double, U1QN>;
using DSiteVec2 = SiteVec<QLTEN_Double, U1U1QN>;
using ZSiteVec = SiteVec<QLTEN_Complex, U1QN>;
using ZSiteVec2 = SiteVec<QLTEN_Complex, U1U1QN>;
using DMPS = FiniteMPS<QLTEN_Double, U1QN>;
using DMPS2 = FiniteMPS<QLTEN_Double, U1U1QN>;
using ZMPS = FiniteMPS<QLTEN_Complex, U1QN>;
using ZMPS2 = FiniteMPS<QLTEN_Complex, U1U1QN>;

struct TestTwoSiteAlgorithmSpinlessFermion : public testing::Test {
  size_t N = 6;
  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("N", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("N", U1QNVal(-1))}), 1)},
                         TenIndexDirType::OUT
  );

  IndexT pb_in = InverseIndex(pb_out);
  DSiteVec dsite_vec_6 = DSiteVec(N, pb_out);
  ZSiteVec zsite_vec_6 = ZSiteVec(N, pb_out);

  DQLTensor did = DQLTensor({pb_in, pb_out});
  DQLTensor dc = DQLTensor({pb_in, pb_out});
  DQLTensor dcdag = DQLTensor({pb_in, pb_out});
  DQLTensor df = DQLTensor({pb_in, pb_out});//insertion operator
  DMPS dmps = DMPS(dsite_vec_6);

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zc = ZQLTensor({pb_in, pb_out});
  ZQLTensor zcdag = ZQLTensor({pb_in, pb_out});
  ZQLTensor zf = ZQLTensor({pb_in, pb_out});
  ZMPS zmps = ZMPS(zsite_vec_6);

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  void SetUp(void) {
    did({0, 0}) = 1;
    did({1, 1}) = 1;
    df({0, 0}) = -1.0;
    df({1, 1}) = 1.0;
    dc({0, 1}) = 1.0; //annihilation
    dcdag({1, 0}) = 1.0; //creation


    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zf({0, 0}) = -1.0;
    zf({1, 1}) = 1.0;
    zc({0, 1}) = 1.0; //annihilation
    zcdag({1, 0}) = 1.0; //creation

    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    MPI_Comm_rank(comm, &rank);
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

QLTEN_Complex OpenSpinlessFreeFermion1DDynamicCorrelation(
    const double time,
    const size_t L, // chain length, actually should be even to avoid degeneracy
    const size_t x1, //first point, number from 0
    const size_t x2
) {
  QLTEN_Complex res(0.0);
  const double t = 1; //hopping parameter, H = -t Sum ci_dag cj + h.c.
  for (size_t k = 1; k <= L; k++) {
    double epsilon_k = -2 * t * std::cos((k * M_PI) / (L + 1));
    if (epsilon_k > 0) {
      res += std::exp(-QLTEN_Complex(0.0, epsilon_k) * time) *
          sin((k * M_PI) / (L + 1) * (x1 + 1)) * sin((k * M_PI) / (L + 1) * (x2 + 1));
    }
  }

  res = (res * 2.0) / (double) (L + 1);
  return res;
}

TEST_F(TestTwoSiteAlgorithmSpinlessFermion, 1DSpinlessFreeFermion) {
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec_6, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    zmpo_gen.AddTerm(-1.0, {zcdag, zc}, {i, i + 1});
    zmpo_gen.AddTerm(-1.0, {zc, zcdag}, {i, i + 1});
  }
  auto zmpo = zmpo_gen.Gen();
  FiniteVMPSSweepParams vmps_sweep_params = FiniteVMPSSweepParams(
      4,
      1, 16, 1.0E-10,
      LanczosParams(1.0E-8)
  );
  double e0(0.0);
  if (rank == 0) {
    RemoveFolder(vmps_sweep_params.mps_path);
    RemoveFolder(vmps_sweep_params.temp_path);
    std::vector<size_t> stat_labs;
    for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
    DirectStateInitMps(zmps, stat_labs);
    zmps.Dump(vmps_sweep_params.mps_path, true);
    e0 = TwoSiteFiniteVMPSWithNoise(zmps, zmpo, vmps_sweep_params);
    double benchmark_e0 = -3.4939592074349334893668128643185;
    EXPECT_NEAR(e0, benchmark_e0, 1e-13);
  }
  ::MPI_Bcast(&e0, 1, MPI_DOUBLE, 0, comm);

  MPITDVPSweepParams<U1QN> tdvp_sweep_params = MPITDVPSweepParams<U1QN>(
      0.01, 30,
      N / 2,
      zcdag, zf, zc, zf, e0,
      10, 16, 1.0E-10,
      LanczosParams(1.0E-8)
  );
  if (rank == 0) {
    RemoveFolder(tdvp_sweep_params.initial_mps_path);
    RemoveFolder(tdvp_sweep_params.measure_temp_path);
  }
  auto dynamic_correlation =
      TwoSiteFiniteTDVP(zmps, zmpo, tdvp_sweep_params, "spinless_fermion_single_particle_dynamic", comm);

  if (rank == 0) {
    for (size_t i = 0; i < dynamic_correlation.size(); i++) {
      QLTEN_Complex correlation_val = dynamic_correlation[i].avg;
      double dtime = dynamic_correlation[i].times[1];
      int x1 = dynamic_correlation[i].sites[0];
      int x2 = dynamic_correlation[i].sites[1];
      QLTEN_Complex benchmark_correlation_val = OpenSpinlessFreeFermion1DDynamicCorrelation(
          dtime, N, x1, x2
      );
      EXPECT_NEAR(correlation_val.real(), benchmark_correlation_val.real(), 1E-10);
      EXPECT_NEAR(correlation_val.imag(), benchmark_correlation_val.imag(), 1E-10);
    }
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  ::testing::InitGoogleTest(&argc, argv);
  int test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}


