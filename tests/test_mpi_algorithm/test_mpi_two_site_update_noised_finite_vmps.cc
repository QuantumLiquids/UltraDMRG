// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-26
*
* Description: QuantumLiquids/MPS project. Unittest for MPI two sites algorithm.
*/


#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "../testing_utils.h"
#include "mpi_test_systems.h"

template<typename TenElemT, typename QNT>
void RunTestVMPSCase(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const double benmrk_e0, const double precision,
    const MPI_Comm &comm
) {
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);
  size_t start_flops = flop;
  Timer vmps_timer("two_site_vmps_with_noise");
  auto e0 = TwoSiteFiniteVMPS(mps, mpo, sweep_params, comm);
  double elapsed_time = vmps_timer.Elapsed();
  size_t end_flops = flop;
  double Gflops_s = (end_flops - start_flops) * 1.e-9 / elapsed_time;
  std::cout << "flops = " << end_flops - start_flops << std::endl;
  std::cout << "Gflops/s = " << Gflops_s << std::endl;
  if (rank == kMPIMasterRank) {
    EXPECT_NEAR(e0, benmrk_e0, precision);
    EXPECT_TRUE(mps.empty());
  }
}

TEST_F(Test2DSpinSystem, 2DHeisenberg) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec_2d, qn0);

  for (auto &p : nn_pairs) {
    dmpo_gen.AddTerm(1, {dsz, dsz}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5, {dsp, dsm}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5, {dsm, dsp}, {p.first, p.second});
  }
  auto dmpo = dmpo_gen.Gen();

  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      100, 100, 1.0E-9,
      LanczosParams(1.0E-7),
      std::vector<double>{0.01, 0.001, 0.0001, 1e-6}
  );

  if (rank == 0) {
    DirectStateInitMps(dmps, stat_labs);
    dmps.Dump(sweep_params.mps_path, true);
    if (IsPathExist(sweep_params.temp_path)) {
      RemoveFolder(sweep_params.temp_path);
    }
  }

  RunTestVMPSCase(dmps, dmpo, sweep_params, -10.264281906484872, 1e-6, comm);

  if (rank == 0) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
  }

  //Complex case
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec_2d, qn0);

  for (auto &p : nn_pairs) {
    zmpo_gen.AddTerm(1, {zsz, zsz}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5, {zsp, zsm}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5, {zsm, zsp}, {p.first, p.second});
  }
  auto zmpo = zmpo_gen.Gen();

  if (rank == 0) {
    DirectStateInitMps(zmps, stat_labs);
    zmps.Dump(sweep_params.mps_path, true);
    if (IsPathExist(sweep_params.temp_path)) {
      RemoveFolder(sweep_params.temp_path);
    }
  }

  RunTestVMPSCase(zmps, zmpo, sweep_params, -10.264281906484872, 1e-6, comm);

  if (rank == 0) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
  }
}

TEST_F(TestTwoSiteAlgorithmElectronPhononSystem, holsteinchain) {
  DSiteVec2 dsite_vec = DSiteVec2(index_set);
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1U1QN>(dsite_vec, qn0);
  for (int i = 0; i < N - Np - 1; i = i + Np + 1) {
    dmpo_gen.AddTerm(-t, bupcF, i, bupa, i + Np + 1);
    dmpo_gen.AddTerm(-t, bdnc, i, Fbdna, i + Np + 1);
    dmpo_gen.AddTerm(t, bupaF, i, bupc, i + Np + 1);
    dmpo_gen.AddTerm(t, bdna, i, Fbdnc, i + Np + 1);
  }
  for (unsigned long i = 0; i < N; i = i + Np + 1) {
    dmpo_gen.AddTerm(U, Uterm, i);
    for (unsigned long j = 0; j < Np; ++j) {
      dmpo_gen.AddTerm(omega, (double) (pow(2, j)) * P1, i + j + 1);
    }
    dmpo_gen.AddTerm(2 * g, {nf, a, a, adag}, {i, i + 1, i + 2, i + 3});
    dmpo_gen.AddTerm(2 * g, {nf, adag, adag, a}, {i, i + 1, i + 2, i + 3});
    dmpo_gen.AddTerm(g, {nf, a, adag, sqrt(2) * P0 + sqrt(6) * P1}, {i, i + 1, i + 2, i + 3});
    dmpo_gen.AddTerm(g, {nf, adag, a, sqrt(2) * P0 + sqrt(6) * P1}, {i, i + 1, i + 2, i + 3});
    dmpo_gen.AddTerm(g, {nf, a + adag, P1, sqrt(3) * P0 + sqrt(7) * P1}, {i, i + 1, i + 2, i + 3});
    dmpo_gen.AddTerm(g, {nf, a + adag, P0, P0 + sqrt(5) * P1}, {i, i + 1, i + 2, i + 3});
  }

  auto dmpo = dmpo_gen.Gen();
  DMPS2 dmps = DMPS2(dsite_vec);
  std::vector<unsigned long> stat_labs(N, 0);
  int qn_label = 1;
  for (int i = 0; i < N; i = i + Np + 1) {
    stat_labs[i] = qn_label;
    qn_label = 3 - qn_label;
  }

  auto sweep_params = FiniteVMPSSweepParams(
      6,
      256, 256, 1.0E-10,
      LanczosParams(1.0E-8),
      std::vector<double>{0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0}
  );
  if (rank == kMPIMasterRank) {
    DirectStateInitMps(dmps, stat_labs);
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    dmps.Dump(sweep_params.mps_path, true);
  }
  RunTestVMPSCase(
      dmps, dmpo, sweep_params,
      -1.9363605088186260, 1.0E-7, comm
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);



  //Complex case
  ZSiteVec2 zsite_vec = ZSiteVec2(index_set);
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1U1QN>(zsite_vec, qn0);
  for (int i = 0; i < N - Np - 1; i = i + Np + 1) {
    zmpo_gen.AddTerm(-t, zbupcF, i, zbupa, i + Np + 1);
    zmpo_gen.AddTerm(-t, zbdnc, i, zFbdna, i + Np + 1);
    zmpo_gen.AddTerm(t, zbupaF, i, zbupc, i + Np + 1);
    zmpo_gen.AddTerm(t, zbdna, i, zFbdnc, i + Np + 1);
  }
  for (unsigned long i = 0; i < N; i = i + Np + 1) {
    zmpo_gen.AddTerm(U, zUterm, i);
    for (unsigned long j = 0; j < Np; ++j) {
      zmpo_gen.AddTerm(omega, (double) (pow(2, j)) * zP1, i + j + 1);
    }
    zmpo_gen.AddTerm(2 * g, {znf, za, za, zadag}, {i, i + 1, i + 2, i + 3});
    zmpo_gen.AddTerm(2 * g, {znf, zadag, zadag, za}, {i, i + 1, i + 2, i + 3});
    zmpo_gen.AddTerm(g, {znf, za, zadag, sqrt(2) * zP0 + sqrt(6) * zP1}, {i, i + 1, i + 2, i + 3});
    zmpo_gen.AddTerm(g, {znf, zadag, za, sqrt(2) * zP0 + sqrt(6) * zP1}, {i, i + 1, i + 2, i + 3});
    zmpo_gen.AddTerm(g, {znf, za + zadag, zP1, sqrt(3) * zP0 + sqrt(7) * zP1}, {i, i + 1, i + 2, i + 3});
    zmpo_gen.AddTerm(g, {znf, za + zadag, zP0, zP0 + sqrt(5) * zP1}, {i, i + 1, i + 2, i + 3});
  }

  auto zmpo = zmpo_gen.Gen();
  ZMPS2 zmps = ZMPS2(zsite_vec);
  if (rank == kMPIMasterRank) {
    DirectStateInitMps(zmps, stat_labs);
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    zmps.Dump(sweep_params.mps_path, true);
  }
  RunTestVMPSCase(
      dmps, dmpo, sweep_params,
      -1.9363605088186260, 1.0E-7, comm
  );
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  ::testing::InitGoogleTest(&argc, argv);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}



