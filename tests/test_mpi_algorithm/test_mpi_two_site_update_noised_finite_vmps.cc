// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-26
*
* Description: GraceQ/mps2 project. Unittest for MPI two sites algorithm.
*/


#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../testing_utils.h"

#include <vector>

#include <stdlib.h>     // system

using namespace qlmps;
using namespace qlten;
namespace mpi = boost::mpi;

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

template<typename TenElemT, typename QNT>
void RunTestVMPSCase(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const double benmrk_e0, const double precision,
    mpi::communicator &world
) {
  size_t start_flops = flop;
  Timer vmps_timer("two_site_vmps_with_noise");
  auto e0 = TwoSiteFiniteVMPS(mps, mpo, sweep_params, world);
  double elapsed_time = vmps_timer.Elapsed();
  size_t end_flops = flop;
  double Gflops_s = (end_flops - start_flops) * 1.e-9 / elapsed_time;
  std::cout << "flops = " << end_flops - start_flops << std::endl;
  std::cout << "Gflops/s = " << Gflops_s << std::endl;
  if (world.rank() == kMasterRank) {
    EXPECT_NEAR(e0, benmrk_e0, precision);
    EXPECT_TRUE(mps.empty());
  }
}

// Test spin systems
struct Test2DSpinSystem : public testing::Test {
  size_t Lx = 4;
  size_t Ly = 4;
  size_t N = Lx * Ly;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);
  DSiteVec dsite_vec_2d = DSiteVec(N, pb_out);
  ZSiteVec zsite_vec_2d = ZSiteVec(N, pb_out);

  DQLTensor did = DQLTensor({pb_in, pb_out});
  DQLTensor dsz = DQLTensor({pb_in, pb_out});
  DQLTensor dsp = DQLTensor({pb_in, pb_out});
  DQLTensor dsm = DQLTensor({pb_in, pb_out});
  DMPS dmps = DMPS(dsite_vec_2d);

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});
  ZMPS zmps = ZMPS(zsite_vec_2d);

  std::vector<std::pair<size_t, size_t>> nn_pairs =
      std::vector<std::pair<size_t, size_t>>(size_t(2 * Lx * Ly - Ly));

  mpi::communicator world;
  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }

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

    auto iter = nn_pairs.begin();
    for (size_t i = 0; i < Lx; i++) {
      for (size_t j = 0; j < Ly; j++) {
        size_t site_a = i * Ly + j;
        if (j != Ly - 1) {
          size_t site_b = site_a + 1;
          iter->first = site_a;
          iter->second = site_b;
        } else {
          size_t site_b = i * Ly;
          iter->first = site_b;
          iter->second = site_a;
        }
        iter++;
      }
    }
    for (size_t i = 0; i < Lx - 1; i++) {
      for (size_t j = 0; j < Ly; j++) {
        size_t site_a = i * Ly + j;
        size_t site_b = (i + 1) * Ly + j;
        iter->first = site_a;
        iter->second = site_b;
        iter++;
      }
    }
    assert(iter == nn_pairs.end());
  }
};

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
  DirectStateInitMps(dmps, stat_labs);

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      100, 100, 1.0E-9,
      LanczosParams(1.0E-7),
      std::vector<double>{0.01, 0.001, 0.0001, 1e-6}
  );

  if (world.rank() == 0) {
    dmps.Dump(sweep_params.mps_path, true);
    if (IsPathExist(sweep_params.temp_path)) {
      RemoveFolder(sweep_params.temp_path);
    }
  }

  RunTestVMPSCase(dmps, dmpo, sweep_params, -10.264281906484872, 1e-6, world);

  if (world.rank() == 0) {
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

  DirectStateInitMps(zmps, stat_labs);
  if (world.rank() == 0) {
    zmps.Dump(sweep_params.mps_path, true);
    if (IsPathExist(sweep_params.temp_path)) {
      RemoveFolder(sweep_params.temp_path);
    }
  }

  RunTestVMPSCase(zmps, zmpo, sweep_params, -10.264281906484872, 1e-6, world);

  if (world.rank() == 0) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
  }
}

// Test fermion models.
struct TestTwoSiteAlgorithmTjSystem2U1Symm : public testing::Test {
  size_t N = 4;
  double t = 3.0;
  double J = 1.0;
  U1U1QN qn0 = U1U1QN({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))});
  IndexT2 pb_out = IndexT2({
                               QNSctT2(U1U1QN({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(1))}), 1),
                               QNSctT2(U1U1QN({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(-1))}), 1),
                               QNSctT2(U1U1QN({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}), 1)},
                           TenIndexDirType::OUT
  );
  IndexT2 pb_in = InverseIndex(pb_out);
  DSiteVec2 dsite_vec_4 = DSiteVec2(N, pb_out);
  ZSiteVec2 zsite_vec_4 = ZSiteVec2(N, pb_out);

  DQLTensor2 df = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dsz = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dsp = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dsm = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dcup = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dcdagup = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dcdn = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dcdagdn = DQLTensor2({pb_in, pb_out});
  DMPS2 dmps = DMPS2(dsite_vec_4);

  ZQLTensor2 zf = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 zsz = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 zsp = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 zsm = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 zcup = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 zcdagup = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 zcdn = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 zcdagdn = ZQLTensor2({pb_in, pb_out});
  ZMPS2 zmps = ZMPS2(zsite_vec_4);

  boost::mpi::communicator world;

  void SetUp(void) {
    df({0, 0}) = -1;
    df({1, 1}) = -1;
    df({2, 2}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({1, 0}) = 1;
    dsm({0, 1}) = 1;
    dcup({0, 2}) = 1;
    dcdagup({2, 0}) = 1;
    dcdn({1, 2}) = 1;
    dcdagdn({2, 1}) = 1;

    zf({0, 0}) = -1;
    zf({1, 1}) = -1;
    zf({2, 2}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({1, 0}) = 1;
    zsm({0, 1}) = 1;
    zcup({0, 2}) = 1;
    zcdagup({2, 0}) = 1;
    zcdn({1, 2}) = 1;
    zcdagdn({2, 1}) = 1;

    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

TEST_F(TestTwoSiteAlgorithmTjSystem2U1Symm, 2DCase) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1U1QN>(dsite_vec_4, qn0);
  std::vector<std::pair<int, int>> nn_pairs = {
      std::make_pair(0, 1),
      std::make_pair(0, 2),
      std::make_pair(2, 3),
      std::make_pair(1, 3)};
  for (auto &p : nn_pairs) {
    dmpo_gen.AddTerm(-t, dcdagup, p.first, dcup, p.second, df);
    dmpo_gen.AddTerm(-t, dcdagdn, p.first, dcdn, p.second, df);
    dmpo_gen.AddTerm(-t, dcup, p.first, dcdagup, p.second, df);
    dmpo_gen.AddTerm(-t, dcdn, p.first, dcdagdn, p.second, df);
    dmpo_gen.AddTerm(J, dsz, p.first, dsz, p.second);
    dmpo_gen.AddTerm(J / 2, dsp, p.first, dsm, p.second);
    dmpo_gen.AddTerm(J / 2, dsm, p.first, dsp, p.second);
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      8,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-8, 20),
      std::vector<double>{0.01, 0.001, 0.0001, 1e-6}
  );

  auto total_div = U1U1QN({QNCard("N", U1QNVal(N - 2)), QNCard("Sz", U1QNVal(0))});
  auto zero_div = U1U1QN({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))});

  // Direct product state initialization.
  DirectStateInitMps(dmps, {2, 0, 1, 2});
  if (world.rank() == 0) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    dmps.Dump(sweep_params.mps_path, true);
  }

  RunTestVMPSCase(dmps, dmpo, sweep_params, -8.868563739680, 1e-6, world);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1U1QN>(zsite_vec_4, qn0);
  for (auto &p : nn_pairs) {
    zmpo_gen.AddTerm(-t, zcdagup, p.first, zcup, p.second, zf);
    zmpo_gen.AddTerm(-t, zcdagdn, p.first, zcdn, p.second, zf);
    zmpo_gen.AddTerm(-t, zcup, p.first, zcdagup, p.second, zf);
    zmpo_gen.AddTerm(-t, zcdn, p.first, zcdagdn, p.second, zf);
    zmpo_gen.AddTerm(J, zsz, p.first, zsz, p.second);
    zmpo_gen.AddTerm(J / 2, zsp, p.first, zsm, p.second);
    zmpo_gen.AddTerm(J / 2, zsm, p.first, zsp, p.second);
  }
  auto zmpo = zmpo_gen.Gen();
  DirectStateInitMps(zmps, {2, 0, 1, 2});
  if (world.rank() == 0) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    zmps.Dump(sweep_params.mps_path, true);
  }
  RunTestVMPSCase(
      zmps, zmpo, sweep_params,
      -8.868563739680, 1.0E-6,
      world
  );
}

// Electron-phonon interaction Holstein chain, ref 10.1103/PhysRevB.57.6376
struct TestTwoSiteAlgorithmElectronPhononSystem : public testing::Test {
  unsigned L = 4;           // The number of electron
  unsigned Np = 3;          // per electron has Np pseudosite
  double t = 1;         // electron hopping
  double g = 1;         // electron-phonon interaction
  double U = 8;         // Hubbard U
  double omega = 5;     // phonon on-site potential
  unsigned N = (1 + Np) * L;     // The length of the mps/mpo

  U1U1QN qn0 = U1U1QN({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))});
  //Fermion(electron)
  IndexT2 pb_outF = IndexT2(
      {
          QNSctT2(U1U1QN({QNCard("N", U1QNVal(2)), QNCard("Sz", U1QNVal(0))}), 1),
          QNSctT2(U1U1QN({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(1))}), 1),
          QNSctT2(U1U1QN({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(-1))}), 1),
          QNSctT2(U1U1QN({QNCard("S", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}), 1)
      },
      TenIndexDirType::OUT
  );
  IndexT2 pb_inF = InverseIndex(pb_outF);
  //Boson(Phonon)
  IndexT2 pb_outB = IndexT2({QNSctT2(qn0, 2)}, TenIndexDirType::OUT);
  IndexT2 pb_inB = InverseIndex(pb_outB);

  //ZSiteVec2 zsite_vec;

  DQLTensor2 nf = DQLTensor2({pb_inF, pb_outF}); //fermion number
  DQLTensor2 bupcF = DQLTensor2({pb_inF, pb_outF});
  DQLTensor2 bupaF = DQLTensor2({pb_inF, pb_outF});
  DQLTensor2 Fbdnc = DQLTensor2({pb_inF, pb_outF});
  DQLTensor2 Fbdna = DQLTensor2({pb_inF, pb_outF});
  DQLTensor2 bupc = DQLTensor2({pb_inF, pb_outF});
  DQLTensor2 bupa = DQLTensor2({pb_inF, pb_outF});
  DQLTensor2 bdnc = DQLTensor2({pb_inF, pb_outF});
  DQLTensor2 bdna = DQLTensor2({pb_inF, pb_outF});
  DQLTensor2 Uterm = DQLTensor2({pb_inF, pb_outF}); // Hubbard Uterm, nup*ndown

  DQLTensor2 a = DQLTensor2({pb_inB, pb_outB}); //bosonic annihilation
  DQLTensor2 adag = DQLTensor2({pb_inB, pb_outB}); //bosonic creation
  DQLTensor2 P1 = DQLTensor2({pb_inB, pb_outB}); // the number of phonon
  DQLTensor2 idB = DQLTensor2({pb_inB, pb_outB}); // bosonic identity
  DQLTensor2 P0 = DQLTensor2({pb_inB, pb_outB});

  std::vector<IndexT2> index_set = std::vector<IndexT2>(N);

  ZQLTensor2 znf = ZQLTensor2({pb_inF, pb_outF}); //fermion number
  ZQLTensor2 zbupcF = ZQLTensor2({pb_inF, pb_outF});
  ZQLTensor2 zbupaF = ZQLTensor2({pb_inF, pb_outF});
  ZQLTensor2 zFbdnc = ZQLTensor2({pb_inF, pb_outF});
  ZQLTensor2 zFbdna = ZQLTensor2({pb_inF, pb_outF});
  ZQLTensor2 zbupc = ZQLTensor2({pb_inF, pb_outF});
  ZQLTensor2 zbupa = ZQLTensor2({pb_inF, pb_outF});
  ZQLTensor2 zbdnc = ZQLTensor2({pb_inF, pb_outF});
  ZQLTensor2 zbdna = ZQLTensor2({pb_inF, pb_outF});
  ZQLTensor2 zUterm = ZQLTensor2({pb_inF, pb_outF}); // Hubbard Uterm, nup*ndown

  ZQLTensor2 za = ZQLTensor2({pb_inB, pb_outB}); //bosonic annihilation
  ZQLTensor2 zadag = ZQLTensor2({pb_inB, pb_outB}); //bosonic creation
  ZQLTensor2 zP1 = ZQLTensor2({pb_inB, pb_outB}); // the number of phonon
  ZQLTensor2 zidB = ZQLTensor2({pb_inB, pb_outB}); // bosonic identity
  ZQLTensor2 zP0 = ZQLTensor2({pb_inB, pb_outB});

  boost::mpi::communicator world;

  void SetUp(void) {
    nf({0, 0}) = 2;
    nf({1, 1}) = 1;
    nf({2, 2}) = 1;

    bupcF({0, 2}) = -1;
    bupcF({1, 3}) = 1;
    Fbdnc({0, 1}) = 1;
    Fbdnc({2, 3}) = -1;
    bupaF({2, 0}) = 1;
    bupaF({3, 1}) = -1;
    Fbdna({1, 0}) = -1;
    Fbdna({3, 2}) = 1;

    bupc({0, 2}) = 1;
    bupc({1, 3}) = 1;
    bdnc({0, 1}) = 1;
    bdnc({2, 3}) = 1;
    bupa({2, 0}) = 1;
    bupa({3, 1}) = 1;
    bdna({1, 0}) = 1;
    bdna({3, 2}) = 1;

    Uterm({0, 0}) = 1;

    adag({0, 1}) = 1;
    a({1, 0}) = 1;
    P1({0, 0}) = 1;
    idB({0, 0}) = 1;
    idB({1, 1}) = 1;
    P0 = idB + (-P1);

    znf = ToComplex(nf);

    zbupcF = ToComplex(bupcF);
    zFbdnc = ToComplex(Fbdnc);
    zbupaF = ToComplex(bupaF);
    zFbdna = ToComplex(Fbdna);
    zbupc = ToComplex(bupc);
    zbdnc = ToComplex(bdnc);
    zbupa = ToComplex(bupa);
    zbdna = ToComplex(bdna);

    zUterm = ToComplex(Uterm);
    zadag = ToComplex(adag);
    za = ToComplex(a);
    zP1 = ToComplex(P1);
    zidB = ToComplex(idB);
    zP0 = ToComplex(P0);

    for (size_t i = 0; i < N; ++i) {
      if (i % (Np + 1) == 0) {
        index_set[i] = pb_outF;     // even site is fermion
      } else {
        index_set[i] = pb_outB;     // odd site is boson
      }
    }

    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

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
  DirectStateInitMps(dmps, stat_labs);
  auto sweep_params = FiniteVMPSSweepParams(
      6,
      256, 256, 1.0E-10,
      LanczosParams(1.0E-8),
      std::vector<double>{0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0}
  );
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    dmps.Dump(sweep_params.mps_path, true);
  }
  RunTestVMPSCase(
      dmps, dmpo, sweep_params,
      -1.9363605088186260, 1.0E-7, world
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
  DirectStateInitMps(zmps, stat_labs);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    zmps.Dump(sweep_params.mps_path, true);
  }
  RunTestVMPSCase(
      dmps, dmpo, sweep_params,
      -1.9363605088186260, 1.0E-7, world
  );
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  boost::mpi::environment env;
  return RUN_ALL_TESTS();
}



