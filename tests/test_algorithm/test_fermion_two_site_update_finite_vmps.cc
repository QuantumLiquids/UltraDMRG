// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-14 16:08
*
* Description: GraceQ/mps2 project. Unittest for two sites algorithm.
*/

#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlmps/one_dim_tn/mpo/mpo.h"                              // MPO
#include "qlmps/one_dim_tn/mpo/mpogen/mpogen.h"                    // MPOGenerator
#include "qlmps/algorithm/vmps/vmps_all.h"
#include "../testing_utils.h"

using namespace qlmps;
using namespace qlten;

using special_qn::fU1QN;
using special_qn::fU1U1QN;

using IndexT = Index<fU1QN>;
using IndexT2 = Index<fU1U1QN>;
using QNSctT = QNSector<fU1QN>;
using QNSctT2 = QNSector<fU1U1QN>;
using QNSctVecT = QNSectorVec<fU1QN>;
using QNSctVecT2 = QNSectorVec<fU1U1QN>;
using DQLTensor = QLTensor<QLTEN_Double, fU1QN>;
using DQLTensor2 = QLTensor<QLTEN_Double, fU1U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, fU1QN>;
using ZQLTensor2 = QLTensor<QLTEN_Complex, fU1U1QN>;
using DSiteVec = SiteVec<QLTEN_Double, fU1QN>;
using DSiteVec2 = SiteVec<QLTEN_Double, fU1U1QN>;
using ZSiteVec = SiteVec<QLTEN_Complex, fU1QN>;
using ZSiteVec2 = SiteVec<QLTEN_Complex, fU1U1QN>;
using DMPS = FiniteMPS<QLTEN_Double, fU1QN>;
using DMPS2 = FiniteMPS<QLTEN_Double, fU1U1QN>;
using ZMPS = FiniteMPS<QLTEN_Complex, fU1QN>;
using ZMPS2 = FiniteMPS<QLTEN_Complex, fU1U1QN>;

template<typename TenElemT, typename QNT>
void RunTestTwoSiteAlgorithmCase(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const double benmrk_e0, const double precision
) {

  size_t start_flops = flop;
  Timer contract_timer("vmps");
  auto e0 = TwoSiteFiniteVMPS(mps, mpo, sweep_params);
  double elapsed_time = contract_timer.Elapsed();
  size_t end_flops = flop;
  double Gflops_s = (end_flops - start_flops) * 1.e-9 / elapsed_time;
  std::cout << "flops = " << end_flops - start_flops << std::endl;

  std::cout << "Gflops/s = " << Gflops_s << std::endl;

  EXPECT_NEAR(e0, benmrk_e0, precision);
  EXPECT_TRUE(mps.empty());
}

struct TestTwoSiteAlgorithmSpinlessFermion : public testing::Test {
  size_t N = 8;
  double t = 1.0;
  fU1QN qn0 = fU1QN(0);
  IndexT pb_out = IndexT({QNSctT(fU1QN(1), 1),
                          QNSctT(fU1QN(0), 1)},
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);
  ZQLTensor f = ZQLTensor({pb_in, pb_out});
  ZQLTensor c = ZQLTensor({pb_in, pb_out});
  ZQLTensor cdag = ZQLTensor({pb_in, pb_out});
  ZQLTensor id = ZQLTensor({pb_in, pb_out});
  ZSiteVec site_vec = ZSiteVec(N, pb_out);
  ZMPS zmps = ZMPS(site_vec);
  void SetUp(void) {
    f({0, 0}) = 1;
    f({1, 1}) = -1;
    c({0, 1}) = 1;
    cdag({1, 0}) = 1;
    id({0, 0}) = 1;
    id({1, 1}) = 1;
  }
};

TEST_F(TestTwoSiteAlgorithmSpinlessFermion, 1DFreeFermion) {
  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, fU1QN>(site_vec, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    zmpo_gen.AddTerm(t, cdag, i, c, i + 1);
    zmpo_gen.AddTerm(-t, c, i, cdag, i + 1);
  }
  auto zmpo = zmpo_gen.Gen();
  DirectStateInitZeroDivMps(zmps, {0, 1, 0, 1, 0, 1, 0, 1});
  auto sweep_params = FiniteVMPSSweepParams(
      5,
      8, 32, 1.0E-9,
      LanczosParams(1.0E-8, 20)
  );
  zmps.Dump(sweep_params.mps_path, true);
  RunTestTwoSiteAlgorithmCase(
      zmps, zmpo, sweep_params,
      -4.75877048314363371, 1.0E-8
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

// Test fermion models.
struct TestTwoSiteAlgorithmTjSystem2U1Symm : public testing::Test {
  size_t N = 4;
  double t = 3.0;
  double J = 1.0;
  fU1U1QN qn0 = fU1U1QN(0, 0);
  IndexT2 pb_out = IndexT2({QNSctT2(fU1U1QN(1, 1), 1),
                            QNSctT2(fU1U1QN(1, -1), 1),
                            QNSctT2(fU1U1QN(0, 0), 1)},
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
  }
};

TEST_F(TestTwoSiteAlgorithmTjSystem2U1Symm, 1DCase) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, fU1U1QN>(dsite_vec_4, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    dmpo_gen.AddTerm(t, dcdagup, i, dcup, i + 1);
    dmpo_gen.AddTerm(t, dcdagdn, i, dcdn, i + 1);
    dmpo_gen.AddTerm(-t, dcup, i, dcdagup, i + 1);
    dmpo_gen.AddTerm(-t, dcdn, i, dcdagdn, i + 1);
    dmpo_gen.AddTerm(J, dsz, i, dsz, i + 1);
    dmpo_gen.AddTerm(J / 2, dsp, i, dsm, i + 1);
    dmpo_gen.AddTerm(J / 2, dsm, i, dsp, i + 1);
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      11,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-8, 20)
  );
  DirectStateInitZeroDivMps(dmps, {2, 1, 2, 0});
  dmps.Dump(sweep_params.mps_path, true);
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -6.947478526233, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, fU1U1QN>(zsite_vec_4, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    zmpo_gen.AddTerm(t, zcdagup, i, zcup, i + 1);
    zmpo_gen.AddTerm(t, zcdagdn, i, zcdn, i + 1);
    zmpo_gen.AddTerm(-t, zcup, i, zcdagup, i + 1);
    zmpo_gen.AddTerm(-t, zcdn, i, zcdagdn, i + 1);
    zmpo_gen.AddTerm(J, zsz, i, zsz, i + 1);
    zmpo_gen.AddTerm(J / 2, zsp, i, zsm, i + 1);
    zmpo_gen.AddTerm(J / 2, zsm, i, zsp, i + 1);
  }
  auto zmpo = zmpo_gen.Gen();
  DirectStateInitZeroDivMps(zmps, {2, 1, 2, 0});
  zmps.Dump(sweep_params.mps_path, true);
  RunTestTwoSiteAlgorithmCase(
      zmps, zmpo, sweep_params,
      -6.947478526233, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

TEST_F(TestTwoSiteAlgorithmTjSystem2U1Symm, 2DCase) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, fU1U1QN>(dsite_vec_4, qn0);
  std::vector<std::pair<int, int>> nn_pairs = {
      std::make_pair(0, 1),
      std::make_pair(0, 2),
      std::make_pair(2, 3),
      std::make_pair(1, 3)};
  for (auto &p: nn_pairs) {
    dmpo_gen.AddTerm(t, dcdagup, p.first, dcup, p.second);
    dmpo_gen.AddTerm(t, dcdagdn, p.first, dcdn, p.second);
    dmpo_gen.AddTerm(-t, dcup, p.first, dcdagup, p.second);
    dmpo_gen.AddTerm(-t, dcdn, p.first, dcdagdn, p.second);
    dmpo_gen.AddTerm(J, dsz, p.first, dsz, p.second);
    dmpo_gen.AddTerm(J / 2, dsp, p.first, dsm, p.second);
    dmpo_gen.AddTerm(J / 2, dsm, p.first, dsp, p.second);
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      10,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-8, 20)
  );

  auto total_div = fU1U1QN(N - 2, 0);
  auto zero_div = fU1U1QN(0, 0);

  // Direct product state initialization.
  DirectStateInitZeroDivMps(dmps, {2, 0, 1, 2});
  dmps.Dump(sweep_params.mps_path, true);
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -8.868563739680, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, fU1U1QN>(zsite_vec_4, qn0);
  for (auto &p: nn_pairs) {
    zmpo_gen.AddTerm(t, zcdagup, p.first, zcup, p.second);
    zmpo_gen.AddTerm(t, zcdagdn, p.first, zcdn, p.second);
    zmpo_gen.AddTerm(-t, zcup, p.first, zcdagup, p.second);
    zmpo_gen.AddTerm(-t, zcdn, p.first, zcdagdn, p.second);
    zmpo_gen.AddTerm(J, zsz, p.first, zsz, p.second);
    zmpo_gen.AddTerm(J / 2, zsp, p.first, zsm, p.second);
    zmpo_gen.AddTerm(J / 2, zsm, p.first, zsp, p.second);
  }
  auto zmpo = zmpo_gen.Gen();
  DirectStateInitZeroDivMps(zmps, {2, 0, 1, 2});
  zmps.Dump(sweep_params.mps_path, true);
  RunTestTwoSiteAlgorithmCase(
      zmps, zmpo, sweep_params,
      -8.868563739680, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

struct TestTwoSiteAlgorithmTjSystem1fU1Symm : public testing::Test {
  fU1QN qn0 = fU1QN(0);
  IndexT pb_out = IndexT({
                             QNSctT(fU1QN(1), 2),
                             QNSctT(fU1QN(0), 1)},
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);
  ZQLTensor sz = ZQLTensor({pb_in, pb_out});
  ZQLTensor sp = ZQLTensor({pb_in, pb_out});
  ZQLTensor sm = ZQLTensor({pb_in, pb_out});
  ZQLTensor cup = ZQLTensor({pb_in, pb_out});
  ZQLTensor cdagup = ZQLTensor({pb_in, pb_out});
  ZQLTensor cdn = ZQLTensor({pb_in, pb_out});
  ZQLTensor cdagdn = ZQLTensor({pb_in, pb_out});
  ZQLTensor ntot = ZQLTensor({pb_in, pb_out});
  ZQLTensor zid = ZQLTensor({pb_in, pb_out});

  void SetUp(void) {
    sz({0, 0}) = 0.5;
    sz({1, 1}) = -0.5;
    sp({1, 0}) = 1;
    sm({0, 1}) = 1;
    cup({0, 2}) = 1;
    cdagup({2, 0}) = 1;
    cdn({1, 2}) = 1;
    cdagdn({2, 1}) = 1;
    ntot({0, 0}) = 1;
    ntot({1, 1}) = 1;
  }
};

/// < Total particle number must be even!
TEST_F(TestTwoSiteAlgorithmTjSystem1fU1Symm, RashbaTermCase) {
  double t = 3;
  double J = 1.0;
  double lamb = 0.03;
  auto ilamb = QLTEN_Complex(0, lamb);
  size_t Nx = 3;
  size_t Ny = 2;
  size_t Ntot = Nx * Ny;
  char BCx = 'p';
  char BCy = 'o';
  ZSiteVec site_vec(Ntot, pb_out);
  auto mpo_gen = MPOGenerator<QLTEN_Complex, fU1QN>(site_vec, qn0);
  for (int x = 0; x < Nx; ++x) {
    for (int y = 0; y < Ny; ++y) {

      if (!((BCx == 'o') && (x == Nx - 1))) {
        auto s0 = coors2idxSquare(x, y, Nx, Ny);
        auto s1 = coors2idxSquare((x + 1) % Nx, y, Nx, Ny);
        KeepOrder(s0, s1);
        std::cout << s0 << " " << s1 << std::endl;
        mpo_gen.AddTerm(t, cdagup, s0, cup, s1);
        mpo_gen.AddTerm(t, cdagdn, s0, cdn, s1);
        mpo_gen.AddTerm(-t, cup, s0, cdagup, s1);
        mpo_gen.AddTerm(-t, cdn, s0, cdagdn, s1);
        mpo_gen.AddTerm(J, sz, s0, sz, s1);
        mpo_gen.AddTerm(J / 2, sp, s0, sm, s1);
        mpo_gen.AddTerm(J / 2, sm, s0, sp, s1);
        mpo_gen.AddTerm(-J / 4, ntot, s0, ntot, s1);
        // SO term
        if (x != Nx - 1) {
          mpo_gen.AddTerm(-lamb, cdagup, s0, cdn, s1);
          mpo_gen.AddTerm(lamb, cup, s0, cdagdn, s1);
          mpo_gen.AddTerm(lamb, cdagdn, s0, cup, s1);
          mpo_gen.AddTerm(-lamb, cdn, s0, cdagup, s1);
        } else {    // At the boundary
          mpo_gen.AddTerm(lamb, cdn, s0, cdagup, s1);
          mpo_gen.AddTerm(-lamb, cdagdn, s0, cup, s1);
          mpo_gen.AddTerm(-lamb, cup, s0, cdagdn, s1);
          mpo_gen.AddTerm(lamb, cdagup, s0, cdn, s1);
        }
      }

      if (!((BCy == 'o') && (y == Ny - 1))) {
        auto s0 = coors2idxSquare(x, y, Nx, Ny);
        auto s2 = coors2idxSquare(x, (y + 1) % Ny, Nx, Ny);
        KeepOrder(s0, s2);
        std::cout << s0 << " " << s2 << std::endl;
        mpo_gen.AddTerm(t, cdagup, s0, cup, s2);
        mpo_gen.AddTerm(t, cdagdn, s0, cdn, s2);
        mpo_gen.AddTerm(-t, cup, s0, cdagup, s2);
        mpo_gen.AddTerm(-t, cdn, s0, cdagdn, s2);
        mpo_gen.AddTerm(J, sz, s0, sz, s2);
        mpo_gen.AddTerm(J / 2, sp, s0, sm, s2);
        mpo_gen.AddTerm(J / 2, sm, s0, sp, s2);
        mpo_gen.AddTerm(-J / 4, ntot, s0, ntot, s2);
        if (y != Ny - 1) {
          mpo_gen.AddTerm(-ilamb, cdagup, s0, cdn, s2);
          mpo_gen.AddTerm(-ilamb, cdagdn, s0, cup, s2);
          mpo_gen.AddTerm(-ilamb, cup, s0, cdagdn, s2);
          mpo_gen.AddTerm(-ilamb, cdn, s0, cdagup, s2);
        } else {    // At the boundary
          mpo_gen.AddTerm(ilamb, cup, s0, cdagdn, s2);
          mpo_gen.AddTerm(ilamb, cdn, s0, cdagup, s2);
          mpo_gen.AddTerm(ilamb, cdagup, s0, cdn, s2);
          mpo_gen.AddTerm(ilamb, cdagdn, s0, cup, s2);
        }
      }
    }
  }
  auto mpo = mpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      8,
      30, 30, 1.0E-4,
      LanczosParams(1.0E-14, 100)
  );
  auto mps = ZMPS(site_vec);
  DirectStateInitZeroDivMps(mps, {0, 1, 0, 2, 2, 1});
  mps.Dump(sweep_params.mps_path, true);
  RunTestTwoSiteAlgorithmCase(
      mps, mpo, sweep_params,
      -14.9151010027592150,
      1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

/*
struct TestTwoSiteAlgorithmHubbardSystem : public testing::Test {
  size_t Nx = 2;
  size_t Ny = 2;
  size_t N = Nx * Ny;
  double t0 = 1.0;
  double t1 = 0.5;
  double U = 2.0;

  fU1U1QN qn0 = fU1U1QN({QNCard("Nup", fU1QNVal(0)), QNCard("Ndn", fU1QNVal(0))});
  IndexT2 pb_out = IndexT2({
                               QNSctT2(fU1U1QN({QNCard("Nup", fU1QNVal(0)), QNCard("Ndn", fU1QNVal(0))}), 1),
                               QNSctT2(fU1U1QN({QNCard("Nup", fU1QNVal(1)), QNCard("Ndn", fU1QNVal(0))}), 1),
                               QNSctT2(fU1U1QN({QNCard("Nup", fU1QNVal(0)), QNCard("Ndn", fU1QNVal(1))}), 1),
                               QNSctT2(fU1U1QN({QNCard("Nup", fU1QNVal(1)), QNCard("Ndn", fU1QNVal(1))}), 1)},
                           TenIndexDirType::OUT
  );
  IndexT2 pb_in = InverseIndex(pb_out);
  DSiteVec2 dsite_vec = DSiteVec2(N, pb_out);
  ZSiteVec2 zsite_vec = ZSiteVec2(N, pb_out);

  DQLTensor2 df = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dnupdn = DQLTensor2({pb_in, pb_out});    // n_up*n_dn
  DQLTensor2 dadagupf = DQLTensor2({pb_in, pb_out});    // a^+_up*f
  DQLTensor2 daup = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dadagdn = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dfadn = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dnaupf = DQLTensor2({pb_in, pb_out});    // -a_up*f
  DQLTensor2 dadagup = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dnadn = DQLTensor2({pb_in, pb_out});
  DQLTensor2 dfadagdn = DQLTensor2({pb_in, pb_out});    // f*a^+_dn
  DMPS2 dmps = DMPS2(dsite_vec);

  ZQLTensor2 zf = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 znupdn = ZQLTensor2({pb_in, pb_out});    // n_up*n_dn
  ZQLTensor2 zadagupf = ZQLTensor2({pb_in, pb_out});    // a^+_up*f
  ZQLTensor2 zaup = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 zadagdn = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 zfadn = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 znaupf = ZQLTensor2({pb_in, pb_out});    // -a_up*f
  ZQLTensor2 zadagup = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 znadn = ZQLTensor2({pb_in, pb_out});
  ZQLTensor2 zfadagdn = ZQLTensor2({pb_in, pb_out});    // f*a^+_dn
  ZMPS2 zmps = ZMPS2(zsite_vec);

  void SetUp(void) {
    df({0, 0}) = 1;
    df({1, 1}) = -1;
    df({2, 2}) = -1;
    df({3, 3}) = 1;

    dnupdn({3, 3}) = 1;

    dadagupf({1, 0}) = 1;
    dadagupf({3, 2}) = -1;
    daup({0, 1}) = 1;
    daup({2, 3}) = 1;
    dadagdn({2, 0}) = 1;
    dadagdn({3, 1}) = 1;
    dfadn({0, 2}) = 1;
    dfadn({1, 3}) = -1;
    dnaupf({0, 1}) = 1;
    dnaupf({2, 3}) = -1;
    dadagup({1, 0}) = 1;
    dadagup({3, 2}) = 1;
    dnadn({0, 2}) = -1;
    dnadn({1, 3}) = -1;
    dfadagdn({2, 0}) = -1;
    dfadagdn({3, 1}) = 1;

    zf({0, 0}) = 1;
    zf({1, 1}) = -1;
    zf({2, 2}) = -1;
    zf({3, 3}) = 1;

    znupdn({3, 3}) = 1;

    zadagupf({1, 0}) = 1;
    zadagupf({3, 2}) = -1;
    zaup({0, 1}) = 1;
    zaup({2, 3}) = 1;
    zadagdn({2, 0}) = 1;
    zadagdn({3, 1}) = 1;
    zfadn({0, 2}) = 1;
    zfadn({1, 3}) = -1;
    znaupf({0, 1}) = 1;
    znaupf({2, 3}) = -1;
    zadagup({1, 0}) = 1;
    zadagup({3, 2}) = 1;
    znadn({0, 2}) = -1;
    znadn({1, 3}) = -1;
    zfadagdn({2, 0}) = -1;
    zfadagdn({3, 1}) = 1;
  }
};


TEST_F(TestTwoSiteAlgorithmHubbardSystem, 2Dcase) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, fU1U1QN>(dsite_vec, qn0);
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      auto s0 = coors2idxSquare(i, j, Nx, Ny);
      dmpo_gen.AddTerm(U, dnupdn, s0);

      if (i != Nx - 1) {
        auto s1 = coors2idxSquare(i + 1, j, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        dmpo_gen.AddTerm(1, -t0 * dadagupf, s0, daup, s1, df);
        dmpo_gen.AddTerm(1, -t0 * dadagdn, s0, dfadn, s1, df);
        dmpo_gen.AddTerm(1, dnaupf, s0, -t0 * dadagup, s1, df);
        dmpo_gen.AddTerm(1, dnadn, s0, -t0 * dfadagdn, s1, df);
      }
      if (j != Ny - 1) {
        auto s1 = coors2idxSquare(i, j + 1, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        dmpo_gen.AddTerm(1, -t0 * dadagupf, s0, daup, s1, df);
        dmpo_gen.AddTerm(1, -t0 * dadagdn, s0, dfadn, s1, df);
        dmpo_gen.AddTerm(1, dnaupf, s0, -t0 * dadagup, s1, df);
        dmpo_gen.AddTerm(1, dnadn, s0, -t0 * dfadagdn, s1, df);
      }

      if (j != Ny - 1) {
        if (i != 0) {
          auto s2 = coors2idxSquare(i - 1, j + 1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          dmpo_gen.AddTerm(1, -t1 * dadagupf, temp_s0, daup, s2, df);
          dmpo_gen.AddTerm(1, -t1 * dadagdn, temp_s0, dfadn, s2, df);
          dmpo_gen.AddTerm(1, dnaupf, temp_s0, -t1 * dadagup, s2, df);
          dmpo_gen.AddTerm(1, dnadn, temp_s0, -t1 * dfadagdn, s2, df);
        }
        if (i != Nx - 1) {
          auto s2 = coors2idxSquare(i + 1, j + 1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          dmpo_gen.AddTerm(1, -t1 * dadagupf, temp_s0, daup, s2, df);
          dmpo_gen.AddTerm(1, -t1 * dadagdn, temp_s0, dfadn, s2, df);
          dmpo_gen.AddTerm(1, dnaupf, temp_s0, -t1 * dadagup, s2, df);
          dmpo_gen.AddTerm(1, dnadn, temp_s0, -t1 * dfadagdn, s2, df);
        }
      }
    }
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      10,
      16, 16, 1.0E-9,
      LanczosParams(1.0E-8, 20)
  );
  auto qn0 = fU1U1QN({QNCard("Nup", fU1QNVal(0)), QNCard("Ndn", fU1QNVal(0))});
  std::vector<size_t> stat_labs(N);
  for (size_t i = 0; i < N; ++i) { stat_labs[i] = (i % 2 == 0 ? 1 : 2); }
  DirectStateInitZeroDivMps(dmps, stat_labs);
  dmps.Dump(sweep_params.mps_path, true);
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -2.828427124746, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, fU1U1QN>(zsite_vec, qn0);
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      auto s0 = coors2idxSquare(i, j, Nx, Ny);
      zmpo_gen.AddTerm(U, znupdn, s0);

      if (i != Nx - 1) {
        auto s1 = coors2idxSquare(i + 1, j, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        zmpo_gen.AddTerm(-t0, zadagupf, s0, zaup, s1, zf);
        zmpo_gen.AddTerm(-t0, zadagdn, s0, zfadn, s1, zf);
        zmpo_gen.AddTerm(-t0, znaupf, s0, zadagup, s1, zf);
        zmpo_gen.AddTerm(-t0, znadn, s0, zfadagdn, s1, zf);
      }
      if (j != Ny - 1) {
        auto s1 = coors2idxSquare(i, j + 1, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        zmpo_gen.AddTerm(-t0, zadagupf, s0, zaup, s1, zf);
        zmpo_gen.AddTerm(-t0, zadagdn, s0, zfadn, s1, zf);
        zmpo_gen.AddTerm(-t0, znaupf, s0, zadagup, s1, zf);
        zmpo_gen.AddTerm(-t0, znadn, s0, zfadagdn, s1, zf);
      }

      if (j != Ny - 1) {
        if (i != 0) {
          auto s2 = coors2idxSquare(i - 1, j + 1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          zmpo_gen.AddTerm(-t1, zadagupf, temp_s0, zaup, s2, zf);
          zmpo_gen.AddTerm(-t1, zadagdn, temp_s0, zfadn, s2, zf);
          zmpo_gen.AddTerm(-t1, znaupf, temp_s0, zadagup, s2, zf);
          zmpo_gen.AddTerm(-t1, znadn, temp_s0, zfadagdn, s2, zf);
        }
        if (i != Nx - 1) {
          auto s2 = coors2idxSquare(i + 1, j + 1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          zmpo_gen.AddTerm(-t1, zadagupf, temp_s0, zaup, s2, zf);
          zmpo_gen.AddTerm(-t1, zadagdn, temp_s0, zfadn, s2, zf);
          zmpo_gen.AddTerm(-t1, znaupf, temp_s0, zadagup, s2, zf);
          zmpo_gen.AddTerm(-t1, znadn, temp_s0, zfadagdn, s2, zf);
        }
      }
    }
  }
  auto zmpo = zmpo_gen.Gen();
  DirectStateInitZeroDivMps(zmps, stat_labs);
  zmps.Dump(sweep_params.mps_path, true);
  RunTestTwoSiteAlgorithmCase(
      zmps, zmpo, sweep_params,
      -2.828427124746, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}


// Test non-uniform local Hilbert spaces system.
// Kondo insulator, ref 10.1103/PhysRevB.97.245119,
struct TestKondoInsulatorSystem : public testing::Test {
  size_t Nx = 4;
  size_t N = 2 * Nx;
  double t = 0.25;
  double Jk = 1.0;
  double Jz = 0.5;

  fU1QN qn0 = fU1QN({QNCard("Sz", fU1QNVal(0))});
  IndexT pb_outE = IndexT(
      {QNSctT(fU1QN({QNCard("Sz", fU1QNVal(0))}), 4)},
      TenIndexDirType::OUT
  );    // extended electron
  IndexT pb_outL = IndexT(
      {QNSctT(fU1QN({QNCard("Sz", fU1QNVal(0))}), 2)},
      TenIndexDirType::OUT
  );    // localized electron
  IndexT pb_inE = InverseIndex(pb_outE);
  IndexT pb_inL = InverseIndex(pb_outL);

  DQLTensor sz = DQLTensor({pb_inE, pb_outE});
  DQLTensor sp = DQLTensor({pb_inE, pb_outE});
  DQLTensor sm = DQLTensor({pb_inE, pb_outE});
  DQLTensor bupcF = DQLTensor({pb_inE, pb_outE});
  DQLTensor bupaF = DQLTensor({pb_inE, pb_outE});
  DQLTensor Fbdnc = DQLTensor({pb_inE, pb_outE});
  DQLTensor Fbdna = DQLTensor({pb_inE, pb_outE});
  DQLTensor bupc = DQLTensor({pb_inE, pb_outE});
  DQLTensor bupa = DQLTensor({pb_inE, pb_outE});
  DQLTensor bdnc = DQLTensor({pb_inE, pb_outE});
  DQLTensor bdna = DQLTensor({pb_inE, pb_outE});

  DQLTensor Sz = DQLTensor({pb_inL, pb_outL});
  DQLTensor Sp = DQLTensor({pb_inL, pb_outL});
  DQLTensor Sm = DQLTensor({pb_inL, pb_outL});

  std::vector<IndexT> pb_set = std::vector<IndexT>(N);

  void SetUp(void) {
    sz({0, 0}) = 0.5;
    sz({1, 1}) = -0.5;
    sp({0, 1}) = 1.0;
    sm({1, 0}) = 1.0;
    bupcF({2, 1}) = -1;
    bupcF({0, 3}) = 1;
    Fbdnc({2, 0}) = 1;
    Fbdnc({1, 3}) = -1;
    bupaF({1, 2}) = 1;
    bupaF({3, 0}) = -1;
    Fbdna({0, 2}) = -1;
    Fbdna({3, 1}) = 1;

    bupc({2, 1}) = 1;
    bupc({0, 3}) = 1;
    bdnc({2, 0}) = 1;
    bdnc({1, 3}) = 1;
    bupa({1, 2}) = 1;
    bupa({3, 0}) = 1;
    bdna({0, 2}) = 1;
    bdna({3, 1}) = 1;

    Sz({0, 0}) = 0.5;
    Sz({1, 1}) = -0.5;
    Sp({0, 1}) = 1.0;
    Sm({1, 0}) = 1.0;

    for (size_t i = 0; i < N; ++i) {
      if (i % 2 == 0) pb_set[i] = pb_outE;   // even site is extended electron
      if (i % 2 == 1) pb_set[i] = pb_outL;   // odd site is localized electron
    }
  }
};


TEST_F(TestKondoInsulatorSystem, doublechain) {
  auto dsite_vec = DSiteVec(pb_set);
  auto dmps = DMPS(dsite_vec);
  auto dmpo_gen = MPOGenerator<QLTEN_Double, fU1QN>(dsite_vec, qn0);
  for (size_t i = 0; i < N - 2; i = i + 2) {
    dmpo_gen.AddTerm(-t, bupcF, i, bupa, i + 2);
    dmpo_gen.AddTerm(-t, bdnc, i, Fbdna, i + 2);
    dmpo_gen.AddTerm(t, bupaF, i, bupc, i + 2);
    dmpo_gen.AddTerm(t, bdna, i, Fbdnc, i + 2);
    dmpo_gen.AddTerm(Jz, Sz, i + 1, Sz, i + 3);
  }
  for (size_t i = 0; i < N; i = i + 2) {
    dmpo_gen.AddTerm(Jk, sz, i, Sz, i + 1);
    dmpo_gen.AddTerm(Jk / 2, sp, i, Sm, i + 1);
    dmpo_gen.AddTerm(Jk / 2, sm, i, Sp, i + 1);
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      5,
      64, 64, 1.0E-9,
      LanczosParams(1.0E-8, 20)
  );

  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitZeroDivMps(dmps, stat_labs);
  dmps.Dump(sweep_params.mps_path, true);

  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo,
      sweep_params,
      -3.180025784229132, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

*/

//// Test noised tow-site vMPS algorithm.
//// Electron-phonon interaction Holstein chain, ref 10.1103/PhysRevB.57.6376
//struct TestHolsteinChain : public testing::Test {
//int L = 4;           // The number of electron
//int Np = 3;          // per electron has Np pseudosite
//double t = 1;         // electron hopping
//double g = 1;         // electron-phonon interaction
//double U = 8;         // Hubbard U
//double omega = 5;     // phonon on-site potential
//int N = (1+Np)*L;     // The length of the mps/mpo

//QN qn0 = QN({ QNNameVal("Nf", 0), QNNameVal("Sz", 0) });
////Fermion(electron)
//Index pb_outF = Index({
//QNSector(
//QN( {QNNameVal("Nf", 2), QNNameVal("Sz",  0)}),
//1
//),
//QNSector(
//QN( {QNNameVal("Nf", 1), QNNameVal("Sz",  1)} ),
//1
//),
//QNSector(
//QN( {QNNameVal("Nf", 1), QNNameVal("Sz", -1)} ),
//1
//),
//QNSector(
//QN( {QNNameVal("Nf", 0), QNNameVal("Sz",  0)} ),
//1
//)
//}, OUT);
//Index pb_inF = InverseIndex(pb_outF);
////Boson(Phonon)
//Index pb_outB = Index({
//QNSector(
//QN( {QNNameVal("Nf", 0), QNNameVal("Sz",  0)} ),
//2
//)
//},OUT);
//Index pb_inB = InverseIndex(pb_outB);
//DQLTensor nf = DQLTensor({pb_inF, pb_outF}); //fermion number
//DQLTensor bupcF =DQLTensor({pb_inF,pb_outF});
//DQLTensor bupaF = DQLTensor({pb_inF,pb_outF});
//DQLTensor Fbdnc = DQLTensor({pb_inF,pb_outF});
//DQLTensor Fbdna = DQLTensor({pb_inF,pb_outF});
//DQLTensor bupc =DQLTensor({pb_inF,pb_outF});
//DQLTensor bupa = DQLTensor({pb_inF,pb_outF});
//DQLTensor bdnc = DQLTensor({pb_inF,pb_outF});
//DQLTensor bdna = DQLTensor({pb_inF,pb_outF});
//DQLTensor Uterm = DQLTensor({pb_inF,pb_outF}); // Hubbard Uterm, nup*ndown

//DQLTensor a = DQLTensor({pb_inB, pb_outB}); //bosonic annihilation
//DQLTensor adag = DQLTensor({pb_inB, pb_outB}); //bosonic creation
//DQLTensor n_a =  DQLTensor({pb_inB, pb_outB}); // the number of phonon
//DQLTensor idB =  DQLTensor({pb_inB, pb_outB}); // bosonic identity
//DQLTensor &P1 = n_a;
//DQLTensor P0 = DQLTensor({pb_inB, pb_outB});

//DTenPtrVec dmps    = DTenPtrVec(N);
//std::vector<Index> pb_set = std::vector<Index>(N);


//void SetUp(void) {
//nf({0,0}) = 2;  nf({1,1}) = 1;  nf({2,2}) = 1;

//bupcF({0,2}) = -1;  bupcF({1,3}) = 1;
//Fbdnc({0,1}) = 1;   Fbdnc({2,3}) = -1;
//bupaF({2,0}) = 1;   bupaF({3,1}) = -1;
//Fbdna({1,0}) = -1;  Fbdna({3,2}) = 1;

//bupc({0,2}) = 1;    bupc({1,3}) = 1;
//bdnc({0,1}) = 1;    bdnc({2,3}) = 1;
//bupa({2,0}) = 1;    bupa({3,1}) = 1;
//bdna({1,0}) = 1;    bdna({3,2}) = 1;

//Uterm({0,0}) = 1;


//adag({0,1}) = 1;
//a({1,0}) = 1;
//n_a({0,0}) = 1;
//idB({0,0}) = 1; idB({1,1}) = 1;
//P0 = idB+(-n_a);
//for(int i =0;i < N; ++i){
//if(i%(Np+1)==0) pb_set[i] = pb_outF; // even site is fermion
//else pb_set[i] = pb_outB; // odd site is boson
//}
//}
//};

//TEST_F(TestHolsteinChain, holsteinchain) {
//SiteVec site_vec(pb_set);
//auto dmpo_gen = MPOGenerator<QLTEN_Double>(site_vec, qn0);
//for (int i = 0; i < N-Np-1; i=i+Np+1) {
//dmpo_gen.AddTerm(-t, bupcF, i, bupa, i+Np+1);
//dmpo_gen.AddTerm(-t, bdnc, i, Fbdna, i+Np+1);
//dmpo_gen.AddTerm( t, bupaF, i, bupc, i+Np+1);
//dmpo_gen.AddTerm( t, bdna, i, Fbdnc, i+Np+1);
//}
//for (int i = 0; i < N; i=i+Np+1) {
//dmpo_gen.AddTerm(U, Uterm, i);
//for(int j = 0; j < Np; ++j) {
//dmpo_gen.AddTerm(omega, (double)(pow(2,j))*n_a, i+j+1);
//}
//dmpo_gen.AddTerm(2*g, {nf, a, a, adag},{i, i+1, i+2, i+3});
//dmpo_gen.AddTerm(2*g, {nf, adag, adag, a},{i, i+1, i+2, i+3});
//dmpo_gen.AddTerm(g,   {nf, a, adag, sqrt(2)*P0+sqrt(6)*P1},{i,i+1,i+2,i+3});
//dmpo_gen.AddTerm(g,   {nf, adag, a, sqrt(2)*P0+sqrt(6)*P1},{i,i+1,i+2,i+3});
//dmpo_gen.AddTerm(g,  {nf,a+adag,P1, sqrt(3)*P0+sqrt(7)*P1},{i,i+1,i+2,i+3});
//dmpo_gen.AddTerm(g,   {nf,a+adag,P0, P0+sqrt(5)*P1},{i,i+1,i+2,i+3});
//}

//auto dmpo = dmpo_gen.Gen();
//std::vector<long> stat_labs(N,0);
//int qn_label = 1;
//for (int i = 0; i < N; i=i+Np+1) {
//stat_labs[i] = qn_label;
//qn_label=3-qn_label;
//}
//DirectStateInitZeroDivMps(dmps, stat_labs, pb_set, qn0);
//auto sweep_params = FiniteVMPSSweepParams(
//5,
//256, 256, 1.0E-10,
//true,
//kTwoSiteAlgoWorkflowInitial,
//LanczosParams(1.0E-10, 40)
//);
//std::vector<double> noise = {0.1,0.1,0.01,0.001};
//RunTestTwoSiteAlgorithmNoiseCase(
//dmps, dmpo,
//sweep_params, noise,
//-1.9363605088186260 , 1.0E-8
//);
//}
