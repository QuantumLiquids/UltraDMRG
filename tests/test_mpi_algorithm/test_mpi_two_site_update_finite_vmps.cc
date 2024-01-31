// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-26
*
* Description: GraceQ/mps2 project. Unittest for MPI two sites algorithm.
*/

#include "gtest/gtest.h"
#include "boost/mpi.hpp"

#include "qlmps/qlmps.h"
#include "qlten/qlten.h"
#include "../testing_utils.h"


using namespace qlmps;
using namespace qlten;

using U1QN = QN<U1QNVal>;
using U1U1QN = QN<U1QNVal, U1QNVal>;

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

boost::mpi::environment env;

template<typename TenElemT, typename QNT>
void RunTestMPIVMPSCase(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const double benmrk_e0, const double precision,
    mpi::communicator &world
) {
  size_t start_flops = flop;
  Timer contract_timer("dmrg");
  auto e0 = TwoSiteFiniteVMPS(mps, mpo, sweep_params, world);
  double elapsed_time = contract_timer.Elapsed();
  size_t end_flops = flop;
  double Gflops_s = (end_flops - start_flops) * 1.e-9 / elapsed_time;
  std::cout << "flops = " << end_flops - start_flops << std::endl;
  if (world.rank() == kMasterRank) {
    EXPECT_NEAR(e0, benmrk_e0, precision);
    EXPECT_TRUE(mps.empty());
  }
}


// Test spin systems
struct TestVMPSSpinSystem : public testing::Test {
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

  }
};

TEST_F(TestVMPSSpinSystem, 1DIsing) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec_6, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    dmpo_gen.AddTerm(1, {dsz, dsz}, {i, i + 1});
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      1, 10, 1.0E-5,
      LanczosParams(1.0E-7)
  );
  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(dmps, stat_labs);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    dmps.Dump(sweep_params.mps_path, true);
  }

  RunTestMPIVMPSCase(dmps, dmpo, sweep_params, -0.25 * (N - 1), 1.0E-10, world);

  dmps.Load(sweep_params.mps_path);
  MeasureOneSiteOp(dmps, dsz, "dsz");
  std::vector<std::vector<size_t>> sites_set;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (i < j) { sites_set.push_back({i, j}); }
    }
  }
  MeasureTwoSiteOp(dmps, {dsz, dsz}, did, sites_set, "dszdsz");
  dmps.clear();



  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec_6, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    zmpo_gen.AddTerm(1, {zsz, zsz}, {i, i + 1});
  }
  auto zmpo = zmpo_gen.Gen();
  sweep_params = FiniteVMPSSweepParams(
      4,
      1, 10, 1.0E-5,
      LanczosParams(1.0E-7)
  );
  DirectStateInitMps(zmps, stat_labs);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    zmps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(zmps, zmpo, sweep_params, -0.25 * (N - 1), 1.0E-10, world);

  zmps.Load(sweep_params.mps_path);
  MeasureOneSiteOp(zmps, zsz, "zsz");
  MeasureTwoSiteOp(zmps, {zsz, zsz}, zid, sites_set, "zszzsz");
  zmps.clear();

  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

TEST_F(TestVMPSSpinSystem, 1DHeisenberg) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec_6, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    dmpo_gen.AddTerm(1, {dsz, dsz}, {i, i + 1});
    dmpo_gen.AddTerm(0.5, {dsp, dsm}, {i, i + 1});
    dmpo_gen.AddTerm(0.5, {dsm, dsp}, {i, i + 1});
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-7)
  );
  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(dmps, stat_labs);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    dmps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      dmps, dmpo, sweep_params,
      -2.493577133888, 1.0E-12,
      world
  );

  // Continue simulation test
  dmps.clear();
  RunTestMPIVMPSCase(
      dmps, dmpo, sweep_params,
      -2.493577133888, 1.0E-12,
      world
  );

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec_6, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    zmpo_gen.AddTerm(1, {zsz, zsz}, {i, i + 1});
    zmpo_gen.AddTerm(0.5, {zsp, zsm}, {i, i + 1});
    zmpo_gen.AddTerm(0.5, {zsm, zsp}, {i, i + 1});
  }
  auto zmpo = zmpo_gen.Gen();

  sweep_params = FiniteVMPSSweepParams(
      4,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-7)
  );
  DirectStateInitMps(zmps, stat_labs);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    zmps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      zmps, zmpo, sweep_params,
      -2.493577133888, 1.0E-12,
      world
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

TEST_F(TestVMPSSpinSystem, 2DHeisenberg) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec_6, qn0);
  std::vector<std::pair<size_t, size_t>> nn_pairs = {
      std::make_pair(0, 1),
      std::make_pair(0, 2),
      std::make_pair(1, 3),
      std::make_pair(2, 3),
      std::make_pair(2, 4),
      std::make_pair(3, 5),
      std::make_pair(4, 5)
  };
  for (auto &p: nn_pairs) {
    dmpo_gen.AddTerm(1, {dsz, dsz}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5, {dsp, dsm}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5, {dsm, dsp}, {p.first, p.second});
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-7)
  );

  // Test direct product state initialization.
  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(dmps, stat_labs);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    dmps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      dmps, dmpo, sweep_params,
      -3.129385241572, 1.0E-12,
      world
  );


  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec_6, qn0);
  for (auto &p: nn_pairs) {
    zmpo_gen.AddTerm(1, {zsz, zsz}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5, {zsp, zsm}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5, {zsm, zsp}, {p.first, p.second});
  }
  auto zmpo = zmpo_gen.Gen();

  sweep_params = FiniteVMPSSweepParams(
      4,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-7)
  );
  DirectStateInitMps(zmps, stat_labs);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    zmps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      zmps, zmpo, sweep_params,
      -3.129385241572, 1.0E-12,
      world
  );
}

TEST_F(TestVMPSSpinSystem, 2DKitaevSimpleCase) {
  size_t Nx = 4;
  size_t Ny = 2;
  size_t N1 = Nx * Ny;
  DSiteVec dsite_vec(N1, pb_out);
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec, qn0);
  for (int x = 0; x < Nx; ++x) {
    for (int y = 0; y < Ny; ++y) {
      if (x % 2 == 1) {
        auto s0 = coors2idxHoneycomb(x, y, Nx, Ny);
        auto s1 = coors2idxHoneycomb(x - 1, y + 1, Nx, Ny);
        KeepOrder(s0, s1);
        dmpo_gen.AddTerm(1, {dsz, dsz}, {s0, s1});
      }
    }
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      8, 8, 1.0E-8,
      LanczosParams(1.0E-10)
  );
  // Test extend direct product state random initialization.
  std::vector<size_t> stat_labs1, stat_labs2;
  for (size_t i = 0; i < N1; ++i) {
    stat_labs1.push_back(i % 2);
    stat_labs2.push_back((i + 1) % 2);
  }
  auto dmps_8sites = DMPS(dsite_vec);
  ExtendDirectRandomInitMps(dmps_8sites, {stat_labs1, stat_labs2}, 2);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    dmps_8sites.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      dmps_8sites, dmpo, sweep_params,
      -1.0, 1.0E-12,
      world
  );

  // Complex Hamiltonian
  ZSiteVec zsite_vec(N1, pb_out);
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec, qn0);
  for (int x = 0; x < Nx; ++x) {
    for (int y = 0; y < Ny; ++y) {
      if (x % 2 == 1) {
        auto s0 = coors2idxHoneycomb(x, y, Nx, Ny);
        auto s1 = coors2idxHoneycomb(x - 1, y + 1, Nx, Ny);
        KeepOrder(s0, s1);
        zmpo_gen.AddTerm(1, {zsz, zsz}, {s0, s1});
      }
    }
  }
  auto zmpo = zmpo_gen.Gen();
  auto zmps_8sites = ZMPS(zsite_vec);
  ExtendDirectRandomInitMps(zmps_8sites, {stat_labs1, stat_labs2}, 2);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    zmps_8sites.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      zmps_8sites, zmpo, sweep_params,
      -1.0, 1.0E-12, world);
}

TEST(TestTwoSiteAlgorithmNoSymmetrySpinSystem, 2DKitaevComplexCase) {
  boost::mpi::communicator world;
  ::testing::TestEventListeners &listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  using TenElemType = QLTEN_Complex;
  using Tensor = QLTensor<TenElemType, U1QN>;
  //-------------Set quantum numbers-----------------
  auto zero_div = U1QN({QNCard("N", U1QNVal(0))});
  auto idx_out = IndexT(
      {QNSctT(U1QN({QNCard("N", U1QNVal(1))}), 2)},
      TenIndexDirType::OUT
  );
  auto idx_in = InverseIndex(idx_out);
  //--------------Single site operators-----------------
  // define the structure of operators
  auto sz = Tensor({idx_in, idx_out});
  auto sx = Tensor({idx_in, idx_out});
  auto sy = Tensor({idx_in, idx_out});
  auto id = Tensor({idx_in, idx_out});
  // define the contents of operators
  sz({0, 0}) = QLTEN_Complex(0.5, 0);
  sz({1, 1}) = QLTEN_Complex(-0.5, 0);
  sx({0, 1}) = QLTEN_Complex(0.5, 0);
  sx({1, 0}) = QLTEN_Complex(0.5, 0);
  sy({0, 1}) = QLTEN_Complex(0, -0.5);
  sy({1, 0}) = QLTEN_Complex(0, 0.5);
  id({0, 0}) = QLTEN_Complex(1, 0);
  id({1, 1}) = QLTEN_Complex(1, 0);
  //---------------Generate the MPO-----------------
  double J = -1.0;
  double K = 1.0;
  double Gm = 0.1;
  double h = 0.1;
  size_t Nx = 3, Ny = 4;
  size_t N = Nx * Ny;
  ZSiteVec site_vec(N, idx_out);
  auto mpo_gen = MPOGenerator<TenElemType, U1QN>(site_vec, zero_div);
  // H =   J * \Sigma_{<ij>} S_i*S_j
  //       K * \Sigma_{<ij>,c-link} Sc_i*Sc_j
  //		 Gm * \Sigma_{<ij>,c-link} Sa_i*Sb_j + Sb_i*Sa_j
  //     - h * \Sigma_i{S^z}
  for (int x = 0; x < Nx; ++x) {
    for (int y = 0; y < Ny; ++y) {
      // use the configuration '/|\' to traverse the square lattice
      // single site operator
      auto site0_num = coors2idx(x, y, Nx, Ny);
      mpo_gen.AddTerm(-h, sz, site0_num);
      mpo_gen.AddTerm(-h, sx, site0_num);
      mpo_gen.AddTerm(-h, sy, site0_num);
      // the '/' part: x-link
      // note that x and y start from 0
      if (y % 2 == 0) {
        auto site0_num = coors2idx(x, y, Nx, Ny);
        auto site1_num = coors2idx(x, y + 1, Nx, Ny);
        std::cout << site0_num << " " << site1_num << std::endl;
        mpo_gen.AddTerm(J, sz, site0_num, sz, site1_num);
        mpo_gen.AddTerm(J, sx, site0_num, sx, site1_num);
        mpo_gen.AddTerm(J, sy, site0_num, sy, site1_num);
        mpo_gen.AddTerm(K, sx, site0_num, sx, site1_num);
        mpo_gen.AddTerm(Gm, sz, site0_num, sy, site1_num);
        mpo_gen.AddTerm(Gm, sy, site0_num, sz, site1_num);
      }
      // the '|' part: z-link
      if (y % 2 == 1) {
        auto site0_num = coors2idx(x, y, Nx, Ny);
        auto site1_num = coors2idx(x, (y + 1) % Ny, Nx, Ny);
        KeepOrder(site0_num, site1_num);
        std::cout << site0_num << " " << site1_num << std::endl;
        mpo_gen.AddTerm(J, sz, site0_num, sz, site1_num);
        mpo_gen.AddTerm(J, sx, site0_num, sx, site1_num);
        mpo_gen.AddTerm(J, sy, site0_num, sy, site1_num);
        mpo_gen.AddTerm(K, sz, site0_num, sz, site1_num);
        mpo_gen.AddTerm(Gm, sx, site0_num, sy, site1_num);
        mpo_gen.AddTerm(Gm, sy, site0_num, sx, site1_num);
      }
      // the '\' part: y-link
      // if (y % 2 == 1) {																	// torus
      if ((y % 2 == 1) && (x != Nx - 1)) {
        auto site0_num = coors2idx(x, y, Nx, Ny);
        auto site1_num = coors2idx(x + 1, y - 1, Nx, Ny);                        // cylinder
        KeepOrder(site0_num, site1_num);
        std::cout << site0_num << " " << site1_num << std::endl;
        mpo_gen.AddTerm(J, sz, site0_num, sz, site1_num);
        mpo_gen.AddTerm(J, sx, site0_num, sx, site1_num);
        mpo_gen.AddTerm(J, sy, site0_num, sy, site1_num);
        mpo_gen.AddTerm(K, sy, site0_num, sy, site1_num);
        mpo_gen.AddTerm(Gm, sz, site0_num, sx, site1_num);
        mpo_gen.AddTerm(Gm, sx, site0_num, sz, site1_num);
      }
    }
  }
  auto mpo = mpo_gen.Gen();

  FiniteMPS<TenElemType, U1QN> mps(site_vec);
  std::vector<size_t> stat_labs(N);
  auto was_up = false;
  for (size_t i = 0; i < N; ++i) {
    if (was_up) {
      stat_labs[i] = 1;
      was_up = false;
    } else if (!was_up) {
      stat_labs[i] = 0;
      was_up = true;
    }
  }
  auto sweep_params = FiniteVMPSSweepParams(
      4,
      60, 60, 1.0E-4,
      LanczosParams(1.0E-10)
  );
  DirectStateInitMps(mps, stat_labs);
  if (world.rank() == 0) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    mps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(mps, mpo, sweep_params, -4.57509167674, 2.0E-10, world);
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

TEST_F(TestTwoSiteAlgorithmTjSystem2U1Symm, 1DCase) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1U1QN>(dsite_vec_4, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    dmpo_gen.AddTerm(-t, dcdagup, i, dcup, i + 1, df);
    dmpo_gen.AddTerm(-t, dcdagdn, i, dcdn, i + 1, df);
    dmpo_gen.AddTerm(-t, dcup, i, dcdagup, i + 1, df);
    dmpo_gen.AddTerm(-t, dcdn, i, dcdagdn, i + 1, df);
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
  DirectStateInitMps(dmps, {2, 1, 2, 0});
  if (world.rank() == 0) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    dmps.Dump(sweep_params.mps_path, true);
  }

  RunTestMPIVMPSCase(
      dmps, dmpo, sweep_params,
      -6.947478526233, 1.0E-10,
      world
  );


  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1U1QN>(zsite_vec_4, qn0);
  for (size_t i = 0; i < N - 1; ++i) {
    zmpo_gen.AddTerm(-t, zcdagup, i, zcup, i + 1, zf);
    zmpo_gen.AddTerm(-t, zcdagdn, i, zcdn, i + 1, zf);
    zmpo_gen.AddTerm(-t, zcup, i, zcdagup, i + 1, zf);
    zmpo_gen.AddTerm(-t, zcdn, i, zcdagdn, i + 1, zf);
    zmpo_gen.AddTerm(J, zsz, i, zsz, i + 1);
    zmpo_gen.AddTerm(J / 2, zsp, i, zsm, i + 1);
    zmpo_gen.AddTerm(J / 2, zsm, i, zsp, i + 1);
  }
  auto zmpo = zmpo_gen.Gen();
  DirectStateInitMps(zmps, {2, 1, 2, 0});
  if (world.rank() == 0) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    zmps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      zmps, zmpo, sweep_params,
      -6.947478526233, 1.0E-10,
      world
  );
}

TEST_F(TestTwoSiteAlgorithmTjSystem2U1Symm, 2DCase) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1U1QN>(dsite_vec_4, qn0);
  std::vector<std::pair<int, int>> nn_pairs = {
      std::make_pair(0, 1),
      std::make_pair(0, 2),
      std::make_pair(2, 3),
      std::make_pair(1, 3)};
  for (auto &p: nn_pairs) {
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
      10,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-8, 20)
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
  RunTestMPIVMPSCase(
      dmps, dmpo, sweep_params,
      -8.868563739680, 1.0E-10,
      world
  );


  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1U1QN>(zsite_vec_4, qn0);
  for (auto &p: nn_pairs) {
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
  RunTestMPIVMPSCase(
      zmps, zmpo, sweep_params,
      -8.868563739680, 1.0E-10,
      world
  );
}

struct TestTwoSiteAlgorithmTjSystem1U1Symm : public testing::Test {
  U1QN qn0 = U1QN({QNCard("N", U1QNVal(0))});
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("N", U1QNVal(1))}), 2),
                             QNSctT(U1QN({QNCard("N", U1QNVal(0))}), 1)},
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);
  ZQLTensor zf = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});
  ZQLTensor zcup = ZQLTensor({pb_in, pb_out});
  ZQLTensor zcdagup = ZQLTensor({pb_in, pb_out});
  ZQLTensor zcdn = ZQLTensor({pb_in, pb_out});
  ZQLTensor zcdagdn = ZQLTensor({pb_in, pb_out});
  ZQLTensor zntot = ZQLTensor({pb_in, pb_out});
  ZQLTensor zid = ZQLTensor({pb_in, pb_out});

  boost::mpi::communicator world;

  void SetUp(void) {
    zf({0, 0}) = -1;
    zf({1, 1}) = -1;
    zf({2, 2}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;
    zcup({2, 0}) = 1;
    zcdagup({0, 2}) = 1;
    zcdn({2, 1}) = 1;
    zcdagdn({1, 2}) = 1;
    zntot({0, 0}) = 1;
    zntot({1, 1}) = 1;
    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zid({2, 2}) = 1;

    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

TEST_F(TestTwoSiteAlgorithmTjSystem1U1Symm, RashbaTermCase) {
  double t = 3.0;
  double J = 1.0;
  double lamb = 0.03;
  auto ilamb = QLTEN_Complex(0, lamb);
  size_t Nx = 3;
  size_t Ny = 2;
  size_t Ntot = Nx * Ny;
  char BCx = 'p';
  char BCy = 'o';
  ZSiteVec site_vec(Ntot, pb_out);
  auto mpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(site_vec, qn0);
  for (int x = 0; x < Nx; ++x) {
    for (int y = 0; y < Ny; ++y) {

      if (!((BCx == 'o') && (x == Nx - 1))) {
        auto s0 = coors2idxSquare(x, y, Nx, Ny);
        auto s1 = coors2idxSquare((x + 1) % Nx, y, Nx, Ny);
        KeepOrder(s0, s1);
        std::cout << s0 << " " << s1 << std::endl;
        mpo_gen.AddTerm(-t, zcdagup, s0, zcup, s1, zf);
        mpo_gen.AddTerm(-t, zcdagdn, s0, zcdn, s1, zf);
        mpo_gen.AddTerm(-t, zcup, s0, zcdagup, s1, zf);
        mpo_gen.AddTerm(-t, zcdn, s0, zcdagdn, s1, zf);
        mpo_gen.AddTerm(J, zsz, s0, zsz, s1);
        mpo_gen.AddTerm(J / 2, zsp, s0, zsm, s1);
        mpo_gen.AddTerm(J / 2, zsm, s0, zsp, s1);
        mpo_gen.AddTerm(-J / 4, zntot, s0, zntot, s1);
        // SO term
        if (x != Nx - 1) {
          mpo_gen.AddTerm(lamb, zcdagup, s0, zcdn, s1, zf);
          mpo_gen.AddTerm(lamb, zcup, s0, zcdagdn, s1, zf);
          mpo_gen.AddTerm(-lamb, zcdagdn, s0, zcup, s1, zf);
          mpo_gen.AddTerm(-lamb, zcdn, s0, zcdagup, s1, zf);
        } else {    // At the boundary
          mpo_gen.AddTerm(lamb, zcdn, s0, zcdagup, s1, zf);
          mpo_gen.AddTerm(lamb, zcdagdn, s0, zcup, s1, zf);
          mpo_gen.AddTerm(-lamb, zcup, s0, zcdagdn, s1, zf);
          mpo_gen.AddTerm(-lamb, zcdagup, s0, zcdn, s1, zf);
        }
      }

      if (!((BCy == 'o') && (y == Ny - 1))) {
        auto s0 = coors2idxSquare(x, y, Nx, Ny);
        auto s2 = coors2idxSquare(x, (y + 1) % Ny, Nx, Ny);
        KeepOrder(s0, s2);
        std::cout << s0 << " " << s2 << std::endl;
        mpo_gen.AddTerm(-t, zcdagup, s0, zcup, s2, zf);
        mpo_gen.AddTerm(-t, zcdagdn, s0, zcdn, s2, zf);
        mpo_gen.AddTerm(-t, zcup, s0, zcdagup, s2, zf);
        mpo_gen.AddTerm(-t, zcdn, s0, zcdagdn, s2, zf);
        mpo_gen.AddTerm(J, zsz, s0, zsz, s2);
        mpo_gen.AddTerm(J / 2, zsp, s0, zsm, s2);
        mpo_gen.AddTerm(J / 2, zsm, s0, zsp, s2);
        mpo_gen.AddTerm(-J / 4, zntot, s0, zntot, s2);
        if (y != Ny - 1) {
          mpo_gen.AddTerm(ilamb, zcdagup, s0, zcdn, s2, zf);
          mpo_gen.AddTerm(ilamb, zcdagdn, s0, zcup, s2, zf);
          mpo_gen.AddTerm(-ilamb, zcup, s0, zcdagdn, s2, zf);
          mpo_gen.AddTerm(-ilamb, zcdn, s0, zcdagup, s2, zf);
        } else {    // At the boundary
          mpo_gen.AddTerm(ilamb, zcup, s0, zcdagdn, s2, zf);
          mpo_gen.AddTerm(ilamb, zcdn, s0, zcdagup, s2, zf);
          mpo_gen.AddTerm(-ilamb, zcdagup, s0, zcdn, s2, zf);
          mpo_gen.AddTerm(-ilamb, zcdagdn, s0, zcup, s2, zf);
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
  DirectStateInitMps(mps, {0, 1, 0, 2, 0, 1});
  if (world.rank() == 0) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    mps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      mps, mpo, sweep_params,
      -11.018692166942165, 1.0E-10,
      world
  );
}

struct TestTwoSiteAlgorithmHubbardSystem : public testing::Test {
  size_t Nx = 2;
  size_t Ny = 2;
  size_t N = Nx * Ny;
  double t0 = 1.0;
  double t1 = 0.5;
  double U = 2.0;

  U1U1QN qn0 = U1U1QN({QNCard("Nup", U1QNVal(0)), QNCard("Ndn", U1QNVal(0))});
  IndexT2 pb_out = IndexT2({
                               QNSctT2(U1U1QN({QNCard("Nup", U1QNVal(0)), QNCard("Ndn", U1QNVal(0))}), 1),
                               QNSctT2(U1U1QN({QNCard("Nup", U1QNVal(1)), QNCard("Ndn", U1QNVal(0))}), 1),
                               QNSctT2(U1U1QN({QNCard("Nup", U1QNVal(0)), QNCard("Ndn", U1QNVal(1))}), 1),
                               QNSctT2(U1U1QN({QNCard("Nup", U1QNVal(1)), QNCard("Ndn", U1QNVal(1))}), 1)},
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

  mpi::communicator world;

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
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

TEST_F(TestTwoSiteAlgorithmHubbardSystem, 2Dcase) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1U1QN>(dsite_vec, qn0);
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
  auto qn0 = U1U1QN({QNCard("Nup", U1QNVal(0)), QNCard("Ndn", U1QNVal(0))});
  std::vector<size_t> stat_labs(N);
  for (size_t i = 0; i < N; ++i) { stat_labs[i] = (i % 2 == 0 ? 1 : 2); }
  DirectStateInitMps(dmps, stat_labs);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    dmps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      dmps, dmpo, sweep_params,
      -2.828427124746, 1.0E-10,
      world
  );

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1U1QN>(zsite_vec, qn0);
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
  DirectStateInitMps(zmps, stat_labs);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    zmps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      zmps, zmpo, sweep_params,
      -2.828427124746, 1.0E-10,
      world
  );
}

// Test non-uniform local Hilbert spaces system.
// Kondo insulator, ref 10.1103/PhysRevB.97.245119,
struct TestKondoInsulatorSystem : public testing::Test {
  size_t Nx = 4;
  size_t N = 2 * Nx;
  double t = 0.25;
  double Jk = 1.0;
  double Jz = 0.5;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
  IndexT pb_outE = IndexT(
      {QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 4)},
      TenIndexDirType::OUT
  );    // extended electron
  IndexT pb_outL = IndexT(
      {QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
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

  mpi::communicator world;

  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }

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
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec, qn0);
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
  DirectStateInitMps(dmps, stat_labs);
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
    dmps.Dump(sweep_params.mps_path, true);
  }
  RunTestMPIVMPSCase(
      dmps, dmpo,
      sweep_params,
      -3.180025784229132, 1.0E-10,
      world
  );
  if (world.rank() == kMasterRank) {
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
  }
}
