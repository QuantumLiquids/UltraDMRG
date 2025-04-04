
// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-23
*
* Description: QuantumLiquids/MPS project. Unittest for DMRG.
*/

#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/algorithm/dmrg/dmrg.h"            //Test object
#include "qlmps/one_dim_tn/mpo/mpogen/mpogen.h"   //MPO Generator
#include "qlmps/model_relevant/sites/model_site_base.h"
#include "qlmps/model_relevant/operators/operators_all.h"

using namespace qlmps;
using namespace qlten;

using special_qn::U1QN;
using special_qn::U1U1QN;

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

// Helpers
inline void KeepOrder(size_t &x, size_t &y) {
  if (x > y) {
    auto temp = y;
    y = x;
    x = temp;
  }
}

inline size_t coors2idx(
    const size_t x, const size_t y, const size_t Nx, const size_t Ny) {
  return x * Ny + y;
}

inline size_t coors2idxSquare(
    const int x, const int y, const size_t Nx, const size_t Ny) {
  return x * Ny + y;
}

inline size_t coors2idxHoneycomb(
    const int x, const int y, const size_t Nx, const size_t Ny) {
  return Ny * (x % Nx) + y % Ny;
}

inline void RemoveFolder(const std::string &folder_path) {
  std::string command = "rm -rf " + folder_path;
  system(command.c_str());
}

template<typename TenElemT, typename QNT>
void RunTestDMRGCase(
    FiniteMPS<TenElemT, QNT> &mps,
    const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const double benmrk_e0, const double precision
) {
  size_t start_flops = flop;
  Timer contract_timer("dmrg");
  auto e0 = FiniteDMRG(mps, mat_repr_mpo, sweep_params);
  double elapsed_time = contract_timer.Elapsed();
  size_t end_flops = flop;
  double Gflops_s = (end_flops - start_flops) * 1.e-9 / elapsed_time;
  std::cout << "flops = " << end_flops - start_flops << std::endl;
  EXPECT_NEAR(e0, benmrk_e0, precision);
  EXPECT_TRUE(mps.empty());
}

// Test spin systems
struct TestDMRGSpinSystem : public testing::Test {
  size_t N = 6;

  sites::SpinOneHalfSite<U1QN> spin_one_half_sites;
  DSiteVec dsite_vec_6 = spin_one_half_sites.GenUniformSites<QLTEN_Double>(N);
  ZSiteVec zsite_vec_6 = spin_one_half_sites.GenUniformSites<QLTEN_Complex>(N);

  DMPS dmps = DMPS(dsite_vec_6);

  ZMPS zmps = ZMPS(zsite_vec_6);

  SpinOneHalfOperators<QLTEN_Double, U1QN> doperators;
  SpinOneHalfOperators<QLTEN_Complex, U1QN> zoperators;

  void SetUp(void) {

  }
};

TEST_F(TestDMRGSpinSystem, 1DIsing) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec_6);
  for (size_t i = 0; i < N - 1; ++i) {
    dmpo_gen.AddTerm(1, {doperators.sz, doperators.sz}, {i, i + 1});
  }
  auto dmpo = dmpo_gen.GenMatReprMPO();

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      1, 10, 1.0E-5,
      LanczosParams(1.0E-7)
  );
  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(dmps, stat_labs);
  dmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(dmps, dmpo, sweep_params, -0.25 * (N - 1), 1.0E-10);

  dmps.Load(sweep_params.mps_path);
  MeasureOneSiteOp(dmps, doperators.sz, "dsz");
  std::vector<std::vector<size_t>> sites_set;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (i < j) { sites_set.push_back({i, j}); }
    }
  }
  MeasureTwoSiteOp(dmps, {doperators.sz, doperators.sz}, doperators.id, sites_set, "dszdsz");
  dmps.clear();

  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec_6);
  for (size_t i = 0; i < N - 1; ++i) {
    zmpo_gen.AddTerm(1, {zoperators.sz, zoperators.sz}, {i, i + 1});
  }
  auto zmpo = zmpo_gen.GenMatReprMPO();
  sweep_params = FiniteVMPSSweepParams(
      4,
      1, 10, 1.0E-5,
      LanczosParams(1.0E-7)
  );
  DirectStateInitMps(zmps, stat_labs);
  zmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(zmps, zmpo, sweep_params, -0.25 * (N - 1), 1.0E-10);

  zmps.Load(sweep_params.mps_path);
  MeasureOneSiteOp(zmps, zoperators.sz, "zsz");
  MeasureTwoSiteOp(zmps, {zoperators.sz, zoperators.sz}, zoperators.id, sites_set, "zszzsz");
  zmps.clear();

  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

TEST_F(TestDMRGSpinSystem, 1DHeisenberg) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec_6);
  for (size_t i = 0; i < N - 1; ++i) {
    dmpo_gen.AddTerm(1, {doperators.sz, doperators.sz}, {i, i + 1});
    dmpo_gen.AddTerm(0.5, {doperators.sp, doperators.sm}, {i, i + 1});
    dmpo_gen.AddTerm(0.5, {doperators.sm, doperators.sp}, {i, i + 1});
  }
  auto dmpo = dmpo_gen.GenMatReprMPO();

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-7)
  );
  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(dmps, stat_labs);
  dmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      dmps, dmpo, sweep_params,
      -2.493577133888, 1.0E-12
  );

  // Continue simulation test
  dmps.clear();
  RunTestDMRGCase(
      dmps, dmpo, sweep_params,
      -2.493577133888, 1.0E-12
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec_6);
  for (size_t i = 0; i < N - 1; ++i) {
    zmpo_gen.AddTerm(1, {zoperators.sz, zoperators.sz}, {i, i + 1});
    zmpo_gen.AddTerm(0.5, {zoperators.sp, zoperators.sm}, {i, i + 1});
    zmpo_gen.AddTerm(0.5, {zoperators.sm, zoperators.sp}, {i, i + 1});
  }
  auto zmpo = zmpo_gen.GenMatReprMPO();

  sweep_params = FiniteVMPSSweepParams(
      4,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-7)
  );
  DirectStateInitMps(zmps, stat_labs);
  zmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      zmps, zmpo, sweep_params,
      -2.493577133888, 1.0E-12
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

TEST_F(TestDMRGSpinSystem, 2DHeisenberg) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec_6);
  std::vector<std::pair<size_t, size_t>> nn_pairs = {
      std::make_pair(0, 1),
      std::make_pair(0, 2),
      std::make_pair(1, 3),
      std::make_pair(2, 3),
      std::make_pair(2, 4),
      std::make_pair(3, 5),
      std::make_pair(4, 5)
  };
  for (auto &p : nn_pairs) {
    dmpo_gen.AddTerm(1, {doperators.sz, doperators.sz}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5, {doperators.sp, doperators.sm}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5, {doperators.sm, doperators.sp}, {p.first, p.second});
  }
  auto dmpo = dmpo_gen.GenMatReprMPO();

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-7)
  );

  // Test direct product state initialization.
  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(dmps, stat_labs);
  dmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      dmps, dmpo, sweep_params,
      -3.129385241572, 1.0E-12
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec_6);
  for (auto &p : nn_pairs) {
    zmpo_gen.AddTerm(1, {zoperators.sz, zoperators.sz}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5, {zoperators.sp, zoperators.sm}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5, {zoperators.sm, zoperators.sp}, {p.first, p.second});
  }
  auto zmpo = zmpo_gen.GenMatReprMPO();

  sweep_params = FiniteVMPSSweepParams(
      4,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-7)
  );
  DirectStateInitMps(zmps, stat_labs);
  zmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      zmps, zmpo, sweep_params,
      -3.129385241572, 1.0E-12
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

TEST_F(TestDMRGSpinSystem, 2DKitaevSimpleCase) {
  size_t Nx = 4;
  size_t Ny = 2;
  size_t N1 = Nx * Ny;
  DSiteVec dsite_vec = spin_one_half_sites.GenUniformSites<QLTEN_Double>(N1);
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec);
  for (int x = 0; x < Nx; ++x) {
    for (int y = 0; y < Ny; ++y) {
      if (x % 2 == 1) {
        auto s0 = coors2idxHoneycomb(x, y, Nx, Ny);
        auto s1 = coors2idxHoneycomb(x - 1, y + 1, Nx, Ny);
        KeepOrder(s0, s1);
        dmpo_gen.AddTerm(1, {doperators.sz, doperators.sz}, {s0, s1});
      }
    }
  }
  auto dmpo = dmpo_gen.GenMatReprMPO();

  auto sweep_params = FiniteVMPSSweepParams(
      4,
      8, 8, 1.0E-4,
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
  dmps_8sites.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      dmps_8sites, dmpo, sweep_params,
      -1.0, 1.0E-12
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  ZSiteVec zsite_vec = spin_one_half_sites.GenUniformSites<QLTEN_Complex>(N1);
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(zsite_vec);
  for (int x = 0; x < Nx; ++x) {
    for (int y = 0; y < Ny; ++y) {
      if (x % 2 == 1) {
        auto s0 = coors2idxHoneycomb(x, y, Nx, Ny);
        auto s1 = coors2idxHoneycomb(x - 1, y + 1, Nx, Ny);
        KeepOrder(s0, s1);
        zmpo_gen.AddTerm(1, {zoperators.sz, zoperators.sz}, {s0, s1});
      }
    }
  }
  auto zmpo = zmpo_gen.GenMatReprMPO();
  auto zmps_8sites = ZMPS(zsite_vec);
  ExtendDirectRandomInitMps(zmps_8sites, {stat_labs1, stat_labs2}, 2);
  zmps_8sites.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      zmps_8sites, zmpo, sweep_params,
      -1.0, 1.0E-12);
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

TEST(TestTwoSiteAlgorithmNoSymmetrySpinSystem, 2DKitaevComplexCase) {
  using TenElemType = QLTEN_Complex;
  using QNT = qlten::special_qn::TrivialRepQN;
  using Tensor = QLTensor<TenElemType, qlten::special_qn::TrivialRepQN>;
  //---------------Define the model parameters-----------------
  double J = -1.0;
  double K = 1.0;
  double Gm = 0.1;
  double h = 0.1;
  size_t Nx = 3, Ny = 4;
  size_t N = Nx * Ny;

  //---------------Generate the MPO-----------------
  sites::SpinOneHalfSite<QNT> spin_one_half_sites;
  auto site_vec = spin_one_half_sites.GenUniformSites<TenElemType>(N);
  SpinOneHalfOperators<TenElemType, QNT> spin_operators;
  auto sz = spin_operators.sz;
  auto sx = spin_operators.GetSx();
  auto sy = spin_operators.GetSy();
  auto mpo_gen = MPOGenerator<TenElemType, QNT>(site_vec);
  // H =   J * \Sigma_{<ij>} S_i*S_j
  //       K * \Sigma_{<ij>,c-link} Sc_i*Sc_j
  //		 Gm * \Sigma_{<ij>,c-link} Sa_i*Sb_j + Sb_i*Sa_j
  //     - h * \Sigma_i{S^z}
  for (int x = 0; x < Nx; ++x) {
    for (int y = 0; y < Ny; ++y) {
      // use the configuration '/|\' to traverse the square lattice
      // single site operator
      auto site0_num = coors2idx(x, y, Nx, Ny);
      mpo_gen.AddTerm(-h, spin_operators.sz, site0_num);
      mpo_gen.AddTerm(-h, spin_operators.GetSx(), site0_num);
      mpo_gen.AddTerm(-h, spin_operators.GetSy(), site0_num);
      // the '/' part: x-link
      // note that x and y start from 0
      if (y % 2 == 0) {
        auto site0_num = coors2idx(x, y, Nx, Ny);
        auto site1_num = coors2idx(x, y + 1, Nx, Ny);
        std::cout << site0_num << " " << site1_num << std::endl;
        mpo_gen.AddTerm(J, spin_operators.sz, site0_num, spin_operators.sz, site1_num);
        mpo_gen.AddTerm(J, spin_operators.GetSx(), site0_num, spin_operators.GetSx(), site1_num);
        mpo_gen.AddTerm(J, spin_operators.GetSy(), site0_num, spin_operators.GetSy(), site1_num);
        mpo_gen.AddTerm(K, spin_operators.GetSx(), site0_num, spin_operators.GetSx(), site1_num);
        mpo_gen.AddTerm(Gm, spin_operators.sz, site0_num, spin_operators.GetSy(), site1_num);
        mpo_gen.AddTerm(Gm, spin_operators.GetSy(), site0_num, spin_operators.sz, site1_num);
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
  auto mpo = mpo_gen.GenMatReprMPO();

  FiniteMPS<TenElemType, QNT> mps(site_vec);
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
  mps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(mps, mpo, sweep_params, -4.57509167674, 2.0E-10);
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

// Test fermion models.
struct TestTwoSiteAlgorithmTjSystem2U1Symm : public testing::Test {
  size_t N = 4;
  double t = 3.0;
  double J = 1.0;
  sites::tJSite<U1U1QN> tj_fermion_sites;

  DSiteVec2 dsite_vec_4 = tj_fermion_sites.GenUniformSites<QLTEN_Double>(N);
  ZSiteVec2 zsite_vec_4 = tj_fermion_sites.GenUniformSites<QLTEN_Complex>(N);

  tJOperators<QLTEN_Double, U1U1QN> doperators;
  tJOperators<QLTEN_Complex, U1U1QN> zoperators;

  DQLTensor2 df = doperators.f;
  DQLTensor2 dsz = doperators.sz;
  DQLTensor2 dsp = doperators.sp;
  DQLTensor2 dsm = doperators.sm;
  DQLTensor2 dcup = doperators.bupa;
  DQLTensor2 dcdagup = doperators.bupc;
  DQLTensor2 dcdn = doperators.bdna;
  DQLTensor2 dcdagdn = doperators.bdnc;
  DMPS2 dmps = DMPS2(dsite_vec_4);

  ZQLTensor2 zf = zoperators.f;
  ZQLTensor2 zsz = zoperators.sz;
  ZQLTensor2 zsp = zoperators.sp;
  ZQLTensor2 zsm = zoperators.sm;
  ZQLTensor2 zcup = zoperators.bupa;
  ZQLTensor2 zcdagup = zoperators.bupc;
  ZQLTensor2 zcdn = zoperators.bdna;
  ZQLTensor2 zcdagdn = zoperators.bdnc;
  ZMPS2 zmps = ZMPS2(zsite_vec_4);

  void SetUp(void) {

  }
};

TEST_F(TestTwoSiteAlgorithmTjSystem2U1Symm, 1DCase) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1U1QN>(dsite_vec_4);
  for (size_t i = 0; i < N - 1; ++i) {
    dmpo_gen.AddTerm(-t, dcdagup, i, dcup, i + 1, df);
    dmpo_gen.AddTerm(-t, dcdagdn, i, dcdn, i + 1, df);
    dmpo_gen.AddTerm(-t, dcup, i, dcdagup, i + 1, df);
    dmpo_gen.AddTerm(-t, dcdn, i, dcdagdn, i + 1, df);
    dmpo_gen.AddTerm(J, dsz, i, dsz, i + 1);
    dmpo_gen.AddTerm(J / 2, dsp, i, dsm, i + 1);
    dmpo_gen.AddTerm(J / 2, dsm, i, dsp, i + 1);
  }
  auto dmpo = dmpo_gen.GenMatReprMPO();

  auto sweep_params = FiniteVMPSSweepParams(
      11,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-8, 20)
  );
  DirectStateInitMps(dmps, {2, 1, 2, 0});
  dmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      dmps, dmpo, sweep_params,
      -6.947478526233, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1U1QN>(zsite_vec_4);
  for (size_t i = 0; i < N - 1; ++i) {
    zmpo_gen.AddTerm(-t, zcdagup, i, zcup, i + 1, zf);
    zmpo_gen.AddTerm(-t, zcdagdn, i, zcdn, i + 1, zf);
    zmpo_gen.AddTerm(-t, zcup, i, zcdagup, i + 1, zf);
    zmpo_gen.AddTerm(-t, zcdn, i, zcdagdn, i + 1, zf);
    zmpo_gen.AddTerm(J, zsz, i, zsz, i + 1);
    zmpo_gen.AddTerm(J / 2, zsp, i, zsm, i + 1);
    zmpo_gen.AddTerm(J / 2, zsm, i, zsp, i + 1);
  }
  auto zmpo = zmpo_gen.GenMatReprMPO();
  DirectStateInitMps(zmps, {2, 1, 2, 0});
  zmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      zmps, zmpo, sweep_params,
      -6.947478526233, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

TEST_F(TestTwoSiteAlgorithmTjSystem2U1Symm, 2DCase) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1U1QN>(dsite_vec_4);
  std::vector<std::pair<int, int>> nn_pairs = {
      std::make_pair(0, 1),
      std::make_pair(0, 2),
      std::make_pair(2, 3),
      std::make_pair(1, 3)};
  for (auto &p : nn_pairs) {
    AddTJHoppingTerms(dmpo_gen, t, p.first, p.second, doperators);
    AddHeisenbergCoupling(dmpo_gen, J, p.first, p.second, doperators);
  }
  auto dmpo = dmpo_gen.GenMatReprMPO();

  auto sweep_params = FiniteVMPSSweepParams(
      10,
      8, 8, 1.0E-9,
      LanczosParams(1.0E-8, 20)
  );

  auto total_div = U1U1QN({QNCard("N", U1QNVal(N - 2)), QNCard("Sz", U1QNVal(0))});
  auto zero_div = U1U1QN({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))});

  // Direct product state initialization.
  DirectStateInitMps(dmps, {2, 0, 1, 2});
  dmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      dmps, dmpo, sweep_params,
      -8.868563739680, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1U1QN>(zsite_vec_4);
  for (auto &p : nn_pairs) {
    AddTJHoppingTerms(zmpo_gen, QLTEN_Complex(t), p.first, p.second, zoperators);
    AddHeisenbergCoupling(zmpo_gen, J, p.first, p.second, zoperators);
  }
  auto zmpo = zmpo_gen.GenMatReprMPO();
  DirectStateInitMps(zmps, {2, 0, 1, 2});
  zmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      zmps, zmpo, sweep_params,
      -8.868563739680, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

struct TestTwoSiteAlgorithmTjSystem1U1Symm : public testing::Test {
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
  auto mpo_gen = MPOGenerator<QLTEN_Complex, U1QN>(site_vec);
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
  auto mpo = mpo_gen.GenMatReprMPO();

  auto sweep_params = FiniteVMPSSweepParams(
      8,
      30, 30, 1.0E-4,
      LanczosParams(1.0E-14, 100)
  );
  auto mps = ZMPS(site_vec);
  DirectStateInitMps(mps, {0, 1, 0, 2, 0, 1});
  mps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      mps, mpo, sweep_params,
      -11.018692166942165, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}

struct TestTwoSiteAlgorithmHubbardSystem : public testing::Test {
  using QNT = U1U1QN;
  size_t Nx = 2;
  size_t Ny = 2;
  size_t N = Nx * Ny;
  double t0 = 1.0;
  double t1 = 0.5;
  double U = 2.0;

  sites::HubbardSite<QNT> sites;

  DSiteVec2 dsite_vec = sites.template GenUniformSites<QLTEN_Double>(N);
  ZSiteVec2 zsite_vec = sites.template GenUniformSites<QLTEN_Complex>(N);

  DMPS2 dmps = DMPS2(dsite_vec);
  ZMPS2 zmps = ZMPS2(zsite_vec);

  HubbardOperators<QLTEN_Double, QNT> doperators;
  HubbardOperators<QLTEN_Complex, QNT> zoperators;

  DQLTensor2 df = doperators.f;
  DQLTensor2 dadagup = doperators.bupc;
  DQLTensor2 daup = doperators.bupa;
  DQLTensor2 dadagdn = doperators.bdnc;
  DQLTensor2 dadn = doperators.bdna;

  ZQLTensor2 zf = zoperators.f;
  ZQLTensor2 zdadagup = zoperators.bupc;
  ZQLTensor2 zdaup = zoperators.bupa;
  ZQLTensor2 zdadagdn = zoperators.bdnc;
  ZQLTensor2 zdadn = zoperators.bdna;

  void SetUp(void) {}
};

TEST_F(TestTwoSiteAlgorithmHubbardSystem, 2Dcase) {
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1U1QN>(dsite_vec);
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      auto s0 = coors2idxSquare(i, j, Nx, Ny);
      dmpo_gen.AddTerm(U, doperators.nupndn, s0);

      if (i != Nx - 1) {
        auto s1 = coors2idxSquare(i + 1, j, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        AddHubbardHoppingTerms(dmpo_gen, t0, s0, s1, doperators);
      }
      if (j != Ny - 1) {
        auto s1 = coors2idxSquare(i, j + 1, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        AddHubbardHoppingTerms(dmpo_gen, t0, s0, s1, doperators);
      }

      if (j != Ny - 1) {
        if (i != 0) {
          auto s2 = coors2idxSquare(i - 1, j + 1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          AddHubbardHoppingTerms(dmpo_gen, t1, temp_s0, s2, doperators);
        }
        if (i != Nx - 1) {
          auto s2 = coors2idxSquare(i + 1, j + 1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          AddHubbardHoppingTerms(dmpo_gen, t1, temp_s0, s2, doperators);
        }
      }
    }
  }
  auto dmpo = dmpo_gen.GenMatReprMPO();

  auto sweep_params = FiniteVMPSSweepParams(
      10,
      16, 16, 1.0E-9,
      LanczosParams(1.0E-8, 20)
  );
  std::vector<size_t> stat_labs(N);
  for (size_t i = 0; i < N; ++i) { stat_labs[i] = (i % 2 == 0 ? 1 : 2); }
  DirectStateInitMps(dmps, stat_labs);
  dmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
      dmps, dmpo, sweep_params,
      -2.828427124746, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<QLTEN_Complex, U1U1QN>(zsite_vec);
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      auto s0 = coors2idxSquare(i, j, Nx, Ny);
      zmpo_gen.AddTerm(U, zoperators.nupndn, s0);

      if (i != Nx - 1) {
        auto s1 = coors2idxSquare(i + 1, j, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        AddHubbardHoppingTerms(zmpo_gen, t0, s0, s1, zoperators);
      }
      if (j != Ny - 1) {
        auto s1 = coors2idxSquare(i, j + 1, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        AddHubbardHoppingTerms(zmpo_gen, t0, s0, s1, zoperators);
      }

      if (j != Ny - 1) {
        if (i != 0) {
          auto s2 = coors2idxSquare(i - 1, j + 1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          AddHubbardHoppingTerms(zmpo_gen, t1, temp_s0, s2, zoperators);
        }
        if (i != Nx - 1) {
          auto s2 = coors2idxSquare(i + 1, j + 1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          AddHubbardHoppingTerms(zmpo_gen, t1, temp_s0, s2, zoperators);
        }
      }
    }
  }
  auto zmpo = zmpo_gen.GenMatReprMPO();
  DirectStateInitMps(zmps, stat_labs);
  zmps.Dump(sweep_params.mps_path, true);
  RunTestDMRGCase(
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
  auto dmpo_gen = MPOGenerator<QLTEN_Double, U1QN>(dsite_vec);
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
  auto dmpo = dmpo_gen.GenMatReprMPO();

  auto sweep_params = FiniteVMPSSweepParams(
      5,
      64, 64, 1.0E-9,
      LanczosParams(1.0E-8, 20)
  );

  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(dmps, stat_labs);
  dmps.Dump(sweep_params.mps_path, true);

  RunTestDMRGCase(
      dmps, dmpo,
      sweep_params,
      -3.180025784229132, 1.0E-10
  );
  RemoveFolder(sweep_params.mps_path);
  RemoveFolder(sweep_params.temp_path);
}



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
//auto dmpo_gen = MPOGenerator<QLTEN_Double>(site_vec);
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

//auto dmpo = dmpo_gen.GenMatReprMPO();
//std::vector<long> stat_labs(N,0);
//int qn_label = 1;
//for (int i = 0; i < N; i=i+Np+1) {
//stat_labs[i] = qn_label;
//qn_label=3-qn_label;
//}
//DirectStateInitMps(dmps, stat_labs, pb_set);
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
