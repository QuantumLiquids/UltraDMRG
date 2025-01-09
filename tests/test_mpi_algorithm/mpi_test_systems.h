/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-09
*
* Description: QuantumLiquids/UltraDMRG project.
*/


#ifndef TEST_MPI_ALGORITHM_MPI_TEST_SYSTEMS_H
#define TEST_MPI_ALGORITHM_MPI_TEST_SYSTEMS_H

#include "qlten/qlten.h"
#include "qlmps/qlmps.h"

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

// Test spin systems
struct Test1DSpinSystem : public testing::Test {
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

  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  void SetUp(void) {
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &rank);
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
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

  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  int mpi_size;

  void SetUp(void) {
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &rank);
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
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

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;

  void SetUp(void) {
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &rank);

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
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

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

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;

  void SetUp(void) {
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &rank);
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
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

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

  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  void SetUp(void) {
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &rank);
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
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

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

  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  void SetUp(void) {
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &rank);
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
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};

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

  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  void SetUp(void) {
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &rank);
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
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

#endif //TEST_MPI_ALGORITHM_MPI_TEST_SYSTEMS_H
