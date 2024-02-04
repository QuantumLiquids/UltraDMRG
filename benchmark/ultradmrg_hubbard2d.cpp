//
// Created by haoxinwang on 04/02/2024.
//

#include "qlmps/qlmps.h"
#include "qlten/qlten.h"

using TenElemT = qlten::QLTEN_Double;

using qlten::QLTensor;

namespace plain_hubbard {
using U1U1QN = qlten::special_qn::U1U1QN;
using QNSctT = qlten::QNSector<U1U1QN>;
using IndexT = qlten::Index<U1U1QN>;
using Tensor = QLTensor<TenElemT, U1U1QN>;
const U1U1QN qn0 = U1U1QN(0, 0); //N(particle number), Sz
const IndexT pb_outF = IndexT({   //QNSctT( U1U1QN(N, Sz), degeneracy )
                                  QNSctT(U1U1QN(2, 0), 1),
                                  QNSctT(U1U1QN(1, 1), 1),
                                  QNSctT(U1U1QN(1, -1), 1),
                                  QNSctT(U1U1QN(0, 0), 1)
                              },
                              qlten::QLTenIndexDirType::OUT
);
const IndexT pb_inF = qlten::InverseIndex(pb_outF);

}//plain_hubbard
namespace plain_hubbard {
//Fermionic operators
extern Tensor sz, sp, sm, id;
extern Tensor f, bupc, bupa, bdnc, bdna;
extern Tensor bupcF, bupaF, Fbdnc, Fbdna;
extern Tensor cupccdnc, cdnacupa, Uterm, nf, nfsquare, nup, ndn;
void OperatorInitial();
}

using namespace qlmps;
using namespace qlten;
using namespace std;

int main(int argc, char *argv[]) {
  namespace mpi = boost::mpi;
  mpi::environment env;
  mpi::communicator world;

  size_t Lx = 16, Ly = 4;
  size_t N = Lx * Ly;
  cout << "The total number of sites: " << N << endl;
  double t = 1, U = 8, t2 = 0;
  cout << "Model parameter: "
       << "t = " << t << ",\n"
       << "t2= " << t2 << ",\n"
       << "U = " << U << endl;
  clock_t startTime, endTime;
  startTime = clock();

  qlten::hp_numeric::SetTensorManipulationThreads(14);

  double e0(0.0); //energy
  using namespace plain_hubbard;
  OperatorInitial();

  const SiteVec<TenElemT, U1U1QN> sites = SiteVec<TenElemT, U1U1QN>(N, pb_outF);
  qlmps::MPOGenerator<TenElemT, U1U1QN> mpo_gen(sites, qn0);

  for (size_t i = 0; i < N; ++i) {
    mpo_gen.AddTerm(U, Uterm, i);
    cout << "add site" << i << "Hubbard U term" << endl;
  }

  //horizontal interaction
  for (size_t i = 0; i < N - Ly; ++i) {
    size_t site1 = i, site2 = i + Ly;
    mpo_gen.AddTerm(-t, bupcF, site1, bupa, site2, f);
    mpo_gen.AddTerm(-t, bdnc, site1, Fbdna, site2, f);
    mpo_gen.AddTerm(t, bupaF, site1, bupc, site2, f);
    mpo_gen.AddTerm(t, bdna, site1, Fbdnc, site2, f);
    cout << "add site (" << site1 << "," << site2 << ")  hopping term" << endl;
  }
  //vertical interaction
  for (size_t i = 0; i < N; ++i) {
    size_t y = i % Ly; //x=i/Ly
    if (y < Ly - 1) {
      size_t site1 = i, site2 = i + 1;
      mpo_gen.AddTerm(-t, bupcF, site1, bupa, site2);
      mpo_gen.AddTerm(-t, bdnc, site1, Fbdna, site2);
      mpo_gen.AddTerm(t, bupaF, site1, bupc, site2);
      mpo_gen.AddTerm(t, bdna, site1, Fbdnc, site2);
      cout << "add site (" << site1 << "," << site2 << ")  hopping term" << endl;
    } else if (Ly > 2) {
      size_t site1 = i - Ly + 1, site2 = i;
      mpo_gen.AddTerm(-t, bupcF, site1, bupa, site2, f);
      mpo_gen.AddTerm(-t, bdnc, site1, Fbdna, site2, f);
      mpo_gen.AddTerm(t, bupaF, site1, bupc, site2, f);
      mpo_gen.AddTerm(t, bdna, site1, Fbdnc, site2, f);
      cout << "add site (" << site1 << "," << site2 << ")  hopping term" << endl;
    }
  }
  //t2
  for (size_t i = 0; i < N - Ly; ++i) {
    size_t y = i % Ly, x = i / Ly;

    size_t Txy = (y + 1) % Ly + (x + 1) * Ly;
    size_t site1 = std::min(i, Txy), site2 = std::max(i, Txy);
    mpo_gen.AddTerm(-t2, bupcF, site1, bupa, site2, f);
    mpo_gen.AddTerm(-t2, bdnc, site1, Fbdna, site2, f);
    mpo_gen.AddTerm(t2, bupaF, site1, bupc, site2, f);
    mpo_gen.AddTerm(t2, bdna, site1, Fbdnc, site2, f);
    cout << "add site (" << site1 << "," << site2 << ")  hopping2 term" << endl;

    size_t Txmy = (y + Ly - 1) % Ly + (x + 1) * Ly;
    site1 = std::min(i, Txmy);
    site2 = std::max(i, Txmy);
    mpo_gen.AddTerm(-t2, bupcF, site1, bupa, site2, f);
    mpo_gen.AddTerm(-t2, bdnc, site1, Fbdna, site2, f);
    mpo_gen.AddTerm(t2, bupaF, site1, bupc, site2, f);
    mpo_gen.AddTerm(t2, bdna, site1, Fbdnc, site2, f);
    cout << "add site (" << site1 << "," << site2 << ")  hopping2 term" << endl;
  }

  qlmps::FiniteMPO<QLTEN_Double, U1U1QN> finite_mpo = mpo_gen.Gen(false);
  finite_mpo.Truncate(1e-16, 1, 1000);
  auto mpo = mpo_gen.Gen();
  auto mro = mpo_gen.GenMatReprMPO();

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, U1U1QN>;
  FiniteMPST mps(sites);

  std::vector<long unsigned int> stat_labs(N);
  size_t site_number_per_hole;

  site_number_per_hole = N / 8;

  int sz_label = 0;
  for (size_t i = 0; i < N; ++i) {
    if (i % site_number_per_hole == site_number_per_hole - 1) {
      stat_labs[i] = 3;
    } else {
      stat_labs[i] = sz_label % 2 + 1;
      sz_label++;
    }
  }

  qlmps::FiniteVMPSSweepParams sweep_params(
      12,
      10, 10, 1e-8,
      qlmps::LanczosParams(1e-9, 100)
  );

  qlmps::DirectStateInitMps(mps, stat_labs);
  cout << "Initial mps as direct product state." << endl;
  mps.Dump(sweep_params.mps_path, true);

  sweep_params.Dmin = 10;
  sweep_params.Dmax = 10;
  e0 = qlmps::FiniteDMRG(mps, mro, sweep_params, world);
  sweep_params.Dmin = 100;
  sweep_params.Dmax = 100;
  e0 = qlmps::FiniteDMRG(mps, mro, sweep_params, world);
  sweep_params.Dmin = 1000;
  sweep_params.Dmax = 1000;
  e0 = qlmps::FiniteDMRG(mps, mro, sweep_params, world);
  sweep_params.Dmin = 2000;
  sweep_params.Dmax = 2000;
  e0 = qlmps::FiniteDMRG(mps, mro, sweep_params, world);

  Timer symbolic_mpo_dmrg_timer("DMRG");
  e0 = qlmps::FiniteDMRG(mps, mro, sweep_params, world);
  symbolic_mpo_dmrg_timer.Suspend();

  endTime = clock();
  double cpu_time = (double) (endTime - startTime) / CLOCKS_PER_SEC;
  cout << "CPU Time : " << cpu_time << "s" << endl;
  return 0;
}