//
// Created by haoxinwang on 04/02/2024.
//

#include "itensor/all.h"

using namespace itensor;

int main() {
  auto Nx = 16, Ny = 4;
  auto N = Nx * Ny;
  auto sites = Electron(N, {"ConserveQNs=", true});

  auto t = 1.0;
  auto U = 8.0;

  auto ampo = AutoMPO(sites);
  auto lattice = squareLattice(Nx, Ny, {"YPeriodic=", true});
  for (auto j : lattice) {
    ampo += -t, "Cdagup", j.s1, "Cup", j.s2;
    ampo += -t, "Cdagup", j.s2, "Cup", j.s1;
    ampo += -t, "Cdagdn", j.s1, "Cdn", j.s2;
    ampo += -t, "Cdagdn", j.s2, "Cdn", j.s1;
  }
  for (auto j : range1(N)) {
    ampo += U, "Nupdn", j;
  }
  auto H = toMPO(ampo);

  auto state = InitState(sites);
  int num_hole = 0;
  for (auto j : range1(N)) {
    if (num_hole < 8) {
      state.set(j, "Emp");
      num_hole++;
    } else {
      state.set(j, (j % 2 == 1 ? "Up" : "Dn"));
    }
  }
  auto psi0 = MPS(state);

  auto sweeps = Sweeps(12);
  sweeps.maxdim() = 10, 20, 100, 200, 400, 1000, 1500, 2000;
  sweeps.noise() = 1E-7, 1E-8, 1E-10, 0;
  sweeps.cutoff() = 1E-8;
  auto [energy, psi] = dmrg(H, psi0, sweeps, {"Quiet=", true});
  sweeps.maxdim() = 2000;
  sweeps.noise() = 0;
  sweeps.niter() = 100;
  auto [energy2, psi2] = dmrg(H, psi, sweeps, {"Quiet=", false, "ErrGoal=", 1e-9});

  return 0;
}