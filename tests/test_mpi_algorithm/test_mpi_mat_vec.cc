// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-12
*
* Description: GraceQ/mps2 project. MPI effective hamiltonian multiply vector unittest
*/

#include "qlmps/algorithm/lanczos_params.h"
#include "../testing_utils.h"
#include "qlten/qlten.h"
#include "qlten/utility/timer.h"
#include "qlmps/algo_mpi/vmps/lanczos_solver_mpi_master.h"

#include "gtest/gtest.h"

#include <vector>
#include <iostream>
#include <fstream>

#ifdef Release
#define NDEBUG
#endif

#include <assert.h>

#include "mkl.h"

using namespace qlmps;
using namespace qlten;

using U1U1QN = QN<U1QNVal, U1QNVal>;
using IndexT = Index<U1U1QN>;
using QNSctT = QNSector<U1U1QN>;
using QNSctVecT = QNSectorVec<U1U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1U1QN>;

// TEST(MPI_LANCZOS_TEST, MatrixMultiplyVector){
int main(int argc, char *argv[]) {

  using std::vector;
  mpi::environment env;
  const MPI_Comm comm = MPI_COMM_WORLD
  size_t thread_num;
  if (argc == 1) {// no input paramter
    thread_num = 12;
  } else {
    thread_num = atoi(argv[1]);
  }
  hp_numeric::SetTensorManipulationThreads(thread_num);
  hp_numeric::SetTensorTransposeNumThreads(thread_num);
  DQLTensor lenv, renv, mpo1, mpo2, mps1, mps2;
  std::vector<DQLTensor *> eff_ham = {&lenv, &mpo1, &mpo2, &renv};
  if (comm.rank() == 0) {
    vector<DQLTensor *> load_ten_list = {&lenv, &renv, &mpo1, &mpo2, &mps1, &mps2};
    vector<std::string> file_name_list = {"lenv.qlten", "renv.qlten", "mpo_ten_l.qlten",
                                          "mpo_ten_r.qlten", "mps_ten_l.qlten", "mps_ten_r.qlten"};
    assert(load_ten_list.size() == file_name_list.size());
    Timer load_tensor_timer("load tensors");
    for (size_t i = 0; i < load_ten_list.size(); i++) {
      std::string file = file_name_list[i];
      std::ifstream ifs(file, std::ios::binary);
      if (!ifs.good()) {
        std::cout << "can not open the file " << file << std::endl;
        exit(1);
      }
      ifs >> *load_ten_list[i];
    }
    load_tensor_timer.PrintElapsed();
    std::cout << "Master has loaded the tensors." << std::endl;
    SendBroadCastQLTensor(comm, lenv, kMPIMasterRank);
    SendBroadCastQLTensor(comm, renv, kMPIMasterRank);
    SendBroadCastQLTensor(comm, mpo1, kMPIMasterRank);
    SendBroadCastQLTensor(comm, mpo2, kMPIMasterRank);
    DQLTensor *state = new DQLTensor();
    Contract(&mps1, &mps2, {{2}, {0}}, state);
    state->ConciseShow();
    Timer mpi_mat_vec_timer("mpi matrix multiply vector");
    DQLTensor *mpi_res = master_two_site_eff_ham_mul_state(eff_ham, state, comm);
    mpi_mat_vec_timer.PrintElapsed();

    Timer single_process_mat_vec_timer("single process matrix multiply vector");
    DQLTensor *single_process_res = eff_ham_mul_two_site_state(eff_ham, state);
    single_process_mat_vec_timer.PrintElapsed();
    DQLTensor diff = (*mpi_res) + (-(*single_process_res));
    EXPECT_NEAR((diff.Normalize()) / (single_process_res->Normalize()), 0.0, 1e-13);
    delete mpi_res;
    delete single_process_res;
    delete state;
  } else {
    RecvBroadCastQLTensor(comm, lenv, kMPIMasterRank);
    RecvBroadCastQLTensor(comm, renv, kMPIMasterRank);
    RecvBroadCastQLTensor(comm, mpo1, kMPIMasterRank);
    RecvBroadCastQLTensor(comm, mpo2, kMPIMasterRank);
    std::cout << "Slave has received the eff_hams" << std::endl;
    slave_two_site_eff_ham_mul_state(eff_ham, comm);
  }
  return 0;
}