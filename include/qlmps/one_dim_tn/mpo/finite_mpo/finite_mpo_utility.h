// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2022/5/10
*
* Description: original GraceQ/XTRG project.
*/

#ifndef QLMPS_ONE_DIM_TN_MPO_FINITE_MPO_FINITE_MPO_UTILITY_H
#define QLMPS_ONE_DIM_TN_MPO_FINITE_MPO_FINITE_MPO_UTILITY_H

#include "qlten/qlten.h"

namespace qlmps {
using namespace qlten;

/* FYI, lanczos figure
 * |----0                       0-----
 * |          2         2            |
 * |          |         |            |
 * |----1 0-------3 0--------3  1-----
 * |          |         |            |
 * |          1       1 2            |
 * |          |        |             |
 * |----2 0-------------------3 2----|
 */


//Forward declaration
template<typename DTenT>
inline double MeasureEE(const DTenT &s, const size_t sdim);

inline std::string GenEnvTenName(
    const std::string &dir, const long blk_len, const std::string temp_path
);

inline void RemoveFile(const std::string &file);

template<typename TenElemT, typename QNT>
void Generate3MPOEnvs(
    const FiniteMPO<TenElemT, QNT> &mpo0,
    const FiniteMPO<TenElemT, QNT> &mpo1,
    const FiniteMPO<TenElemT, QNT> &mpo2,
    const std::string temp_path
);

template<typename TenElemT, typename QNT, char dir>
std::pair<size_t, double> MPOProductVariationalSingleStep(
    const QLTensor<TenElemT, QNT> &lenv,
    const QLTensor<TenElemT, QNT> &renv,
    const QLTensor<TenElemT, QNT> &mpo1l,
    const QLTensor<TenElemT, QNT> &mpo1r,
    const QLTensor<TenElemT, QNT> &mpo2l,
    const QLTensor<TenElemT, QNT> &mpo2r,
    const QLTEN_Double trunc_err,
    const size_t Dmin,
    const size_t Dmax,
    QLTensor<TenElemT, QNT> &mpo_out_l,
    QLTensor<TenElemT, QNT> &mpo_out_r,
    bool output_info = false
);

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> UpdateLenv(
    const QLTensor<TenElemT, QNT> &lenv,
    const QLTensor<TenElemT, QNT> &mpo_ten_o,
    const QLTensor<TenElemT, QNT> &mpo_ten1,
    const QLTensor<TenElemT, QNT> &mpo_ten2
);

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> UpdateRenv(
    const QLTensor<TenElemT, QNT> &renv,
    const QLTensor<TenElemT, QNT> &mpo_ten_o,
    const QLTensor<TenElemT, QNT> &mpo_ten1,
    const QLTensor<TenElemT, QNT> &mpo_ten2
);

template<typename TenElemT, typename QNT>
void MpoAllScale(
    FiniteMPO<TenElemT, QNT> &mpo,
    const TenElemT factor
) {
  for (QLTensor<TenElemT, QNT> *ptensor : mpo) {
    ptensor->MultiplyByScalar(factor);
  }
  return;
}

template<typename TenElemT, typename QNT>
FiniteMPO<TenElemT, QNT> GenerateIdentityMPO(
    const FiniteMPO<TenElemT, QNT> &sample_mpo //e.g. Hamiltonian
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  const size_t N = sample_mpo.size();
  FiniteMPO<TenElemT, QNT> identity(N);
  const Index<QNT> trivial_index_in = sample_mpo[0].GetIndexes()[0];
  const Index<QNT> trivial_index_out = InverseIndex(trivial_index_in);
  for (size_t i = 0; i < N; i++) {
    Index<QNT> pb2 = sample_mpo[i].GetIndexes()[2];
    Index<QNT> pb1 = sample_mpo[i].GetIndexes()[1];
    identity(i) = new Tensor({trivial_index_in,
                              pb1, pb2,
                              trivial_index_out});
    const size_t physical_dim = pb1.dim();
    for (size_t j = 0; j < physical_dim; j++) {
      identity[i]({0, j, j, 0}) = TenElemT(1.0);
    }
    if (Tensor::IsFermionic() && pb1.GetDir() == OUT) {
      identity[i].ActFermionPOps();
    }
  }
  return identity;
}

/**
 * op1_mpo * mpo2, op1_mpo above mpo2 bottom
 * no compression
 * @tparam TenElemT
 * @tparam QNT
 * @param input_mpo1
 * @param input_mpo2
 * @param output_mpo
 */
template<typename TenElemT, typename QNT>
void MpoProduct(
    const FiniteMPO<TenElemT, QNT> &mpo1,
    const FiniteMPO<TenElemT, QNT> &mpo2,
    FiniteMPO<TenElemT, QNT> &output_mpo
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  const size_t N = mpo1.size();
  for (size_t i = 0; i < N; i++) {
    if (output_mpo(i) != nullptr) {
      delete output_mpo(i);
    }
    output_mpo(i) = new Tensor();
  }

  for (size_t i = 0; i < N; i++) {
    Tensor temp;
    Contract(mpo1(i), mpo2(i), {{1}, {2}}, &temp);
    temp.FuseIndex(0, 3);
    temp.FuseIndex(2, 4);
    temp.Transpose({1, 3, 2, 0});
    output_mpo[i] = std::move(temp);
#ifndef NDEBUG
    const QNT qn1 = mpo1[i].Div();
    const QNT qn2 = mpo2[i].Div();
    const QNT qn_sum = output_mpo[i].Div();
    assert(qn1 + qn2 == qn_sum);
#endif
  }
  return;
}




/*
 * |----0                       0-----
 * |          2          2           |
 * |          |          |           |
 * |----1 0-------3 0--------3  1-----
 * |          |         |            |
 * |          1         1            |
 * |          2         2            |
 * |          |         |            |
 * |----2 0-------3 0-------3   2----|
 *            |         |
 *            1         1
 */

/** op1_mpo * mpo2, op1_mpo above mpo2 bottom
 *  compression according the bond dimension, by variational method.
 *  two-site update
 *  The function assume the output_mpo offer a good initial MPO
 *  except its elements are pointer pointing to nullptr,
 *  in which case we will copy a op1_mpo to output_mpo
 *
 *  convergence are justified by the norm2
 *  Finally, the central of the output_mpo is put in site 1.
 *  The output MPO is NOT normalized.
 *
 * @param mpo1
 * @param mpo2
 * @param output_mpo
 * @param Dmax
 * @return the 2 norm of the output MPO
 */
template<typename TenElemT, typename QNT>
void MpoProduct(
    const FiniteMPO<TenElemT, QNT> &mpo1,
    const FiniteMPO<TenElemT, QNT> &mpo2,
    FiniteMPO<TenElemT, QNT> &output_mpo,
    const size_t Dmin,
    const size_t Dmax, // the bond dimension
    const double trunc_err, // truncation error when svd decomposition
    const size_t sweep_time_max,   // max sweep time when variational sweep
    const double sweep_converge_tolerance,
    const std::string temp_path,
    const bool output_info
) {
  using Tensor = QLTensor<TenElemT, QNT>;
  const size_t N = mpo1.size();
  double output_mpo_inited(true);
  for (size_t i = 0; i < N; i++) {
    if (output_mpo(i) == nullptr) {
      output_mpo_inited = false;
      break;
    }
  }
  if (!output_mpo_inited) {
    output_mpo = mpo1;
  }
  const size_t indent_level = 1;
  std::cout << "\n"
            << IndentPrinter(indent_level)
            << "left canonicalize and generate environment tensors. " << std::endl;
//  Timer initial_timer("product_initial");
  output_mpo.Centralize(0);

  if (!IsPathExist(temp_path)) {
    CreatPath(temp_path);
  }

  Generate3MPOEnvs(output_mpo, mpo1, mpo2, temp_path);

//  initial_timer.PrintElapsed();


  double norm2(0.0);
  double trunc_err_max(0.0);
  size_t trunc_D_max(0);
  for (size_t sweep = 0; sweep < sweep_time_max; sweep++) {
    std::cout << IndentPrinter(indent_level) << "sweep " << sweep << std::endl;
    Timer sweep_timer("sweep");

    Tensor lenv;
    //load lenv 0;
    ReadQLTensorFromFile(lenv, GenEnvTenName("l", 0, temp_path));
    for (size_t lsite = 0; lsite < N - 2; lsite++) {
      Tensor renv;
      size_t rsite = lsite + 1;
      //load renv, delete it;
      size_t renv_len = N - rsite - 1;
      std::string renv_file = GenEnvTenName("r", renv_len, temp_path);
      ReadQLTensorFromFile(renv, renv_file);
      RemoveFile(renv_file);
      //update lsite, lsite+1, direction 'r'
      if (output_info) {
        std::cout << IndentPrinter(indent_level)
                  << " site = ("
                  << std::setw(4) << lsite
                  << ", "
                  << std::setw(4) << rsite
                  << ")"
                  << std::scientific;
      }
      auto [actual_D, actual_trunc_err] =
          MPOProductVariationalSingleStep<TenElemT, QNT, 'r'>(lenv, renv, mpo1[lsite], mpo1[rsite],
                                                              mpo2[lsite], mpo2[rsite], trunc_err, Dmin, Dmax,
                                                              output_mpo[lsite], output_mpo[rsite], output_info);
      trunc_err_max = std::max(trunc_err_max, actual_trunc_err);
      trunc_D_max = std::max(trunc_D_max, actual_D);
      //update lenv_next, dump it;
      lenv = std::move(UpdateLenv(lenv, output_mpo[lsite], mpo1[lsite], mpo2[lsite]));
      std::string lenv_file = GenEnvTenName("l", rsite, temp_path);
      WriteQLTensorTOFile(lenv, lenv_file);
    }

    //load renv 0;
    Tensor renv;
    ReadQLTensorFromFile(renv, GenEnvTenName("r", 0, temp_path));
    for (size_t lsite = N - 2; lsite > 0; lsite--) {
      Tensor lenv;
      size_t rsite = lsite + 1;
      size_t lenv_len = lsite;
      //load lenv, delete it;
      std::string lenv_file = GenEnvTenName("l", lenv_len, temp_path);
      ReadQLTensorFromFile(lenv, lenv_file);
      RemoveFile(lenv_file);

      //update lsite, lsite+1, direction 'l'
      if (output_info) {
        std::cout << IndentPrinter(indent_level)
                  << " site = ("
                  << std::setw(4) << lsite
                  << ", "
                  << std::setw(4) << rsite
                  << ")"
                  << std::scientific;
      }
      auto [actual_D, actual_trunc_err] =
          MPOProductVariationalSingleStep<TenElemT, QNT, 'l'>(lenv, renv, mpo1[lsite], mpo1[rsite],
                                                              mpo2[lsite], mpo2[rsite], trunc_err, Dmin, Dmax,
                                                              output_mpo[lsite], output_mpo[rsite], output_info);
      trunc_err_max = std::max(trunc_err_max, actual_trunc_err);
      trunc_D_max = std::max(trunc_D_max, actual_D);
      //update renv_next, dump it;
      renv = std::move(UpdateRenv(renv, output_mpo[rsite], mpo1[rsite], mpo2[rsite]));
      std::string renv_file = GenEnvTenName("r", N - lsite - 1, temp_path);
      WriteQLTensorTOFile(renv, renv_file);
    }
    double new_norm2 = output_mpo[1].Get2Norm();
    std::cout << IndentPrinter(indent_level) << "norm2 = " << new_norm2
              << " tunc_err : " << std::scientific << trunc_err_max
              << " Dmax :" << trunc_D_max
              << std::endl;
    sweep_timer.PrintElapsed();
    std::cout << "\n";
    if (sweep >= 1 && std::abs(new_norm2 - norm2) / norm2 < sweep_converge_tolerance) {
      break;
    } else {
      norm2 = new_norm2;
    }
  }

  output_mpo.tens_cano_type_[0] = MPOTenCanoType::LEFT;
  output_mpo.tens_cano_type_[1] = MPOTenCanoType::NONE;
  for (size_t i = 2; i < N; i++) {
    output_mpo.tens_cano_type_[i] = MPOTenCanoType::RIGHT;
  }
  output_mpo.center_ = 1;
  return;
}

/*
 *   The contraction:
 *
 *   |----0                         0-----
 *   |            2          2           |
 *   |            |          |           |
 * lenv----1 0--mpo1l--3 0---1r--3  1---renv
 *   |            |         |            |
 *   |            1         1            |
 *   |            2         2            |
 *   |            |         |            |
 *   |----2 0--mpo2l--3 0---2r--3   2----|
 *                |         |
 *                1         1
 *
 *
 */
template<typename TenElemT, typename QNT, char dir>
std::pair<size_t, double> MPOProductVariationalSingleStep(
    const QLTensor<TenElemT, QNT> &lenv,
    const QLTensor<TenElemT, QNT> &renv,
    const QLTensor<TenElemT, QNT> &mpo1l,
    const QLTensor<TenElemT, QNT> &mpo1r,
    const QLTensor<TenElemT, QNT> &mpo2l,
    const QLTensor<TenElemT, QNT> &mpo2r,
    const QLTEN_Double trunc_err,
    const size_t Dmin,
    const size_t Dmax,
    QLTensor<TenElemT, QNT> &mpo_out_l,
    QLTensor<TenElemT, QNT> &mpo_out_r,
    bool output_info
) {
  Timer update_timer("two_site_fvmpo_update");
  using Tensor = QLTensor<TenElemT, QNT>;
  Tensor *ptemp0 = new Tensor();
  Tensor *ptemp1 = new Tensor();
  Tensor *ptemp2 = new Tensor();
  Contract(&lenv, &mpo2l, {{2}, {0}}, ptemp0);
  Contract(ptemp0, &mpo1l, {{1, 3}, {0, 1}}, ptemp1);

  (*ptemp0) = Tensor();
  Contract(&mpo2r, &renv, {{3}, {2}}, ptemp0);
  Contract(ptemp0, &mpo1r, {{2, 4}, {1, 3}}, ptemp2);

  (*ptemp0) = Tensor();
  Contract(ptemp1, ptemp2, {{2, 4}, {0, 3}}, ptemp0);
  delete ptemp1;
  delete ptemp2;
  ptemp0->Transpose({0, 1, 2, 3, 5, 4});
  QLTEN_Double actual_trunc_err;
  size_t D;

  using DTenT = QLTensor<QLTEN_Double, QNT>;
  DTenT s;

  if (dir == 'r') {
    Tensor vt;
    mpo_out_l = Tensor();
    mpo_out_r = Tensor();
    SVD(ptemp0, 3, Div(mpo1l), trunc_err, Dmin, Dmax, &mpo_out_l, &s, &vt, &actual_trunc_err, &D);
    Contract(&s, &vt, {{1}, {0}}, &mpo_out_r);
  } else if (dir == 'l') {
    Tensor u;
    mpo_out_l = Tensor();
    mpo_out_r = Tensor();
    SVD(ptemp0, 3, Div(mpo1l), trunc_err, Dmin, Dmax, &u, &s, &mpo_out_r, &actual_trunc_err, &D);
    Contract(&u, &s, {{3}, {0}}, &mpo_out_l);
  } else {
    assert(false);
  }
  s.Normalize();
  delete ptemp0;
  auto ee = MeasureEE(s, D);

  auto update_elapsed_time = update_timer.Elapsed();
  //cout site i at out_side of the function
  if (output_info) {
    std::cout << " TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
              << " D = " << std::setw(5) << D
              << " Time = " << std::setw(8) << update_elapsed_time
              << " S = " << std::setw(10) << std::setprecision(7) << ee;
    std::cout << std::scientific << std::endl;
  }
  return std::make_pair(D, actual_trunc_err);
}

/**
 * generate the left and right envs of 3 mpo tensor network. op0_mpo is above op1_mpo and op1_mpo is above mpo2
 * The data of environment tensors will dump to the disk.
 * For two site update so mpo[0] has no renv but mpo[1] has.
 * left_boundary = 0; right boundary = N-1.
 *
 * op0_mpo should be dagged
 * @tparam TenElemT
 * @tparam QNT
 * @param mpo0
 * @param mpo1
 * @param mpo2
 */
template<typename TenElemT, typename QNT>
void Generate3MPOEnvs(
    const FiniteMPO<TenElemT, QNT> &mpo0,
    const FiniteMPO<TenElemT, QNT> &mpo1,
    const FiniteMPO<TenElemT, QNT> &mpo2,
    const std::string temp_path
) {
  using TenT = QLTensor<TenElemT, QNT>;
  auto N = mpo0.size();
  assert(N == mpo1.size());
  assert(N == mpo2.size());

  auto trivial_index0 = mpo0.back().GetIndexes()[3];
  auto trivial_index1 = InverseIndex(mpo1.back().GetIndexes()[3]);
  auto trivial_index2 = InverseIndex(mpo2.back().GetIndexes()[3]);
  TenT renv = TenT({trivial_index0, trivial_index1, trivial_index2});
  renv({0, 0, 0}) = 1;

  std::string file = GenEnvTenName("r", 0, temp_path);
  WriteQLTensorTOFile(renv, file);

  for (size_t i = 1; i < N - 1; ++i) {
    TenT renv_next = UpdateRenv(renv, mpo0[N - i], mpo1[N - i], mpo2[N - i]); //question: if it's efficient?
    std::string file = GenEnvTenName("r", i, temp_path);
    WriteQLTensorTOFile(renv_next, file);
    renv = std::move(renv_next);
  }

  trivial_index0 = mpo0.front().GetIndexes()[0];
  trivial_index1 = InverseIndex(mpo1.front().GetIndexes()[0]);
  trivial_index2 = InverseIndex(mpo2.front().GetIndexes()[0]);
  TenT lenv = TenT({trivial_index0, trivial_index1, trivial_index2});
  lenv({0, 0, 0}) = 1;
  file = GenEnvTenName("l", 0, temp_path);
  WriteQLTensorTOFile(lenv, file);
  return;
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> UpdateRenv(
    const QLTensor<TenElemT, QNT> &renv,
    const QLTensor<TenElemT, QNT> &mpo_ten_o,
    const QLTensor<TenElemT, QNT> &mpo_ten1,
    const QLTensor<TenElemT, QNT> &mpo_ten2
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT renv_next, temp0, temp1;
  TenT mpo_ten_o_dag = Dag(mpo_ten_o);
  mpo_ten_o_dag.Transpose({0, 2, 1, 3});
//  mpo_ten_o.Show();
//  renv.Show();
  Contract(&mpo_ten_o_dag, &renv, {{3}, {0}}, &temp0);

  Contract(&temp0, &mpo_ten1, {{1, 3}, {2, 3}}, &temp1);
  Contract(&temp1, &mpo_ten2, {{1, 4, 2}, {1, 2, 3}}, &renv_next);
  return renv_next;
}

/*            2
 *            |
 * |----0 0---mpo_o---3
 * |          |
 * |          1
 * |          2
 * |          |
 * |----1 0---op1_mpo---3
 * |          |
 * |          1
 * |          2
 * |          |
 * |----2 0---mpo2---3
 *            |
 *            1
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> UpdateLenv(
    const QLTensor<TenElemT, QNT> &lenv,
    const QLTensor<TenElemT, QNT> &mpo_ten_o,
    const QLTensor<TenElemT, QNT> &mpo_ten1,
    const QLTensor<TenElemT, QNT> &mpo_ten2
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT lenv_next, temp0, temp1;
  TenT mpo_ten_o_dag = Dag(mpo_ten_o);
  mpo_ten_o_dag.Transpose({0, 2, 1, 3});
  Contract(&mpo_ten_o_dag, &lenv, {{0}, {0}}, &temp0);
  Contract(&temp0, &mpo_ten1, {{0, 3}, {2, 0}}, &temp1);
  Contract(&temp1, &mpo_ten2, {{2, 0, 3}, {0, 1, 2}}, &lenv_next);
  return lenv_next;
}

}//qlmps

#endif //QLMPS_ONE_DIM_TN_MPO_FINITE_MPO_FINITE_MPO_UTILITY_H
