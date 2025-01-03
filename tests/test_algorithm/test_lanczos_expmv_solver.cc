// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>>
* Creation Date: 2021-10-13
*
* Description: GraceQ/mps2 project. Lanczos exponential of matrix times vector algorithm unittests.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlten/utility/timer.h"

#include "qlmps/consts.h"           // exp
#include "qlmps/algorithm/tdvp/lanczos_expmv_solver_impl.h"
#include "../testing_utils.h"

#ifdef Release
#define NDEBUG
#endif

using namespace qlmps;
using namespace qlten;

using U1QN = special_qn::U1QN;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

size_t d = 2;
size_t D = 10;
size_t dh = 2;

/* First Part: test function TridiagExpme1Solver */
class SymmetricTriDiagonalMatrix {
 public:
  //Constructor
  SymmetricTriDiagonalMatrix(void) = default;
  SymmetricTriDiagonalMatrix(const size_t n) : n(n), a(n, 0.0), b(n - 1, 0.0) {}
  SymmetricTriDiagonalMatrix(const size_t n,
                             const std::vector<double> &a,
                             const std::vector<double> &b) :
      n(n), a(a), b(b) {
    assert(a.size() >= n);
    assert(b.size() >= (n - 1));
  }

  size_t n;              //linear size;
  std::vector<double> a; //diagonal elements, a.size() >=n;
  std::vector<double> b; //second-diagonal elements, b.size() >= n-1;

  void ToColumnMajorFullMatrix(double *M);//M is n^2 length array
  void ToRawMajorFullMatrix(double *M) { ToColumnMajorFullMatrix(M); }
  void ToFullMatrix(double *M) { ToColumnMajorFullMatrix(M); }
};

void SymmetricTriDiagonalMatrix::ToColumnMajorFullMatrix(double *M) {
  for (size_t i = 0; i < n; i++) {
    M[i * n + i] = a[i];
  }
  for (size_t i = 0; i < n - 1; i++) {
    M[i * n + (i + 1)] = b[i];
    M[(i + 1) * n + i] = b[i];
  }
}

struct TestTridiagExpme1SolverExample : public testing::Test {
  SymmetricTriDiagonalMatrix matrix2x2_example1 = SymmetricTriDiagonalMatrix(2, {0.5, 0.3}, {0.2});
  double delta1 = -1.3;

  //matlab result
  QLTEN_Complex
      res_example1[2] = {QLTEN_Complex(0.76772272947713149360282613997697, 0.58726872368826332770908038583002),
                         QLTEN_Complex(-0.12737709795879115226568956131814, 0.22246872080662932757988414778083)};

  SymmetricTriDiagonalMatrix matrix5x5_example2 =
      SymmetricTriDiagonalMatrix(
          5,
          {1.8, 2.4, 0.5, 6.3, 0.3},
          {1.1, 0.2, 8.5, 0.9});

  double delta2 = 1.5;
  QLTEN_Complex res_example2[5]
      {QLTEN_Complex(0.13627869919086488259551970259054, -0.2627770725485846226021635629877),
       QLTEN_Complex(0.0095890697466718029240428933235307, 0.9546547280068733432045746667427),
       QLTEN_Complex(-0.000013884256143736063197576847960679, 0.013856219332059563353887199355086),
       QLTEN_Complex(0.0018295388155743035323713696627124, -0.022002544240195875957510907028336),
       QLTEN_Complex(-0.0094873011859597823625112056333819, -0.012217334778703219794193657321557)};
};

void RunTestTridiagExpme1SolverCase(
    const SymmetricTriDiagonalMatrix &matrix,
    const double delta,
    QLTEN_Complex *benchmark_res
) {
  const size_t n = matrix.n;
  QLTEN_Complex *res = new QLTEN_Complex[n];
  TridiagExpme1Solver(matrix.a, matrix.b, n, delta, res);
  EXPECT_NEAR(Distance(res, benchmark_res, n), 0.0, 1e-13);
  delete[] res;
}

TEST_F(TestTridiagExpme1SolverExample, TestTridiagExpme1Solver) {
  RunTestTridiagExpme1SolverCase(matrix2x2_example1, delta1, res_example1);
  RunTestTridiagExpme1SolverCase(matrix5x5_example2, delta2, res_example2);
}

/* Second Part: test function LanczosExpmvSolver */

//Only has no quantum number case now
struct TestLanczos : public testing::Test {
  QNT qn0 = QNT({QNCard("Sz", U1QNVal(0))});
  IndexT idx_din = IndexT({QNSctT(qn0, d)}, TenIndexDirType::IN);
  IndexT idx_dout = InverseIndex(idx_din);
  IndexT idx_Din = IndexT({QNSctT(qn0, D)}, TenIndexDirType::IN);
  IndexT idx_Dout = InverseIndex(idx_Din);
  IndexT idx_vin = IndexT({QNSctT(qn0, dh)}, TenIndexDirType::IN);
  IndexT idx_vout = InverseIndex(idx_vin);
};

/**
 *
 *
 * @tparam TenElemT
 * @param eff_ham
 * @output res_matrix,  raw major,
 *                      only upper triangular (include diagonal) elements is saved,
 *                      lower triangular(without diagonal) elements is set as zers.
 *                      note allocate memory outside the function:
 *                      res_matrix = (TenElemT *) malloc(dense_mat_size * sizeof(TenElemT)
  );
 */
template<typename TenElemT, typename QNT>
void EffectiveHamiltonianToDenseMatrixRepr(
    const std::vector<QLTensor<TenElemT, QNT> *> &eff_ham,
    TenElemT *res_matrix
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT *eff_ham_ten = new TenT;
  Contract(eff_ham[0], eff_ham[1], {{1}, {0}}, eff_ham_ten);
  InplaceContract(eff_ham_ten, eff_ham[2], {{4}, {0}});
  InplaceContract(eff_ham_ten, eff_ham[3], {{6}, {1}});
  eff_ham_ten->Transpose({0, 3, 5, 6, 1, 2, 4, 7});

  const size_t dense_mat_dim = eff_ham[0]->GetShape()[0]
      * eff_ham[1]->GetShape()[1]
      * eff_ham[2]->GetShape()[1]
      * eff_ham[3]->GetShape()[0];

  size_t idx = 0;
  for (auto &coors : GenAllCoors(eff_ham_ten->GetShape())) {
    const size_t eff_mat_raw = idx / dense_mat_dim;
    const size_t eff_mat_col = idx % dense_mat_dim;
    if (eff_mat_raw > eff_mat_col) {
      res_matrix[idx] = 0.0; //lower triangular
    } else {
      res_matrix[idx] = (*eff_ham_ten)(coors);
    }
    idx++;
  }
  delete eff_ham_ten;
}

template<typename TenElemT, typename QNT>
void StateToDenseVectorRepr(
    const QLTensor<TenElemT, QNT> &state,
    TenElemT *res_vector
) {
  const size_t vector_dim = state.size();
  size_t idx = 0;
  for (CoorsT &coors : GenAllCoors(state.GetShape())) {
    res_vector[idx] = state(coors);
    idx++;
  }
}

/**
 *  $$\exp( - 1i  \delta  A)  v$$
 *  for full Hermitian matrix, only upper elements in input matrix is valid.
 *
 *  method: $$A = V D V^\dag$, D is diagonal, and every column of V is A's eigenvector.
 *
 * @param matrix $A$
 * @param vector $v$
 * @param n
 * @param step_length
 * @param res
 */
inline void TridiagExpmvSolver(
    const QLTEN_Complex *matrix,
    const QLTEN_Complex *vector,
    const size_t n,
    const double step_length,
    QLTEN_Complex *res
) {
  QLTEN_Complex *eigenvectors = new QLTEN_Complex[n * n];
  std::memcpy(eigenvectors, matrix, n * n * sizeof(QLTEN_Complex));//unified code for CPU or CUDA code.
  double *w = new double[n];
  LapackeSyev(
      LAPACK_ROW_MAJOR, 'V', 'U',
      n, eigenvectors, n, w);
  QLTEN_Complex *eigenvectors_mul_v = new QLTEN_Complex[n];
  QLTEN_Complex alpha(1.0), beta(0.0);
  cblas_zgemv(CblasRowMajor, CblasConjTrans, n, n, &alpha, eigenvectors, n,
              vector, 1, &beta, eigenvectors_mul_v, 1);
  QLTEN_Complex *exp_of_eigenvals_mulVv = new QLTEN_Complex[n];
  for (size_t i = 0; i < n; i++) {
    exp_of_eigenvals_mulVv[i] = qlmps::complex_exp(QLTEN_Complex(0.0, -step_length * w[i])) * eigenvectors_mul_v[i];
  }
  cblas_zgemv(CblasRowMajor, CblasNoTrans, n, n, &alpha, eigenvectors, n,
              exp_of_eigenvals_mulVv, 1, &beta, res, 1);
  delete[] eigenvectors;
  delete[] w;
  delete[] eigenvectors_mul_v;
  delete[] exp_of_eigenvals_mulVv;
}

template<typename QNT>
void RunTestTwoSiteLanczosExpmvSolverCase(
    const std::vector<QLTensor<QLTEN_Complex, QNT> *> &eff_ham,
    QLTensor<QLTEN_Complex, QNT> *pinit_state,
    const double step_length,
    const LanczosParams &lanczos_params
) {
  using TenT = QLTensor<QLTEN_Complex, QNT>;
  size_t n = pinit_state->size();
  QLTEN_Complex *hamiltonian_dense_matrix = new QLTEN_Complex[n * n];
  QLTEN_Complex *initial_state_dense_matrix = new QLTEN_Complex[n];
  QLTEN_Complex *res_state_dense_matrix = new QLTEN_Complex[n];
  QLTEN_Complex *benchmark_res_state_dense_matrix = new QLTEN_Complex[n];
  EffectiveHamiltonianToDenseMatrixRepr(eff_ham, hamiltonian_dense_matrix);
  StateToDenseVectorRepr(*pinit_state, initial_state_dense_matrix);

  TridiagExpmvSolver(hamiltonian_dense_matrix,
                     initial_state_dense_matrix,
                     n,
                     step_length,
                     benchmark_res_state_dense_matrix);

  Timer timer("two_site_lancz");
  ExpmvRes<TenT> lancz_res = LanczosExpmvSolver(
      eff_ham,
      pinit_state,
      &eff_ham_mul_two_site_state,
      step_length,
      lanczos_params
  );
  timer.PrintElapsed();
  printf("lanczos iter = %zu", lancz_res.iters);
  StateToDenseVectorRepr(*lancz_res.expmv,
                         res_state_dense_matrix);

  double sum = 0;
  for (size_t i = 0; i < n; i++) {
    sum += qlten::norm(benchmark_res_state_dense_matrix[i]);
  }
  double norm = std::sqrt(sum);
  EXPECT_NEAR(Distance(res_state_dense_matrix,
                       benchmark_res_state_dense_matrix,
                       n) / norm,
              0.0,
              1.0e-13);

  delete[] hamiltonian_dense_matrix;
  delete[] initial_state_dense_matrix;
  delete[] benchmark_res_state_dense_matrix;
  delete[] res_state_dense_matrix;
  delete lancz_res.expmv;
}

TEST_F(TestLanczos, RunTestTwoSiteLanczosExpmvSolverCase) {
  // Tensor with double elements.
  LanczosParams lanczos_params(1.0E-14, 100);

  // Tensor with complex elements.
  auto zlblock = ZQLTensor({idx_Din, idx_vout, idx_Dout});
  auto zlsite = ZQLTensor({idx_vin, idx_din, idx_dout, idx_vout});
  auto zrblock = ZQLTensor({idx_Dout, idx_vin, idx_Din});
  auto zblock_random_mat = new QLTEN_Complex[D * D];
  RandCplxHerMat(zblock_random_mat, D);
  for (size_t i = 0; i < D; ++i) {
    for (size_t j = 0; j < D; ++j) {
      for (size_t k = 0; k < dh; ++k) {
        zlblock({i, k, j}) = zblock_random_mat[(i * D + j)];
        zrblock({j, k, i}) = zblock_random_mat[(i * D + j)];
      }
    }
  }
  delete[] zblock_random_mat;
  auto zsite_random_mat = new QLTEN_Complex[d * d];
  RandCplxHerMat(zsite_random_mat, d);
  for (size_t i = 0; i < d; ++i) {
    for (size_t j = 0; j < d; ++j) {
      for (size_t k = 0; k < dh; ++k) {
        zlsite({k, i, j, k}) = zsite_random_mat[(i * d + j)];
      }
    }
  }
  delete[] zsite_random_mat;
  auto zrsite = ZQLTensor(zlsite);
  auto pzinit_state = new ZQLTensor({idx_Din, idx_dout, idx_dout, idx_Dout});

  // Finish iteration when Lanczos error targeted.
  srand(0);
  pzinit_state->Random(qn0);
  RunTestTwoSiteLanczosExpmvSolverCase(
      {&zlblock, &zlsite, &zrsite, &zrblock},
      pzinit_state,
      0.1,
      lanczos_params
  );
}
