# UltraDMRG: A Powerful 1D Tensor Network Library for Simulating Strongly Correlated Electron Systems

[![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**UltraDMRG** is a powerful and efficient library for performing large-scale,
high-performance calculations using 1D tensor network algorithms.
It is specifically designed to address the complexities of simulating
two-dimensional strongly correlated electron systems, making previously
untamable problems more accessible. Our goal is to lower the barriers to
simulating strongly correlated systems. The package is header-only.

___

## Features

UltraDMRG offers the following key features:

- [x] MPI/CUDA parallelization of Density Matrix Renormalization Group
- [x] MPI/CUDA parallelization of MPS-based time-dependent variational principle algorithms
- [x] Finite-temperature calculations

___

## To-Do List

- [ ] infinite DMRG
- [ ] DMRG low-energy excitation states
- [ ] documentation

___

## Performance Benchmark

As a demonstration of the performance of UltraDMRG,
we conducted a benchmark comparing the performance of UltraDMRG with ITensor(C++).
Specifically, we focused on comparing the DMRG sweep time using both packages.
The test model is the $4 \times 16$ Hubbard cylinder at a doping level of 1/8,
and we maintain a kept state dimension of $D=2000$ with incorporating $U(1)\times U(1)$ symmetry.

![benchmark](./benchmark/benchmark_hubbard4x16U1U1.png)

Note that ITensor utilizes the Davidson method for diagonalizing the Hamiltonian,
whereas UltraDMRG employs the Lanczos method.
This difference in methodology makes a performance comparison not straightforward.
To ensure a fair evaluation, we established consistent parameters for both packages
in the following way.
The truncation error cut-off was set at $10^{-8}$,
and the diagonalized Hamiltonian accuracy at $10^{-9}$.
We also set the iteration times for both the Davidson and Lanczos methods to 100,
effectively approaching infinity.
We specifically selected sweep times exclusively from near-converged sweeps.
We also close the MPI acceleration in UltraDMRG.
We hope above setting can ensure an unbiased comparison.

The codes used in the benchmark can be found in the directory `./benchmark`.
The results of the performance benchmark,
showcasing the time of single sweep, are presented in the accompanying figure.

___

## Dependence

UltraDMRG is header-only and requires no installation dependencies.
However, building test cases or practical DMRG/TDVP programs based on this project requires:

- **C++ Compiler:** C++17 or above
- **Build System:** CMake (version 3.12 or higher)
- **Math Libraries:** Intel MKL or OpenBLAS
- **Parallelization:** MPI
- **Tensor Operations:** [QuantumLiquids/TensorToolkit](https://github.com/QuantumLiquids/TensorToolkit)
- **GPU Acceleration (optional):** CUDA compiler, cuBLAS, cuSolver, cuTensor2
- **Testing (optional):** GoogleTest

___

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/QuantumLiquids/UltraDMRG.git
   cd UltraDMRG
   ```

2. Build using CMake:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install/location
   make install
   ```

This will install **UltraDMRG** into the specified directory.

### Building Tests

To build and run the unit tests:

1. From the `build` directory:
    ```bash
    cmake .. -DQLMPS_BUILD_UNITTEST=ON \
             -DCMAKE_CXX_COMPILER=icpc/g++/clang++ \
             -DGTest_DIR=/path/to/googletest \
             -DQLMPS_USE_GPU=ON/OFF \
             -DCUTENSOR_ROOT=/path/to/cutensor/if/using/cuda
    make -j16
    ```

2. Run the tests:
    ```bash
    ctest
    ```

---

## Author

Hao-Xin Wang

For inquiries, questions, or collaboration opportunities, please contact Hao-Xin via email:
[wanghaoxin1996@gmail.com](mailto:wanghaoxin1996@gmail.com).

## Acknowledgments

UltraDMRG is built upon the foundation laid by the [GraceQ/MPS2](https://mps2.gracequantum.org) project.
While initially inspired by GraceQ/mps2,
UltraDMRG expands upon its capabilities by adding additional 1D tensor-network algorithms, dramatically improving
performance, and most
importantly, introducing support for MPI/CUDA accelerations.
We would like to express our gratitude to the following individuals for their contributions and guidance:

- Rong-Yang Sun, the author of [GraceQ/mps2](https://mps2.gracequantum.org), for creating the initial framework that
  served as the basis for UltraDMRG.
- Yi-Fan Jiang, providing me with extensive help and guidance in writing parallel DMRG
- Hong Yao, my PhD advisor. His encouragement and continuous support
  of computational resources played crucial roles in the implementation of parallel DMRG.
- Zhen-Cheng Gu, my advisor, one of the pioneers in the field of tensor network.

Their expertise and support have been invaluable in the development of UltraDMRG.

## License

UltraDMRG is released under the LGPL3 License. Please see the LICENSE file for more details.
