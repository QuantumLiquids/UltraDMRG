# UltraDMRG: A Powerful 1D Tensor Network Library for Simulating Strongly Correlated Electron Systems

QuantumLiquids/UltraDMRG is a powerful and efficient library for performing large-scale,
high-performance calculations using one-dimensional tensor network algorithms.
It is specifically designed to tackle the complexities of simulating untamable
two-dimensional strongly correlated electron systems.
Our goal in creating this package is to lower the barriers
associated with simulating strongly correlated electron systems, 
offering a user-friendly and accessible solution for researchers in this field.

## Functionality

UltraDMRG offers the following key features:

- [x] MPI parallelization of Density Matrix Renormalization Group
- [x] MPI parallelization of MPS-based time-dependent variational principle algorithm
- [x] Finite-temperature calculation

## To-Do List

- [ ] infinite DMRG 
- [ ] DMRG low-energy excitation states

## Dependence

Please note that the project requires the following dependencies
to be installed in order to build and run successfully:

- C++17 Compiler
- CMake (version 3.12 or higher)
- Intel MKL or OpenBlas
- MPI
- Boost::serialization, Boost::mpi (version 1.74 or higher)
- [QuantumLiquids/TensorToolkit](https://github.com/QuantumLiquids/TensorToolkit)
- GoogleTest (if testing is required)

## Install

Clone the repository into a desired directory and change into that location:

```
git clone https://github.com/QuantumLiquids/UltraDMRG.git
cd UltraDMRG
```

Using CMake:

```
mkdir build && cd build
cmake .. 
make -j4 && make install

```

You may want to specify `CMAKE_CXX_COMPILER` as your favorite C++ compiler,
and `CMAKE_INSTALL_PREFIX` as your install directory when you're calling `cmake`

## Author

Hao-Xin Wang

For any inquiries or questions regarding the project,
you can reach out to Hao-Xin via email at wanghaoxin1996@gmail.com.

## Acknowledgments

UltraDMRG is built upon the foundation laid by the [GraceQ/MPS2](https://mps2.gracequantum.org) project.
While initially inspired by GraceQ/mps2,
UltraDMRG expands upon its capabilities by adding additional 1D tensor-network algorithms, dramatically improving
performance, and most
importantly, introducing support for MPI parallelization.
We would like to express our gratitude to the following individuals for their contributions and guidance:

- Rong-Yang Sun, the author of [GraceQ/mps2](https://mps2.gracequantum.org), for creating the initial framework that
  served as the basis for UltraDMRG.
- Yi-Fan Jiang, providing me with extensive help and guidance in writing parallel DMRG
- Hong Yao, my PhD advisor. His encouragement and continuous support
  of computational resources played crucial roles in the implementation of parallel DMRG.
- Zhen-Cheng Gu, my postdoc advisor, one of the pioneers in the field of tensor network.

Their expertise and support have been invaluable in the development of UltraDMRG.

## Benchmark

To be appeared

## License

UltraDMRG is released under the LGPL3 License. Please see the LICENSE file for more details.
