// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-07-29 14:29
*
* Description: QuantumLiquids/UltraDMRG project. One-dimensional vector representation of the local Hilbert spaces of the system.
*/

/**
@file site_vec.h
@brief One-dimensional vector representation of the local Hilbert spaces of the
       system.
*/
#ifndef QLMPS_SITE_VEC_H
#define QLMPS_SITE_VEC_H


#include "qlten/qlten.h"  // Index, IndexVec, InverseIndex

#include <vector>   // vector

#include <assert.h>       // assert


#ifdef Release
  #define NDEBUG
#endif


namespace qlmps {
using namespace qlten;


// Helpers
template <typename QNT>
inline Index<QNT> SetIndexDirOut(const Index<QNT> &idx) {
  if (idx.GetDir() == TenIndexDirType::OUT) {
    return idx;
  } else {
    return InverseIndex(idx);
  }
}


template <typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> GenIdOp(const Index<QNT> &idx_out) {
  auto idx_in = InverseIndex(idx_out);
  QLTensor<TenElemT, QNT> id_op({idx_in, idx_out});
  for (size_t i = 0; i < idx_out.dim(); ++i) { id_op(i, i) = 1.0; }
  return id_op;
}


/**
Vector of the local Hilbert spaces of the system.

@tparam TenT The type of tensor used in the system.

@since version 0.2.0
*/
template <typename TenElemT, typename QNT>
class SiteVec {
public:
  using TenT = QLTensor<TenElemT, QNT>;
  /**
  Create a system with N identical sites.

  @param N Total number of sites (> 0) of the system.
  @param local_hilbert_space Local (on-site) Hilbert space represented by a
         Index.

  @since version 0.2.0
  */
  SiteVec(const size_t N, const Index<QNT> &local_hilbert_space) {
    assert(N > 0);
    size = N;
    sites = IndexVec<QNT>(N, SetIndexDirOut(local_hilbert_space));
    GenIdOpsVec_();
  }

  /**
  Create a sites vector using a vector of arbitrary Indexes.

  @param local_hilbert_spaces Local (on-site) Hilbert spaces on the each site.

  @since version 0.2.0
  */
  SiteVec(const IndexVec<QNT> &local_hilbert_spaces) {
    size = local_hilbert_spaces.size();
    assert(size > 0);
    sites.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      sites.emplace_back(SetIndexDirOut(local_hilbert_spaces[i]));
    }
    GenIdOpsVec_();
  }

  /**
  The copy constructor.

  @param site_vec A SiteVec instance.

  @since version 0.2.0
  */
  SiteVec(const SiteVec &site_vec) :
      size(site_vec.size), sites(site_vec.sites), id_ops(site_vec.id_ops) {}

  /**
  Copy a SiteVec.

  @param rhs A SiteVec instance.

  @since version 0.2.0
  */
  SiteVec<TenElemT, QNT> &operator=(const SiteVec &rhs) {
    size = rhs.size;
    sites = rhs.sites;
    id_ops = rhs.id_ops;
    return *this;
  }

  /**
  Equivalence check.

  @param rhs A SiteVec instance.
  */
  bool operator==(const SiteVec &rhs) const {
    return sites == rhs.sites;
  }

  bool operator!=(const SiteVec &rhs) const {
    return !(*this == rhs);
  }

  size_t size;                  ///< The size of the SiteVec, i.e. the size of the system.
  IndexVec<QNT> sites;          ///< Local Hilbert spaces represented by a vector of Index with OUT direction.
  std::vector<TenT> id_ops;     ///< Identity operators on each site.

private:
  /**
  Generate identity operators for each site.
  */
  void GenIdOpsVec_(void);
};


template <typename TenElemT, typename QNT>
void SiteVec<TenElemT, QNT>::GenIdOpsVec_(void) {
  id_ops.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    id_ops.emplace_back(GenIdOp<TenElemT, QNT>(sites[i]));
  }
}
} /* qlmps */
#endif /* ifndef QLMPS_SITE_VEC_H */
