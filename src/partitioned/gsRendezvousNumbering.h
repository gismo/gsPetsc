/** @file gsRendezvousNumbering.h

    @brief Distributed dof ownership and global row numbering, without
    any global-size array on any rank.

    Every rank knows the global (mapper) indices of the dofs active on
    its elements. A dof is owned by the lowest rank that touches it, and
    the rows of the global system are grouped contiguously by owning
    rank (ordered by global index within a rank), i.e. the PETSc row
    layout follows the element partition.

    The ownership is resolved with a rendezvous: global indices are block
    distributed over the ranks; each rank sends its dofs to the rank
    responsible for them, which determines owner and row and sends the
    row back. Memory per rank is O(local dofs + N/P + P), communication
    is two all-to-all exchanges.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <mpi.h>
#include <vector>
#include <numeric>
#include <algorithm>

namespace gismo
{

/**
   @brief Computes the global row of every local dof.

   @param comm       communicator
   @param N          global number of dofs (rows); every index in [0,N)
                     must be local to at least one rank
   @param localDofs  sorted, unique global indices in [0,N) of the dofs
                     active on this rank (owned and ghost)
   @param[out] rows  rows[k] = global row of localDofs[k]
   @return           number of rows owned by this rank
*/
inline index_t rendezvousNumbering(MPI_Comm comm, index_t N,
                                   const std::vector<index_t> & localDofs,
                                   std::vector<index_t> & rows)
{
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    GISMO_ENSURE(sizeof(index_t) == sizeof(int), "rendezvousNumbering: 32-bit index_t expected");

    const index_t blk = (N + P - 1) / P;            // rendezvous block size
    const auto dest = [blk](index_t g) { return static_cast<int>(g / blk); };
    const index_t myFirst = std::min<index_t>(rank * blk, N);
    const index_t myLast  = std::min<index_t>(myFirst + blk, N);

    // 1. send the local dofs to their rendezvous rank (localDofs is sorted,
    //    so the send buffer is the list itself)
    std::vector<int> scount(P, 0), sdispl(P, 0), rcount(P), rdispl(P, 0);
    for (index_t g : localDofs) ++scount[dest(g)];
    for (int r = 1; r < P; ++r) sdispl[r] = sdispl[r-1] + scount[r-1];
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, comm);
    for (int r = 1; r < P; ++r) rdispl[r] = rdispl[r-1] + rcount[r-1];
    std::vector<index_t> recv(rdispl[P-1] + rcount[P-1]);
    MPI_Alltoallv(localDofs.data(), scount.data(), sdispl.data(), MPI_INT,
                  recv.data(), rcount.data(), rdispl.data(), MPI_INT, comm);

    // 2. owner of each dof of my block: lowest requesting rank
    std::vector<int> owner(myLast - myFirst, P);
    for (int r = 0; r < P; ++r)
        for (int k = rdispl[r]; k < rdispl[r] + rcount[r]; ++k)
            owner[recv[k] - myFirst] = std::min(owner[recv[k] - myFirst], r);

    // 3. rows: grouped by owner, ordered by global index within an owner.
    //    Row of g = (rows owned by lower ranks) + (rows of the same owner in
    //    lower rendezvous blocks) + (position within my block)
    std::vector<long long> cnt(P, 0), before(P, 0), total(P, 0);
    for (int o : owner)
    {
        GISMO_ENSURE(o < P, "rendezvousNumbering: a dof is not active on any rank");
        ++cnt[o];
    }
    MPI_Exscan(cnt.data(), before.data(), P, MPI_LONG_LONG, MPI_SUM, comm);
    if (0 == rank) std::fill(before.begin(), before.end(), 0);
    MPI_Allreduce(cnt.data(), total.data(), P, MPI_LONG_LONG, MPI_SUM, comm);
    std::vector<long long> offset(P, 0); // first row of each owner
    for (int o = 1; o < P; ++o) offset[o] = offset[o-1] + total[o-1];
    GISMO_ENSURE(offset[P-1] + total[P-1] == N, "rendezvousNumbering: rows do not add up to N");

    std::vector<index_t> blockRow(owner.size());
    {
        std::vector<long long> next(P);
        for (int o = 0; o < P; ++o) next[o] = offset[o] + before[o];
        for (size_t i = 0; i != owner.size(); ++i)
            blockRow[i] = static_cast<index_t>(next[owner[i]]++);
    }

    // 4. send the rows back, in the order they were requested
    for (index_t & g : recv) g = blockRow[g - myFirst];
    rows.resize(localDofs.size());
    MPI_Alltoallv(recv.data(), rcount.data(), rdispl.data(), MPI_INT,
                  rows.data(), scount.data(), sdispl.data(), MPI_INT, comm);

    return static_cast<index_t>(total[rank]);
}

} // namespace gismo
