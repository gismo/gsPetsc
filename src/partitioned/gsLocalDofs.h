/** @file gsLocalDofs.h

    @brief Rank-local dof numbering for element-partitioned assembly.

    The free dofs that are active on the elements of a rank (owned and
    ghost dofs) are collected and the dof mapper is localized
    (gsDofMapper::localize), so that the assembler, gsFeSolution and the
    Dirichlet values work with rank-local sizes. The returned local to
    global map is the only link to the global numbering.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <gsCore/gsDofMapper.h>
#include <gsDomain/gsDomain.h>
#include <gsExpressions/gsFeSpace.h>

namespace gismo
{

/// @brief Free dofs (as returned by \a mapper.index(), all components)
/// active on the elements of \a domain; sorted and unique.
///
/// \note Interface terms that couple to the neighbouring patch
/// (assembleIfc) need the dofs of the element across the interface as
/// well; they are not included here.
template<class T>
std::vector<index_t> localFreeDofs(const gsDomain<T> & domain,
                                   const gsFunctionSet<T> & basis,
                                   const gsDofMapper & mapper)
{
    std::vector<index_t> result;
    gsMatrix<index_t> act;
    for (auto it = domain.beginAll(); it != domain.endAll(); ++it)
    {
        const index_t p = it.patchIndex();
        basis.piece(p).active_into(it.centerPoint(), act);
        for (index_t c = 0; c != mapper.numComponents(); ++c)
            for (index_t i = 0; i != act.rows(); ++i)
            {
                const index_t ii = mapper.index(act(i), p, c);
                if (mapper.is_free_index(ii)) result.push_back(ii);
            }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    result.shrink_to_fit();
    return result;
}

/// @brief Localizes \a mapper to the dofs active on \a domain and returns
/// the map from local dof to global row: \a rowOf(local) =
/// perm(global), or the global index if \a perm is null.
template<class T>
gsVector<index_t> localizeMapper(const gsDomain<T> & domain,
                                 const gsFunctionSet<T> & basis,
                                 gsDofMapper & mapper,
                                 const gsVector<index_t> * perm = nullptr)
{
    const std::vector<index_t> l2g = localFreeDofs(domain, basis, mapper);
    gsVector<index_t> rowOf(l2g.size());
    for (size_t k = 0; k != l2g.size(); ++k)
        rowOf[k] = perm ? (*perm)[l2g[k] - mapper.firstIndex()] : l2g[k] - mapper.firstIndex();
    mapper.localize(l2g);
    return rowOf;
}

/// @brief Localizes the dof mapper of the space \a u of a gsExprAssembler
/// to the dofs \a localDofs (see gsDofMapper::localize). The mapper is
/// shared by all copies of the space (like gsFeSpace::setupMapper), the
/// Dirichlet values remain valid.
template<class T>
void localizeSpace(const expr::gsFeSpace<T> & u, const std::vector<index_t> & localDofs)
{
    const_cast<expr::gsFeSpace<T>&>(u).mapper().localize(localDofs);
}

/// @brief As localizeMapper(), for the dof mapper of the space \a u of a
/// gsExprAssembler. The mapper is shared by all copies of the space
/// (like gsFeSpace::setupMapper), the Dirichlet values remain valid.
template<class T>
gsVector<index_t> localizeSpace(const expr::gsFeSpace<T> & u,
                                const gsDomain<T> & domain,
                                const gsVector<index_t> * perm = nullptr)
{
    gsDofMapper & mapper = const_cast<expr::gsFeSpace<T>&>(u).mapper();
    return localizeMapper(domain, u.source(), mapper, perm);
}

} // namespace gismo
