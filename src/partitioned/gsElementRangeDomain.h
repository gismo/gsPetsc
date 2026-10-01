/** @file gsElementRangeDomain.h

    @brief A domain consisting of a contiguous range of elements of another
    domain. Used to hand a rank-local element subset to gsExprAssembler /
    gsExprEvaluator via setIntegrationDomain().

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <gsDomain/gsDomain.h>

namespace gismo
{

/**
   @brief Iterator over a range of a base domain iterator.

   The element ids are local, i.e. they run from 0 to the size of the
   range. This keeps the id-based end comparison and the OpenMP
   chunking of gsDomain::allElements() valid.
 */
template <class T>
class gsElementRangeIterator : public gsDomainIterator<T>
{
    typedef gsDomainIterator<T> Base;
    gsDomainIteratorWrapper<T> m_it;
    index_t m_first; // global id of the first element of the range

public:
    typedef typename Base::uPtr uPtr;

    gsElementRangeIterator(gsDomainIteratorWrapper<T> it, index_t first)
    : Base(0), m_it(give(it)), m_first(first)
    { }

    uPtr clone() const override
    {
        gsElementRangeIterator * res = new gsElementRangeIterator(m_it, m_first);
        res->m_id = this->m_id;
        return uPtr(res);
    }

    void next() override { ++m_it; }
    void next(index_t k) override { m_it += k; }
    void prev() override { --m_it; }
    void prev(index_t k) override { m_it -= k; }

    gsVector<T> lowerCorner() const override { return m_it.lowerCorner(); }
    gsVector<T> upperCorner() const override { return m_it.upperCorner(); }
    bool isBoundaryElement() const override { return m_it.isBoundaryElement(); }

    index_t patchIndex()     const override { return m_it.patchIndex(); }
    index_t subdomainIndex() const override { return m_it.subdomainIndex(); }
    size_t  localId()        const override { return m_it.localId(); }

    /// Id of the current element in the numbering of the base domain
    index_t globalId() const { return m_first + this->m_id; }
};

/**
   @brief The elements [first, last) of a base domain (e.g. the
   composite domain of a gsMultiBasis).

   Only element iteration is restricted; everything else (dimension,
   degree, subdomains) is forwarded to the base domain. Boundary
   iteration is not restricted, i.e. boundary terms are still
   assembled by every rank.
 */
template <class T>
class gsElementRangeDomain : public gsDomain<T>
{
    typedef gsDomain<T> Base;
    typename Base::Ptr m_base;
    index_t m_first, m_last;

public:
    typedef typename Base::Ptr  Ptr;
    typedef typename Base::iterator iterator;

    gsElementRangeDomain(typename Base::Ptr base, index_t first, index_t last)
    : m_base(give(base)), m_first(first), m_last(last)
    {
        GISMO_ENSURE(0 <= m_first && m_first <= m_last &&
                     m_last <= static_cast<index_t>(m_base->numElements()),
                     "Invalid element range ["<<m_first<<","<<m_last<<")");
    }

    /// Block partition of the elements of \a base: part \a k out of \a n
    static Ptr blockPartition(typename Base::Ptr base, index_t k, index_t n)
    {
        const index_t ne = base->numElements();
        const index_t q = ne / n, r = ne % n;
        const index_t first = k*q + std::min(k,r);
        const index_t last  = first + q + (k<r ? 1 : 0);
        return Ptr(new gsElementRangeDomain(give(base), first, last));
    }

    index_t first() const { return m_first; }
    index_t last()  const { return m_last;  }
    const gsDomain<T> & base() const { return *m_base; }

    iterator beginAll() const override
    {
        iterator it = m_base->beginAll();
        it += m_first;
        return iterator(new gsElementRangeIterator<T>(give(it), m_first));
    }

    size_t numElements() const override { return m_last - m_first; }

    Ptr subdomain(index_t k) const override { return m_base->subdomain(k); }
    size_t nPieces() const override { return m_base->nPieces(); }

    iterator beginBdr(const boxSide bs = boundary::all) const override
    { return m_base->beginBdr(bs); }
    iterator endBdr(const boxSide bs = boundary::all) const override
    { return m_base->endBdr(bs); }
    size_t numElementsBdr(boxSide const & s = boundary::all) const override
    { return m_base->numElementsBdr(s); }

    short_t degree(short_t i = 0) const override { return m_base->degree(i); }
    short_t dim() const override { return m_base->dim(); }
    gsMatrix<T> boundingBox() const override { return m_base->boundingBox(); }

    std::ostream &print(std::ostream &os) const override
    {
        os << "Element range ["<<m_first<<","<<m_last<<") of: ";
        return m_base->print(os);
    }
};

} // namespace gismo
