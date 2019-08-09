/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/**
 * @file InterObjectRelationship.hpp
 */

#ifndef INTEROBJECTRELATION_H_
#define INTEROBJECTRELATION_H_


//#include "Common/typedefs.h"
#include <map>

#include "managers/ObjectManagerBase.hpp"

namespace geosx
{
template < typename BASETYPE >
class InterObjectRelation : public BASETYPE
{
public:

  using base_type = BASETYPE;

  /// equals operator that sets *this to a single value of any type
  template<typename rTYPE> InterObjectRelation& operator=( const rTYPE& rhs )
  {
    BASETYPE::operator=(rhs);
    return (*this);
  }

  const base_type & Base() const { return static_cast<const BASETYPE&>(*this); }
  base_type & Base() { return dynamic_cast<BASETYPE&>(*this); }

  void SetRelatedObject( ObjectManagerBase const * const relatedObject )
  { m_relatedObject = relatedObject; }

  const ObjectManagerBase * RelatedObject() const
  { return m_relatedObject; }

  globalIndex_array const & RelatedObjectLocalToGlobal() const
  { return this->m_relatedObject->m_localToGlobalMap; }

  const unordered_map<globalIndex,localIndex>& RelatedObjectGlobalToLocal() const
  { return this->m_relatedObject->m_globalToLocalMap; }

private:
  ObjectManagerBase const * m_relatedObject = nullptr;
};

typedef InterObjectRelation<array2d<localIndex>>                FixedOneToManyRelation;
}

#endif /* INTEROBJECTRELATION_H_ */
