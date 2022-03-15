/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2019 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2019 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2019 Total, S.A
 * Copyright (c) 2019-     GEOSX Contributors
 * All right reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 * @file MsrsbCoupledLevelBuilder.cpp
 */

#include "MsrsbCoupledLevelBuilder.hpp"

#include "linearAlgebra/interfaces/InterfaceTypes.hpp"

#include "linearAlgebra/solvers/PreconditionerNull.hpp"

namespace geosx
{

namespace multiscale
{

template< typename LAI >
MsrsbCoupledLevelBuilder< LAI >::MsrsbCoupledLevelBuilder( string name,
                                                           LinearSolverParameters::Multiscale params )
  : LevelBuilderBase< LAI >( std::move( name ), std::move( params ) )
{
  GEOSX_ASSERT( !m_params.subParams.empty() );
  for( LinearSolverParameters::Multiscale const * const p : m_params.subParams )
  {
    GEOSX_ASSERT_MSG( p != nullptr, "Sub-preconditioner parameters for each block must be set by the solver" );
    m_builders.emplace_back( std::make_unique< MsrsbLevelBuilder< LAI > >( GEOSX_FMT( "{}_{}", name, p->label ), *p ) );
  }
}

template< typename LAI >
void MsrsbCoupledLevelBuilder< LAI >::createSmoothers( bool const useBlock )
{
  using PtrToSmootherGetter = PreconditionerBase< LAI > & ( MsrsbLevelBuilder< LAI >::* )();
  auto const makeSmoother = [&]( PtrToSmootherGetter const func ) -> std::unique_ptr< PreconditionerBase< LAI > >
  {
    if( useBlock )
    {
      GEOSX_ERROR_IF_NE_MSG( m_params.subParams.size(), 2, "More than 2 blocks are not supported yet" );
      auto smoother = std::make_unique< BlockPreconditioner< LAI > >( BlockShapeOption::UpperTriangular,
                                                                      SchurComplementOption::RowsumDiagonalProbing,
                                                                      BlockScalingOption::None );
      for( integer i = 0; i < m_params.subParams.size(); ++i )
      {
        LinearSolverParameters params;
        params.preconditionerType = m_params.subParams[i]->smoother.type;
        DofManager::SubComponent comp{ m_params.subParams[i]->fieldName, { m_builders[i]->numComp(), true } };
        smoother->setupBlock( i, { comp }, &( ( *m_builders[i] ).*func)() );
        // TODO: reverse block order for post-smoother?
      }
      return smoother;
    }
    else
    {
      LinearSolverParameters params;
      params.preconditionerType = m_params.smoother.type;
      return LAI::createPreconditioner( params );
    }
  };

  using PreOrPost = LinearSolverParameters::AMG::PreOrPost;
  PreOrPost const & preOrPost = m_params.smoother.preOrPost;

  m_presmoother = preOrPost == PreOrPost::pre || preOrPost == PreOrPost::both
                  ? makeSmoother( &MsrsbLevelBuilder< LAI >::presmoother )
                  : std::make_unique< PreconditionerNull< LAI > >();
  m_postsmoother = preOrPost == PreOrPost::post || preOrPost == PreOrPost::both
                   ? makeSmoother( &MsrsbLevelBuilder< LAI >::postsmoother )
                   : std::make_unique< PreconditionerNull< LAI > >();
}

template< typename LAI >
void MsrsbCoupledLevelBuilder< LAI >::initializeFineLevel( DomainPartition & domain,
                                                           DofManager const & dofManager,
                                                           MPI_Comm const & comm )
{
  localIndex numLocalRows = 0;
  for( auto & builder : m_builders )
  {
    builder->initializeFineLevel( domain, dofManager, comm );
    numLocalRows += builder->matrix().numLocalRows();
  }

  // Create a "fake" fine matrix (no data, just correct sizes/comms for use at coarse level init)
  m_matrix.createWithLocalSize( numLocalRows, numLocalRows, 0, comm );

  createSmoothers( true ); // TODO decide
}

template< typename LAI >
void MsrsbCoupledLevelBuilder< LAI >::initializeCoarseLevel( LevelBuilderBase< LAI > & fine_level )
{
  MsrsbCoupledLevelBuilder< LAI > & fine = dynamicCast< MsrsbCoupledLevelBuilder< LAI > & >( fine_level );
  GEOSX_ASSERT( fine.m_builders.size() == m_builders.size() );
  localIndex numLocalRows = 0;
  for( size_t i = 0; i < m_builders.size(); ++i )
  {
    m_builders[i]->initializeCoarseLevel( *fine.m_builders[i] );
    numLocalRows += m_builders[i]->matrix().numLocalRows();
  }

  // Create a "fake" coarse matrix (no data, just correct sizes/comms), to be computed later
  m_matrix.createWithLocalSize( numLocalRows, numLocalRows, 0, fine.matrix().comm() );

  createSmoothers( false ); // TODO decide
}

template< typename LAI >
void MsrsbCoupledLevelBuilder< LAI >::compute( Matrix const & fineMatrix )
{
  GEOSX_UNUSED_VAR( fineMatrix );
  // TODO
}

// -----------------------
// Explicit Instantiations
// -----------------------
#ifdef GEOSX_USE_TRILINOS
template class MsrsbCoupledLevelBuilder< TrilinosInterface >;
#endif

#ifdef GEOSX_USE_HYPRE
template class MsrsbCoupledLevelBuilder< HypreInterface >;
#endif

#ifdef GEOSX_USE_PETSC
template class MsrsbCoupledLevelBuilder< PetscInterface >;
#endif

} // namespace multiscale

} // namespace geosx
