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
    string levelName = GEOSX_FMT( "{}_{}", name, p->label );
    m_builders.emplace_back( std::make_unique< MsrsbLevelBuilder< LAI > >( std::move( levelName ), *p ) );
  }
}

template< typename LAI >
void MsrsbCoupledLevelBuilder< LAI >::createSmoothers( bool const useBlock )
{
  auto const makeSmoother = [&]( auto const getSmoother ) -> std::unique_ptr< PreconditionerBase< LAI > >
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
        smoother->setupBlock( i, { comp }, getSmoother( *m_builders[i] ) );
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
                  ? makeSmoother( []( auto & b ) { return &b.presmoother(); } )
                  : std::make_unique< PreconditionerNull< LAI > >();
  m_postsmoother = preOrPost == PreOrPost::post || preOrPost == PreOrPost::both
                   ? makeSmoother( []( auto & b ) { return &b.postsmoother(); } )
                   : std::make_unique< PreconditionerNull< LAI > >();
}

template< typename LAI >
void MsrsbCoupledLevelBuilder< LAI >::initializeFineLevel( DomainPartition & domain,
                                                           DofManager const & dofManager,
                                                           MPI_Comm const & comm )
{
  localIndex numLocalRows = 0;
  for( size_t i = 0; i < m_builders.size(); ++i )
  {
    m_builders[i]->initializeFineLevel( domain, dofManager, comm );
    m_fields.push_back( { m_params.subParams[i]->fieldName, { m_builders[i]->numComp(), true } } );
    numLocalRows += m_builders[i]->matrix().numLocalRows();
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
  //localIndex numLocalRows = 0;
  for( size_t i = 0; i < m_builders.size(); ++i )
  {
    m_builders[i]->initializeCoarseLevel( *fine.m_builders[i] );
    m_fields.push_back( fine.m_fields[i] );
    //numLocalRows += m_builders[i]->matrix().numLocalRows();
  }

  localIndex const numLocalRows =
    std::accumulate( m_builders.begin(), m_builders.end(), localIndex{},
                     []( localIndex const s, auto const & b ){ return s + b->matrix().numLocalRows(); } );

  // Create a "fake" coarse matrix (no data, just correct sizes/comms), to be computed later
  m_matrix.createWithLocalSize( numLocalRows, numLocalRows, 0, fine.matrix().comm() );

  createSmoothers( false ); // TODO decide
}

template< typename LAI >
void MsrsbCoupledLevelBuilder< LAI >::compute( Matrix const & fineMatrix )
{
  std::vector< Matrix > blocks( m_builders.size() );

  for( size_t i = 0; i < m_builders.size(); ++i )
  {

    //m_builders[i]->compute( );
  }
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
