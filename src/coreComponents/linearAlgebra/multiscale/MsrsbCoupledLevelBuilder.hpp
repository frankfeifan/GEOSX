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
 * @file MsrsbCoupledLevelBuilder.hpp
 */

#ifndef GEOSX_MSRSBCOUPLEDLEVELBUILDER_HPP
#define GEOSX_MSRSBCOUPLEDLEVELBUILDER_HPP

#include "linearAlgebra/multiscale/MsrsbLevelBuilder.hpp"
#include "linearAlgebra/solvers/BlockPreconditioner.hpp"

namespace geosx
{

namespace multiscale
{

template< typename LAI >
class MsrsbCoupledLevelBuilder : public LevelBuilderBase< LAI >
{

public:

  /// Alias for base type
  using Base = LevelBuilderBase< LAI >;

  /// Alias for vector type
  using Vector = typename Base::Vector;

  /// Alias for matrix type
  using Matrix = typename Base::Matrix;

  /// Alias for operator type
  using Operator = typename Base::Operator;

  explicit MsrsbCoupledLevelBuilder( string name, LinearSolverParameters::Multiscale params );

  virtual Operator const & prolongation() const override
  {
    return m_prolongation;
  }

  virtual Operator const & restriction() const override
  {
    return *m_restriction;
  }

  virtual Matrix const & matrix() const override
  {
    return m_matrix;
  }

  virtual PreconditionerBase< LAI > const & presmoother() const override
  {
    return *m_presmoother;
  }

  virtual PreconditionerBase< LAI > const & postsmoother() const override
  {
    return *m_postsmoother;
  }

  virtual void initializeFineLevel( DomainPartition & domain,
                                    DofManager const & dofManager,
                                    MPI_Comm const & comm ) override;

  virtual void initializeCoarseLevel( LevelBuilderBase< LAI > & fine_level ) override;

  virtual void compute( Matrix const & fineMatrix ) override;

private:

  void createSmoothers( bool const useBlock );

  using Base::m_params;
  using Base::m_name;

  /// Prolongation matrix P
  Matrix m_prolongation;

  /// Restriction (kept as abstract operator to allow for memory efficiency, e.g. when R = P^T)
  std::unique_ptr< Operator > m_restriction;

  /// Level operator matrix
  Matrix m_matrix;

  /// Levels for each sub-problem
  std::vector< std::unique_ptr< MsrsbLevelBuilder< LAI > > > m_builders;

  /// Pre-smoother
  std::unique_ptr< PreconditionerBase< LAI > > m_presmoother;

  /// Post-smoother
  std::unique_ptr< PreconditionerBase< LAI > > m_postsmoother;

};

} // namespace multiscale

} // namespace geosx

#endif //GEOSX_MSRSBCOUPLEDLEVELBUILDER_HPP
