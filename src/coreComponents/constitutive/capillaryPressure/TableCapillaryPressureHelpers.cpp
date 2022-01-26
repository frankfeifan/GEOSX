/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2020 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2020 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2020 TotalEnergies
 * Copyright (c) 2019-     GEOSX Contributors
 * All rights reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 * @file TableCapillaryPressureHelpers.cpp
 */

#include "TableCapillaryPressureHelpers.hpp"

#include "common/DataTypes.hpp"

namespace geosx
{

namespace constitutive
{

void
TableCapillaryPressureHelpers::validateCapillaryPressureTable( TableFunction const & capPresTable,
                                                               string const & fullConstitutiveName,
                                                               bool const capPresMustBeIncreasing )
{
  ArrayOfArraysView< real64 const > coords = capPresTable.getCoordinates();

  GEOSX_THROW_IF_NE_MSG( capPresTable.getInterpolationMethod(), TableFunction::InterpolationType::Linear,
                         GEOSX_FMT( "{}: in table '{}' interpolation method must be linear", fullConstitutiveName, capPresTable.getName() ),
                         InputError );
  GEOSX_THROW_IF_NE_MSG( capPresTable.numDimensions(), 1,
                         GEOSX_FMT( "{}: table '{}' must have a single independent coordinate", fullConstitutiveName, capPresTable.getName() ),
                         InputError );
  GEOSX_THROW_IF_LT_MSG( coords.sizeOfArray( 0 ), 2,
                         GEOSX_FMT( "{}: table `{}` must contain at least two values", fullConstitutiveName, capPresTable.getName() ),
                         InputError );

  arraySlice1d< real64 const > phaseVolFrac = coords[0];
  arrayView1d< real64 const > const capPres = capPresTable.getValues();

  for( localIndex i = 1; i < coords.sizeOfArray( 0 ); ++i )
  {
    // check phase volume fraction
    GEOSX_THROW_IF( phaseVolFrac[i] < 0 || phaseVolFrac[i] > 1,
                    GEOSX_FMT( "{}: in table '{}' values must be between 0 and 1", fullConstitutiveName, capPresTable.getName() ),
                    InputError );

    // note that the TableFunction class has already checked that the coordinates are monotone

    // check the monotonicity of the capillary pressure table
    if( capPresMustBeIncreasing )
    {
      GEOSX_THROW_IF( !isZero( capPres[i] ) && (capPres[i] - capPres[i-1]) < -1e-10,
                      GEOSX_FMT( "{}: in table '{}' values must be strictly increasing", fullConstitutiveName, capPresTable.getName() ),
                      InputError );
    }
    else
    {
      GEOSX_THROW_IF( !isZero( capPres[i] ) && (capPres[i] - capPres[i-1]) > 1e-10,
                      GEOSX_FMT( "{}: in table '{}' values must be strictly decreasing", fullConstitutiveName, capPresTable.getName() ),
                      InputError );
    }
  }
}


} // namespace constitutive

} // namespace geosx