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
 * @file EzrokhiBrineViscosity.hpp
 */

#ifndef GEOSX_CONSTITUTIVE_FLUID_PVTFUNCTIONS_EZROKHIBRINEVISCOSITY_HPP_
#define GEOSX_CONSTITUTIVE_FLUID_PVTFUNCTIONS_EZROKHIBRINEVISCOSITY_HPP_

#include "PVTFunctionBase.hpp"

#include "functions/TableFunction.hpp"

namespace geosx
{

namespace constitutive
{

namespace PVTProps
{

class EzrokhiBrineViscosityUpdate final : public PVTFunctionBaseUpdate
{
public:

  EzrokhiBrineViscosityUpdate( arrayView1d< real64 const > const & componentMolarWeight,
                               TableFunction const & waterViscosityTable,
                               integer const CO2Index,
                               integer const waterIndex,
                               real64 const coef0,
                               real64 const coef1,
                               real64 const coef2 )
    : PVTFunctionBaseUpdate( componentMolarWeight ),
    m_waterViscosityTable( waterViscosityTable.createKernelWrapper()),
    m_CO2Index( CO2Index ),
    m_waterIndex ( waterIndex ),
    m_coef0( coef0 ),
    m_coef1( coef1 ),
    m_coef2( coef2 )
  {}

  template< int USD1 >
  GEOSX_HOST_DEVICE
  void compute( real64 const & pressure,
                real64 const & temperature,
                arraySlice1d< real64 const, USD1 > const & phaseComposition,
                real64 & value,
                bool useMass ) const;

  template< int USD1, int USD2, int USD3, int USD4 >
  GEOSX_HOST_DEVICE
  void compute( real64 const & pressure,
                real64 const & temperature,
                arraySlice1d< real64 const, USD1 > const & phaseComposition,
                arraySlice1d< real64 const, USD2 > const & dPhaseComposition_dPressure,
                arraySlice1d< real64 const, USD2 > const & dPhaseComposition_dTemperature,
                arraySlice2d< real64 const, USD3 > const & dPhaseComposition_dGlobalCompFraction,
                real64 & value,
                real64 & dValue_dPressure,
                real64 & dValue_dTemperature,
                arraySlice1d< real64, USD4 > const & dValue_dGlobalCompFraction,
                bool useMass ) const;

  virtual void move( LvArray::MemorySpace const space, bool const touch ) override
  {
    PVTFunctionBaseUpdate::move( space, touch );
    m_waterViscosityTable.move( space, touch );
  }

protected:

  /// Table with water viscosity tabulated as a function (T)
  TableFunction::KernelWrapper m_waterViscosityTable;

  /// Index of the CO2 phase
  integer m_CO2Index;

  /// Index of the water phase
  integer m_waterIndex;

  real64 m_coef0;

  real64 m_coef1;

  real64 m_coef2;

};


class EzrokhiBrineViscosity : public PVTFunctionBase
{
public:

  EzrokhiBrineViscosity( string const & name,
                         string_array const & inputPara,
                         string_array const & componentNames,
                         array1d< real64 > const & componentMolarWeight );

  virtual ~EzrokhiBrineViscosity() override = default;

  static string catalogName() { return "EzrokhiBrineViscosity"; }

  virtual string getCatalogName() const override final { return catalogName(); }

  virtual PVTFunctionType functionType() const override
  {
    return PVTFunctionType::VISCOSITY;
  }

  /// Type of kernel wrapper for in-kernel update
  using KernelWrapper = EzrokhiBrineViscosityUpdate;

  /**
   * @brief Create an update kernel wrapper.
   * @return the wrapper
   */
  KernelWrapper createKernelWrapper() const;

private:

  void makeCoefficients( string_array const & inputPara );

  /// Table with water viscosity tabulated as a function (T)
  TableFunction const * m_waterViscosityTable;


  /// Index of the CO2 phase
  integer m_CO2Index;

  /// Index of the water phase
  integer m_waterIndex;

  real64 m_coef0;

  real64 m_coef1;

  real64 m_coef2;

};

template< int USD1 >
GEOSX_HOST_DEVICE
void EzrokhiBrineViscosityUpdate::compute( real64 const & pressure,
                                           real64 const & temperature,
                                           arraySlice1d< real64 const, USD1 > const & phaseComposition,
                                           real64 & value,
                                           bool useMass ) const
{
  GEOSX_UNUSED_VAR( pressure, useMass );
  real64 const waterVisc = m_waterViscosityTable.compute( &temperature );
  // we have to convert molar component phase fraction (phaseComposition[m_CO2Index]) to mass fraction
  real64 const massPhaseCompositionCO2 = phaseComposition[m_CO2Index] * m_componentMolarWeight[m_CO2Index] /
                                         ( phaseComposition[m_CO2Index] * m_componentMolarWeight[m_CO2Index] +
                                           phaseComposition[m_waterIndex] * m_componentMolarWeight[m_waterIndex]);


  value = ( m_coef0  + temperature * ( m_coef1 + m_coef2 * temperature ) ) * massPhaseCompositionCO2;
  value = waterVisc * pow( 10, value );
}

template< int USD1, int USD2, int USD3, int USD4 >
GEOSX_HOST_DEVICE
void EzrokhiBrineViscosityUpdate::compute( real64 const & pressure,
                                           real64 const & temperature,
                                           arraySlice1d< real64 const, USD1 > const & phaseComposition,
                                           arraySlice1d< real64 const, USD2 > const & dPhaseComposition_dPressure,
                                           arraySlice1d< real64 const, USD2 > const & dPhaseComposition_dTemperature,
                                           arraySlice2d< real64 const, USD3 > const & dPhaseComposition_dGlobalCompFraction,
                                           real64 & value,
                                           real64 & dValue_dPressure,
                                           real64 & dValue_dTemperature,
                                           arraySlice1d< real64, USD4 > const & dValue_dGlobalCompFraction,
                                           bool useMass ) const
{
  GEOSX_UNUSED_VAR( pressure, useMass );
  real64 waterVisc_dTemperature;
  real64 const waterVisc = m_waterViscosityTable.compute( &temperature, &waterVisc_dTemperature );

  real64 const coefPhaseComposition = m_coef0 + temperature * ( m_coef1 + m_coef2 * temperature );

  // we have to convert molar component phase fraction (phaseComposition[m_CO2Index]) to mass fraction
  real64 const waterMWInv = 1.0 / (phaseComposition[m_CO2Index] * m_componentMolarWeight[m_CO2Index]
                                   + phaseComposition[m_waterIndex] * m_componentMolarWeight[m_waterIndex]);

  real64 const massPhaseCompositionCO2 = phaseComposition[m_CO2Index] * m_componentMolarWeight[m_CO2Index] * waterMWInv;

  real64 const exponent = coefPhaseComposition * massPhaseCompositionCO2;

  real64 const exponent_dPressure = coefPhaseComposition * dPhaseComposition_dPressure[m_CO2Index];
  real64 const exponent_dTemperature = coefPhaseComposition * dPhaseComposition_dTemperature[m_CO2Index] +
                                       ( m_coef1 + 2 * m_coef2 * temperature) * massPhaseCompositionCO2;

  // compute only common part of derivatives w.r.t. CO2 and water phase compositions
  // later to be multiplied by (phaseComposition[m_waterIndex]) and ( -phaseComposition[m_CO2Index] ) respectively
  real64 const exponent_dPhaseComp = coefPhaseComposition * m_componentMolarWeight[m_CO2Index] * m_componentMolarWeight[m_waterIndex] * waterMWInv * waterMWInv;

  real64 const exponentPowered = pow( 10, exponent );

  value = waterVisc * exponentPowered;
  real64 const dValueCoef = log( 10 ) * value;

  real64 const dValue_dPhaseComp = dValueCoef * exponent_dPhaseComp;
  dValue_dPressure = dValueCoef * exponent_dPressure;
  dValue_dTemperature = dValueCoef * exponent_dTemperature + waterVisc_dTemperature * exponentPowered;

  // here, we multiply common part of derivatives by specific coefficients
  dValue_dGlobalCompFraction[m_CO2Index] = dValue_dPhaseComp * phaseComposition[m_waterIndex] * dPhaseComposition_dGlobalCompFraction[m_CO2Index][m_CO2Index];
  dValue_dGlobalCompFraction[m_waterIndex] = dValue_dPhaseComp * ( -phaseComposition[m_CO2Index] ) * dPhaseComposition_dGlobalCompFraction[m_waterIndex][m_waterIndex];
}

} // end namespace PVTProps

} // end namespace constitutive

} // end namespace geosx

#endif //GEOSX_CONSTITUTIVE_FLUID_PVTFUNCTIONS_EZROKHIBRINEVISCOSITY_HPP_