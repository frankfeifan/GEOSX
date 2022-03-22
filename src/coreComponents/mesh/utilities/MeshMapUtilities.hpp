/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2019 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2019 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2019 TotalEnergies
 * Copyright (c) 2019-     GEOSX Contributors
 * All right reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 * @file MeshMapUtilities.hpp
 */
#ifndef GEOSX_MESH_UTILITIES_MESHMAPUTILITIES_HPP
#define GEOSX_MESH_UTILITIES_MESHMAPUTILITIES_HPP

#include "common/DataTypes.hpp"
#include "mesh/ElementRegionManager.hpp"


namespace geosx
{

/**
 * @brief This namespace contains helper functions that facilitate access
 *        into the assortment of maps used by GEOSX mesh object managers
 *        (e.g. array2d/array1d(array1d)/ArrayOfArrays/ArrayOfSets, etc.)
 */
namespace meshMapUtilities
{

//////////////////////////////////////////////////////////////////////////////////

/**
 * @brief @return the size of the map along first dimension
 * @tparam T type of map element
 * @tparam USD unit-stride dimension of the map
 * @param map reference to the map
 */
template< typename T, int USD >
GEOSX_HOST_DEVICE
inline localIndex size0( arrayView2d< T, USD > const & map )
{
  return map.size( 0 );
}

/**
 * @brief @return the size of the map along first dimension
 * @tparam T type of map element
 * @param map reference to the map
 */
template< typename T >
GEOSX_HOST_DEVICE
inline localIndex size0( ArrayOfArraysView< T > const & map )
{
  return map.size();
}

/**
 * @brief @return the size of the map along first dimension
 * @tparam T type of map element
 * @param map reference to the map
 */
template< typename T >
GEOSX_HOST_DEVICE
inline localIndex size0( ArrayOfSetsView< T > const & map )
{
  return map.size();
}

//////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Transposes an input map (array2d, ArrayOfArrays or ArrayOfSets)
 * @tparam POLICY execution policy to use
 * @tparam VIEW_TYPE type of view of the source map
 * @param srcMap the source map
 * @param dstSize number of target objects ("cols" of @p srcMap)
 * @param overAlloc overallocation (extra capacity per row) for resulting map
 * @return the transpose of @p srcMap stored as ArrayOfArrays (most general form)
 */
template< typename POLICY, typename VIEW_TYPE >
ArrayOfArrays< std::remove_const_t< typename VIEW_TYPE::ValueType > >
transposeIndexMap( VIEW_TYPE const & srcMap,
                   localIndex const dstSize,
                   localIndex const overAlloc = 0 )
{
  // Count the number of elements in each set
  array1d< localIndex > counts( dstSize );
  counts.setValues< POLICY >( overAlloc );
  forAll< POLICY >( size0( srcMap ), [srcMap, counts = counts.toView()] ( localIndex const srcIndex )
  {
    for( localIndex const dstIndex : srcMap[ srcIndex ] )
    {
      RAJA::atomicInc< AtomicPolicy< POLICY > >( &counts[ dstIndex ] );
    }
  } );

  // Allocate storage for the transpose map
  ArrayOfArrays< localIndex > dstMap;
  dstMap.resizeFromCapacities< parallelHostPolicy >( dstSize, counts.data() );

  // Fill the sub-arrays with unsorted entries
  forAll< POLICY >( size0( srcMap ), [srcMap, dstMap = dstMap.toView()] ( localIndex const srcIndex )
  {
    for( localIndex const dstIndex : srcMap[ srcIndex ] )
    {
      dstMap.emplaceBackAtomic< AtomicPolicy< POLICY > >( dstIndex, srcIndex );
    }
  } );

  return dstMap;
}

namespace internal
{

/// Element identifier containing (region index, subregion index, element index).
using ElementTuple = std::tuple< localIndex, localIndex, localIndex >;

template< typename POLICY >
void convertFromElementTupleMap( ArrayOfArraysView< ElementTuple const > const & elemList,
                                 OrderedVariableToManyElementRelation & toElement )
{
  ArrayOfArrays< localIndex > & toElementRegionList = toElement.m_toElementRegion;
  ArrayOfArrays< localIndex > & toElementSubRegionList = toElement.m_toElementSubRegion;
  ArrayOfArrays< localIndex > & toElementList = toElement.m_toElementIndex;

  localIndex const numObjects = elemList.size();

  toElementRegionList.resizeFromOffsets( numObjects, elemList.toViewConst().getOffsets() );
  toElementSubRegionList.resizeFromOffsets( numObjects, elemList.toViewConst().getOffsets() );
  toElementList.resizeFromOffsets( numObjects, elemList.toViewConst().getOffsets() );

  forAll< parallelHostPolicy >( numObjects, [elemList = elemList.toViewConst(),
                                             elemRegion = toElementRegionList.toView(),
                                             elemSubRegion = toElementSubRegionList.toView(),
                                             elemIndex = toElementList.toView()]( localIndex const objIndex )
  {
    arraySlice1d< ElementTuple const > const elems = elemList[ objIndex ];
    for( ElementTuple const & e : elems )
    {
      elemRegion.emplaceBack( objIndex, std::get< 0 >( e ) );
      elemSubRegion.emplaceBack( objIndex, std::get< 1 >( e ) );
      elemIndex.emplaceBack( objIndex, std::get< 2 >( e ) );
    }
  } );
}

template< typename POLICY >
void convertFromElementTupleMap( ArrayOfArraysView< ElementTuple const > const & elemList,
                                 FixedToManyElementRelation & toElement )
{
  array2d< localIndex > & toElementRegionList = toElement.m_toElementRegion;
  array2d< localIndex > & toElementSubRegionList = toElement.m_toElementSubRegion;
  array2d< localIndex > & toElementList = toElement.m_toElementIndex;

  localIndex const numObjects = elemList.size();
  localIndex const maxNumElem = toElementList.size( 1 );

  GEOSX_ERROR_IF_NE( toElementRegionList.size( 1 ), maxNumElem );
  GEOSX_ERROR_IF_NE( toElementSubRegionList.size( 1 ), maxNumElem );

  toElementRegionList.resizeDimension< 0 >( numObjects );
  toElementSubRegionList.resizeDimension< 0 >( numObjects );
  toElementList.resizeDimension< 0 >( numObjects );

  // We allow a fixed-size map to represent a variable relationship, as long as
  // the number of elements does not exceed the fixed size (set by the caller).
  // In this case, a dummy value "-1" is used to represent non-present elements.
  toElementRegionList.setValues< POLICY >( -1 );
  toElementSubRegionList.setValues< POLICY >( -1 );
  toElementList.setValues< POLICY >( -1 );

  forAll< parallelHostPolicy >( numObjects, [=, // needed to optionally capture maxNumElem in Debug
                                             elemList = elemList.toViewConst(),
                                             elemRegion = toElementRegionList.toView(),
                                             elemSubRegion = toElementSubRegionList.toView(),
                                             elemIndex = toElementList.toView()]( localIndex const objIndex )
  {
    arraySlice1d< ElementTuple const > const elems = elemList[ objIndex ];
    GEOSX_ASSERT_GE( maxNumElem, elems.size() );
    localIndex count = 0;
    for( ElementTuple const & e : elems )
    {
      elemRegion( objIndex, count ) = std::get< 0 >( e );
      elemSubRegion( objIndex, count ) = std::get< 1 >( e );
      elemIndex( objIndex, count ) = std::get< 2 >( e );
      ++count;
    }
  } );
}

}

/**
 * @brief Build to-element map by inverting existing maps in element subregions.
 * @tparam BASEMAP underlying type of to-element map
 * @tparam FUNC type of @p cellToObjectGetter
 * @param elementRegionManager element region manager
 * @param numObjects number of objects (nodes, faces, etc.) for which maps are built
 * @param toElements container for to-element (region, subregion, index) maps
 * @param cellToObjectGetter function used to extract maps from subregions
 * @param overAlloc overallocation for the resulting maps
 *                  (extra capacity per row, only meaningful for variable-secondary-size containers)
 */
template< typename BASEMAP, typename FUNC >
void buildToElementMaps( ElementRegionManager const & elementRegionManager,
                         localIndex const numObjects,
                         ToElementRelation< BASEMAP > & toElements,
                         FUNC cellToObjectGetter,
                         localIndex const overAlloc = 0 )
{
  // Calculate the number of entries in each sub-array
  array1d< localIndex > elemCounts( numObjects );
  elemCounts.setValues< serialPolicy >( overAlloc );
  elementRegionManager.forElementSubRegionsComplete< CellElementSubRegion >( [&]( localIndex const,
                                                                                  localIndex const,
                                                                                  ElementRegionBase const &,
                                                                                  CellElementSubRegion const & subRegion )
  {
    auto const elemToObject = cellToObjectGetter( subRegion );
    forAll< parallelHostPolicy >( subRegion.size(), [elemCounts = elemCounts.toView(),
                                                     elemToObject]( localIndex const ei )
    {
      auto const objects = elemToObject[ ei ];
      // can't use range-based for loop when slice is not contiguous
      for( localIndex i = 0; i < objects.size(); ++i )
      {
        RAJA::atomicInc< parallelHostAtomic >( &elemCounts[ objects[i] ] );
      }
    } );
  } );

  // Allocate memory
  ArrayOfArrays< internal::ElementTuple > elemList;
  elemList.resizeFromCapacities< parallelHostPolicy >( numObjects, elemCounts.data() );

  // Populate map of tuples in parallel
  elementRegionManager.forElementSubRegionsComplete< CellElementSubRegion >( [&]( localIndex const er,
                                                                                  localIndex const esr,
                                                                                  ElementRegionBase const &,
                                                                                  CellElementSubRegion const & subRegion )
  {
    auto const elemToObject = cellToObjectGetter( subRegion );
    forAll< parallelHostPolicy >( subRegion.size(), [elemList = elemList.toView(),
                                                     elemToObject, er, esr]( localIndex const ei )
    {
      auto const objects = elemToObject[ ei ];
      // can't use range-based for loop when slice is not contiguous
      for( localIndex i = 0; i < objects.size(); ++i )
      {
        elemList.emplaceBackAtomic< parallelHostAtomic >( objects[i], er, esr, ei );
      }
    } );
  } );

  // Sort each element list to ensure unique race-condition-free map order
  forAll< parallelHostPolicy >( numObjects, [elemList = elemList.toView()]( localIndex const objIndex )
  {
    arraySlice1d< internal::ElementTuple > const elems = elemList[ objIndex ];
    LvArray::sortedArrayManipulation::makeSorted( elems.begin(), elems.end() );
  } );

  // Finally, split arrays-of-tuples into separate arrays
  internal::convertFromElementTupleMap< parallelHostPolicy >( elemList.toViewConst(), toElements );
}

} // namespace meshMapUtilities

} // namespace geosx

#endif //GEOSX_MESH_UTILITIES_MESHMAPUTILITIES_HPP
