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
 * @file TimeHistoryCollection.hpp
 */

#ifndef GEOSX_FILEIO_TIMEHISTORY_TIMEHISTORYCOLLECTION_HPP_
#define GEOSX_FILEIO_TIMEHISTORY_TIMEHISTORYCOLLECTION_HPP_

#include "dataRepository/BufferOpsDevice.hpp"
#include "dataRepository/HistoryDataSpec.hpp"
#include "events/tasks/TaskBase.hpp"
#include "mesh/DomainPartition.hpp"

#include <functional>

namespace geosx
{

class DomainPartition;

using namespace dataRepository;

/**
 * @class HistoryCollection
 *
 * A task class for serializing time history data into a buffer for later I/O.
 */
class HistoryCollection : public TaskBase
{
public:
  /// @copydoc geosx::dataRepository::Group::Group(string const & name, Group * const parent)
  HistoryCollection( string const & name, Group * parent ):
    TaskBase( name, parent )
  {   }

  // Forwarding public...
  void initializePostSubGroups() override {};

  /**
   * @brief Get the number of discrete collection operations this collector conducts.
   * @return The number of collection operations for this collector.
   */
  virtual localIndex getCollectionCount() const = 0;

  /**
   * @brief Get the metadata for what this collector collects.
   * @param domain The DomainPartition.
   * @param collectionIdx Which collected item to get metadata for.
   * @return A HistoryMetadata object describing  the history data being collected by this collector.
   */
  virtual HistoryMetadata getMetadata( DomainPartition const & domain, localIndex collectionIdx ) = 0;

  /**
   * @brief Get the name of the object being targeted for collection.
   * @return The collection target's name
   */
  virtual const string & getTargetName() const = 0;

  /**
   * @brief Register a callback that gives the current head of the time history data buffer.
   * @param collectionIdx Which collection item to register the buffer callback for.
   * @param bufferCall A functional that when invoked returns a pointer to the head of a buffer at least large enough to
   *                    serialize one timestep of history data into.
   * @note This is typically meant to callback to BufferedHistoryIO::GetBufferHead( )
   */
  virtual void registerBufferCall( localIndex collectionIdx, std::function< buffer_unit_type *() > bufferCall ) = 0;

  /**
   * @brief Get a metadata object relating the the Time variable itself.
   * @return A HistroyMetadata object describing the Time variable.
   */
  virtual HistoryMetadata getTimeMetadata() const = 0;

  /**
   * @brief Register a callback that gives the current head of the time data buffer.
   * @param timeBufferCall A functional that when invoked returns a pointer to the head of a buffer at least large enough to
   *                       serialize one instance of the Time variable into.
   * @note This is typically meant to callback to BufferedHistoryIO::GetBufferHead( )
   */
  virtual void registerTimeBufferCall( std::function< buffer_unit_type *() > timeBufferCall ) = 0;

  /**
   * @brief Get the number of collectors of meta-information (set indices, etc) writing time-independent information during initialization.
   * @return The number of collectors of meta-information for this collector.
   */
  virtual localIndex numMetaDataCollectors() const = 0;

  /**
   * @brief Get a pointer to a collector of meta-information for this collector.
   * @param metaIdx Which of the meta-info collectors to return. (see HistoryCollection::numMetaDataCollectors()).
   * @return A unique pointer to the HistoryCollection object used for meta-info collection. Intented to fall out of scope and desctruct
   * immediately after being used to perform output during simulation initialization.
   */
  virtual HistoryCollection & getMetaCollector( localIndex metaIdx ) = 0;

  /**
   * @brief Update the indices from the sets being collected.
   * @param domain The DomainPartition of the problem.
   */
  virtual void updateSetsIndices( DomainPartition & domain ) = 0;
};

}

#endif
