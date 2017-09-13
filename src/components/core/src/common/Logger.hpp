/*
 * Logger.hpp
 *
 *  Created on: Jul 17, 2017
 *      Author: settgast1
 */

#ifndef SRC_COMPONENTS_CORE_SRC_COMMON_LOGGER_HPP_
#define SRC_COMPONENTS_CORE_SRC_COMMON_LOGGER_HPP_

#if ATK_FOUND
#include "slic/slic.hpp"
#include "slic/GenericOutputStream.hpp"
#endif


namespace geosx
{
void abort();

#if ATK_FOUND
#define GEOS_ERROR(msg) SLIC_ERROR(msg)
#else
#define GEOS_ERROR(msg) abort();
#endif

}

#endif /* SRC_COMPONENTS_CORE_SRC_COMMON_LOGGER_HPP_ */
