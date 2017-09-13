/*
 * Logger.cpp
 *
 *  Created on: Aug 31, 2017
 *      Author: settgast
 */

#include "Logger.hpp"
#include <mpi.h>
#include <stdlib.h>

namespace geosx
{

void abort()
{
#if USE_MPI == 1
  int mpi = 0;
  MPI_Initialized( &mpi );
  if ( mpi )
  {
    MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
  }
  else
#endif
  {
    exit( EXIT_FAILURE );
  }
}
}
