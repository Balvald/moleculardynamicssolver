#ifndef __LJ__H
#define __LJ__H

#include "neighbors.h"
#include "atoms.h"

using Positions_t = Eigen::Array3Xd; 
using Velocities_t = Eigen::Array3Xd; 
using Forces_t = Eigen::Array3Xd;
using Names_t = Eigen::Array<std::string, Eigen::Dynamic, 1>;


double lj_direct_summation(Atoms &atoms,  NeighborList &neighbors, double epsilon = 1.0, double sigma = 1.0, double cutoff = 0.5);


#endif  // __LJ__H