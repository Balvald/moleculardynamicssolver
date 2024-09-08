#ifndef __LJ_DIRECT_SUMMATION_H
#define __LJ_DIRECT_SUMMATION_H

#include "atoms.h"

using Positions_t = Eigen::Array3Xd; 
using Velocities_t = Eigen::Array3Xd; 
using Forces_t = Eigen::Array3Xd;
using Names_t = Eigen::Array<std::string, Eigen::Dynamic, 1>;


double lj_direct_summation(Atoms &atoms, double epsilon = 1.0, double sigma = 1.0);


#endif  // __LJ_DIRECT_SUMMATION_H