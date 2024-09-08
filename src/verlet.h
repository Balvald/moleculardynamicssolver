#ifndef __VERLET_H
#define __VERLET_H

#include <Eigen/Dense>

#include "atoms.h"


using Positions_t = Eigen::Array3Xd; 
using Velocities_t = Eigen::Array3Xd; 
using Forces_t = Eigen::Array3Xd;
using Names_t = Eigen::Array<std::string, Eigen::Dynamic, 1>;

// verlet steps for Milestone 1
void verlet_step1(double &x, double &y, double &z, double &vx, double &vy, double &vz,
                  double fx, double fy, double fz, double timestep);
void verlet_step2(double &vx, double &vy, double &vz, double fx, double fy, double fz,
                  double timestep);

// verlet steps for Milestone 2
void verlet_step1(Eigen::Array3Xd &positions, Eigen::Array3Xd &velocities, const Eigen::Array3Xd &forces, double timestep);
void verlet_step2(Eigen::Array3Xd &velocities, const Eigen::Array3Xd &forces, double timestep);

// verlet steps for Milestone 3
void verlet_step1(Atoms &atoms, double timestep);
void verlet_step2(Atoms &atoms, double timestep);

#endif  // __VERLET_H