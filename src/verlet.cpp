#include "verlet.h"

#include <iostream>

void verlet_step1(double &x, double &y, double &z,
    	          double &vx, double &vy, double &vz,
                  double fx, double fy, double fz,
                  double timestep)
{
    vx += 0.5 * fx * timestep / 1.0;  // 1.0 is the mass
    vy += 0.5 * fy * timestep / 1.0;
    vz += 0.5 * fz * timestep / 1.0;

    x += vx * timestep;
    y += vy * timestep;
    z += vz * timestep;
}

void verlet_step2(double &vx, double &vy, double &vz,
                  double fx, double fy, double fz,
                  double timestep)
{
    vx += 0.5 * fx * timestep / 1.0;  // 1.0 is the mass
    vy += 0.5 * fy * timestep / 1.0;
    vz += 0.5 * fz * timestep / 1.0;
}

void verlet_step1(Eigen::Array3Xd &positions, Eigen::Array3Xd &velocities, const Eigen::Array3Xd &forces, double timestep)
{
    velocities.row(0) += 0.5 * forces.row(0) * timestep / 1.0;  // 1.0 is the mass
    velocities.row(1) += 0.5 * forces.row(1) * timestep / 1.0;
    velocities.row(2) += 0.5 * forces.row(2) * timestep / 1.0;
    
    // std::cout << "forces: " << forces << std::endl;

    positions.row(0) += velocities.row(0) * timestep;
    positions.row(1) += velocities.row(1) * timestep;
    positions.row(2) += velocities.row(2) * timestep;

    //velocities += 0.5 * forces * timestep / 1.0;  // 1.0 is the mass

    //positions += velocities * timestep;

    /*
    vx += 0.5 * fx * timestep / 1.0;  // 1.0 is the mass
    vy += 0.5 * fy * timestep / 1.0;
    vz += 0.5 * fz * timestep / 1.0;

    x += vx * timestep;
    y += vy * timestep;
    z += vz * timestep;
    */
}

void verlet_step2(Eigen::Array3Xd &velocities, const Eigen::Array3Xd &forces, double timestep)
{
    // std::cout << "forces: " << forces << std::endl;

    velocities.row(0) += 0.5 * forces.row(0) * timestep / 1.0;  // 1.0 is the mass
    velocities.row(1) += 0.5 * forces.row(1) * timestep / 1.0;
    velocities.row(2) += 0.5 * forces.row(2) * timestep / 1.0;

    // velocities += 0.5 * forces * timestep / 1.0;  // 1.0 is the mass

    /*
    vx += 0.5 * fx * timestep / 1.0;  // 1.0 is the mass
    vy += 0.5 * fy * timestep / 1.0;
    vz += 0.5 * fz * timestep / 1.0;
    */
}

void verlet_step1(Atoms &atoms, double timestep)
{

    atoms.velocities += (0.5 * atoms.forces * timestep).rowwise() / atoms.masses.transpose();   // update velocities

    atoms.positions += atoms.velocities * timestep;            // update positions

    /*
    atoms.velocities.row(0) += 0.5 * atoms.forces.row(0) * timestep / 1.0;  // 1.0 is the mass
    atoms.velocities.row(1) += 0.5 * atoms.forces.row(1) * timestep / 1.0;
    atoms.velocities.row(2) += 0.5 * atoms.forces.row(2) * timestep / 1.0;
    
    atoms.positions.row(0) += atoms.velocities.row(0) * timestep;
    atoms.positions.row(1) += atoms.velocities.row(1) * timestep;
    atoms.positions.row(2) += atoms.velocities.row(2) * timestep;
    */


    //velocities += 0.5 * forces * timestep / 1.0;  // 1.0 is the mass

    //positions += velocities * timestep;

    /*
    vx += 0.5 * fx * timestep / 1.0;  // 1.0 is the mass
    vy += 0.5 * fy * timestep / 1.0;
    vz += 0.5 * fz * timestep / 1.0;

    x += vx * timestep;
    y += vy * timestep;
    z += vz * timestep;
    */
}

void verlet_step2(Atoms &atoms, double timestep)
{
    //atoms.velocities.row(0) += 0.5 * atoms.forces.row(0) * timestep / 1.0;  // 1.0 is the mass
    //atoms.velocities.row(1) += 0.5 * atoms.forces.row(1) * timestep / 1.0;
    //atoms.velocities.row(2) += 0.5 * atoms.forces.row(2) * timestep / 1.0;

    atoms.velocities += (0.5 * atoms.forces * timestep).rowwise() / atoms.masses.transpose();   // update velocities<

    // velocities += 0.5 * forces * timestep / 1.0;  // 1.0 is the mass

    /*
    vx += 0.5 * fx * timestep / 1.0;  // 1.0 is the mass
    vy += 0.5 * fy * timestep / 1.0;
    vz += 0.5 * fz * timestep / 1.0;
    */
}
