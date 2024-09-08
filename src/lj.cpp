#include "lj.h"
#include "atoms.h"
#include "neighbors.h"

#include <Eigen/Dense>



double lj(double r, double epsilon, double sigma)
{
    return 4 * epsilon * (pow(sigma / r, 12) - pow(sigma / r, 6));
}


double lj_d(double r, double epsilon, double sigma)
{
    return 4 * epsilon * ((12 * pow(sigma, 12) / pow(r, 13)) - (6 * pow(sigma, 6) / pow(r, 7)));
}


double lj_direct_summation(Atoms &atoms, NeighborList &neighbors, double epsilon, double sigma, double cutoff)
{
    double energy = 0.0;
    
    atoms.forces.setZero();

    for(auto [i, j] : neighbors)
    {
        if (i < j)
        {
            Eigen::Array3Xd r_ij{atoms.positions.col(i) - atoms.positions.col(j)};
            double r{sqrt(r_ij.square().sum())};
            
            energy += lj(r, epsilon, sigma);
            
            double lj_derivative{lj_d(r, epsilon, sigma)};
            atoms.forces.col(i) += lj_derivative * r_ij / r;
            atoms.forces.col(j) -= lj_derivative * r_ij / r;
        }
    }

    return energy;
}


double lj_direct_summation_alt(Atoms &atoms, NeighborList &neighbors, double epsilon, double sigma, double cutoff)
{
    double energy = 0.0;

    double sigma_over_cutoff = sigma / cutoff;
    double cutoff_energy = 4 * epsilon * (std::pow(sigma_over_cutoff, 12) - std::pow(sigma_over_cutoff, 6));
    
    atoms.forces.setZero();

    neighbors.update(atoms, cutoff);

    for(auto [i, j] : neighbors)
    {
        if (i < j)
        {
            Eigen::Array3d r_ij = atoms.positions.col(i) - atoms.positions.col(j);
            double dist = r_ij.matrix().norm();
            double sigma_over_r = sigma / dist;
            Eigen::Array3d force_ij = (r_ij/dist) * 24 * (epsilon / dist) * (2 * std::pow(sigma_over_r, 12) - std::pow(sigma_over_r, 6));
            energy += 4 * epsilon * (std::pow(sigma_over_r, 12) - std::pow(sigma_over_r, 6)) - cutoff_energy;

            atoms.forces.col(i) += force_ij;
            atoms.forces.col(j) -= force_ij;
        }
    }

    return energy;
}
