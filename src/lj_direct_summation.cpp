#include "lj_direct_summation.h"
#include <Eigen/Dense>
#include "atoms.h"


double lj(double r, double epsilon, double sigma)
{
    return 4 * epsilon * (pow(sigma / r, 12) - pow(sigma / r, 6));
}


double lj_d(double r, double epsilon, double sigma)
{
    return 4 * epsilon * ((12 * pow(sigma, 12) / pow(r, 13)) - (6 * pow(sigma, 6) / pow(r, 7)));
}


double lj_direct_summation(Atoms &atoms, double epsilon, double sigma)
{
    double energy = 0.0;
    
    atoms.forces.setZero();

    for (size_t i = 0; i < atoms.nb_atoms(); i++)
    {
        for (size_t j = i + 1; j < atoms.nb_atoms(); j++)
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


double lj_direct_summation_alt(Atoms &atoms, double epsilon, double sigma)
{
    double energy = 0.0;
    
    atoms.forces.setZero();

    for (size_t i = 0; i < atoms.nb_atoms(); i++)
    {
        for (size_t j = i + 1; j < atoms.nb_atoms(); j++)
        {
            Eigen::Array3Xd r_ij = atoms.positions.col(i) - atoms.positions.col(j);
            double dist = r_ij.matrix().norm();
            double sigma_over_r = sigma / dist;
            double lj_potential = 4 * epsilon * (std::pow(sigma_over_r, 12) - std::pow(sigma_over_r, 6));
            Eigen::Array3Xd force_ij = (48 * epsilon * std::pow(sigma, 12) / std::pow(dist, 14) - 24 * epsilon * std::pow(sigma, 6) / std::pow(dist, 8)) * r_ij;
            atoms.forces.col(i) += force_ij;
            atoms.forces.col(j) -= force_ij;

            energy += lj_potential;
        }
        
    }
    

    return energy;
}