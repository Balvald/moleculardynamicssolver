/*
 * Copyright 2021 Lars Pastewka
 *
 * ### MIT license
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <iostream>

#include "ducastelle.h"

/*
 * This is the embedded atom method potential described in
 *     Ducastelle, "Modules élastiques des métaux de transition", J. Phys. 31,
 * 1055 (1970) Gupta, "Lattice relaxation at a metal surface", Phys. Rev. B 23,
 * 6265 (1981) Cleri, Rosato, "Tight-binding potentials for transition metals
 * and alloys", Phys. Rev. B 48, 22 (1993) The default values for the parameters
 * are the Au parameters from Cleri & Rosato's paper.
 */
double ducastelle(Atoms &atoms, const NeighborList &neighbor_list,
                  double cutoff, double A, double xi, double p, double q,
                  double re)
{
    auto cutoff_sq{cutoff * cutoff};
    double xi_sq{xi * xi};

    // Reset energies and forces. This needs to be turned off if multiple
    // potentials are present.
    atoms.forces.setZero();

    // std::cout << "Computing Ducastelle potential" << std::endl;

    // compute embedding energies
    Eigen::ArrayXd embedding(atoms.nb_atoms()); // contains first density, later energy
    embedding.setZero();
    for (auto [i, j] : neighbor_list) {
        if (i < j) {
            Eigen::Vector3d distance_vector{atoms.positions.col(i) -
                                            atoms.positions.col(j)};
            auto distance_sq = distance_vector.squaredNorm();
            if (distance_sq < cutoff_sq) {
                double density_contribution{
                    xi_sq *
                    std::exp(-2 * q * (std::sqrt(distance_sq) / re - 1.0))};
                embedding(i) += density_contribution;
                embedding(j) += density_contribution;
            }
        }
    }

    // std::cout << "After 1 iteration over neighbors" << std::endl;

    // compute embedding contribution to the potential energy
    embedding = -embedding.sqrt();

    // per-atom energies
    Eigen::ArrayXd energies{embedding};

    // compute forces
    for (auto [i, j] : neighbor_list) {
        if (i < j) {
            double d_embedding_density_i{0};
            // this is the derivative of sqrt(embedding)
            if (embedding(i) != 0)
                d_embedding_density_i = 1 / (2 * embedding(i));

            Eigen::Vector3d distance_vector{atoms.positions.col(i) -
                                            atoms.positions.col(j)};
            auto distance_sq = distance_vector.squaredNorm();
            if (distance_sq < cutoff_sq) {
                double distance{std::sqrt(distance_sq)};
                double d_embedding_density_j{0};
                // this is the derivative of sqrt(embedding)
                if (embedding(j) != 0)
                    d_embedding_density_j = 1 / (2 * embedding(j));

                // repulsive energy and derivative of it with respect to
                // distance
                double repulsive_energy{2 * A *
                                        std::exp(-p * (distance / re - 1.0))};
                double d_repulsive_energy{-repulsive_energy * p / re};

                // derivative of embedding energy contributions
                double fac{-2 * q / re * xi_sq *
                           std::exp(-2 * q * (distance / re - 1.0))};

                // pair force
                Eigen::Array3d pair_force{
                    (d_repulsive_energy +
                     fac * (d_embedding_density_i + d_embedding_density_j)) *
                    distance_vector.normalized()};

                // sum per-atom energies
                repulsive_energy *= 0.5;
                energies(i) += repulsive_energy;
                energies(j) += repulsive_energy;

                // sum per-atom forces
                atoms.forces.col(i) -= pair_force;
                atoms.forces.col(j) += pair_force;
            }
        }
    }

    // std::cout << "After 2 iteration over neighbors" << std::endl;


    // Return total potential energy
    return energies.sum();
}


/*
Essentially the same as ducastelle, but only returns a subset of the energies computed here.
*/
double ducastelle_local(Atoms &atoms, const NeighborList &neighbor_list, size_t n,
                         double cutoff, double A, double xi, double p, double q,
                         double re)
{
    auto cutoff_sq{cutoff * cutoff};
    double xi_sq{xi * xi};

    // Reset energies and forces. This needs to be turned off if multiple
    // potentials are present.
    atoms.forces.setZero();

    // compute embedding energies
    Eigen::ArrayXd embedding(atoms.nb_atoms()); // contains first density, later energy
    embedding.setZero();
    try
    {
        for (auto [i, j] : neighbor_list)
        {
            if (i < j)
            {
                Eigen::Vector3d distance_vector{atoms.positions.col(i) -
                                                atoms.positions.col(j)};
                auto distance_sq = distance_vector.squaredNorm();
                if (distance_sq < cutoff_sq)
                {
                    double density_contribution{xi_sq * std::exp(-2 * q * (std::sqrt(distance_sq) / re - 1.0))};
                    embedding(i) += density_contribution;
                    embedding(j) += density_contribution;
                }
            }
        }

    }
    catch(const std::exception& e)
    {
        std::cout << e.what() << '\n';
    }
    

    // compute embedding contribution to the potential energy
    embedding = -embedding.sqrt();

    // per-atom energies
    Eigen::ArrayXd energies{embedding};

    try
    {
  
        // compute forces
        for (auto [i, j] : neighbor_list) {
            if (i < j) {
                double d_embedding_density_i{0};
                // this is the derivative of sqrt(embedding)
                if (embedding(i) != 0)
                    d_embedding_density_i = 1 / (2 * embedding(i));

                Eigen::Vector3d distance_vector{atoms.positions.col(i) -
                                                atoms.positions.col(j)};
                auto distance_sq = distance_vector.squaredNorm();
                if (distance_sq < cutoff_sq) {
                    double distance{std::sqrt(distance_sq)};
                    double d_embedding_density_j{0};
                    // this is the derivative of sqrt(embedding)
                    if (embedding(j) != 0)
                        d_embedding_density_j = 1 / (2 * embedding(j));

                    // repulsive energy and derivative of it with respect to
                    // distance
                    double repulsive_energy{2 * A *
                                            std::exp(-p * (distance / re - 1.0))};
                    double d_repulsive_energy{-repulsive_energy * p / re};

                    // derivative of embedding energy contributions
                    double fac{-2 * q / re * xi_sq *
                            std::exp(-2 * q * (distance / re - 1.0))};

                    // pair force
                    Eigen::Array3d pair_force{
                        (d_repulsive_energy +
                        fac * (d_embedding_density_i + d_embedding_density_j)) *
                        distance_vector.normalized()};

                    // sum per-atom energies
                    repulsive_energy *= 0.5;
                    energies(i) += repulsive_energy;
                    energies(j) += repulsive_energy;

                    // sum per-atom forces
                    atoms.forces.col(i) -= pair_force;
                    atoms.forces.col(j) += pair_force;
                }
            }
        }

    }
    catch(const std::exception& e)
    {
        std::cout << e.what() << '\n';
    }
  
    // Return only energies of subset to avoid summation of ghost atoms.
    return energies.head(n).sum();
}

std::pair<double, Eigen::Matrix3d> ducastelle_local_stress(Atoms &atoms, const NeighborList &neighbor_list, size_t n,
                                                 double cutoff, double A, double xi, double p, double q,
                                                 double re)
{
    auto cutoff_sq{cutoff * cutoff};
    double xi_sq{xi * xi};

    // Reset energies and forces. This needs to be turned off if multiple
    // potentials are present.
    atoms.forces.setZero();
    Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();

    // compute embedding energies
    Eigen::ArrayXd embedding(atoms.nb_atoms()); // contains first density, later energy
    embedding.setZero();
    try
    {
        for (auto [i, j] : neighbor_list)
        {
            if (i < j)
            {
                Eigen::Vector3d distance_vector{atoms.positions.col(i) -
                                                atoms.positions.col(j)};
                auto distance_sq = distance_vector.squaredNorm();
                if (distance_sq < cutoff_sq)
                {
                    double density_contribution{xi_sq * std::exp(-2 * q * (std::sqrt(distance_sq) / re - 1.0))};
                    embedding(i) += density_contribution;
                    embedding(j) += density_contribution;
                }
            }
        }

    }
    catch(const std::exception& e)
    {
        std::cout << e.what() << '\n';
    }
    

    // compute embedding contribution to the potential energy
    embedding = -embedding.sqrt();

    // per-atom energies
    Eigen::ArrayXd energies{embedding};

    try
    {
  
        // compute forces
        for (auto [i, j] : neighbor_list) {
            if (i < j) {
                double d_embedding_density_i{0};
                // this is the derivative of sqrt(embedding)
                if (embedding(i) != 0)
                    d_embedding_density_i = 1 / (2 * embedding(i));

                Eigen::Vector3d distance_vector{atoms.positions.col(i) -
                                                atoms.positions.col(j)};
                auto distance_sq = distance_vector.squaredNorm();
                if (distance_sq < cutoff_sq) {
                    double distance{std::sqrt(distance_sq)};
                    double d_embedding_density_j{0};
                    // this is the derivative of sqrt(embedding)
                    if (embedding(j) != 0)
                        d_embedding_density_j = 1 / (2 * embedding(j));

                    // repulsive energy and derivative of it with respect to
                    // distance
                    double repulsive_energy{2 * A *
                                            std::exp(-p * (distance / re - 1.0))};
                    double d_repulsive_energy{-repulsive_energy * p / re};

                    // derivative of embedding energy contributions
                    double fac{-2 * q / re * xi_sq *
                            std::exp(-2 * q * (distance / re - 1.0))};

                    // pair force
                    Eigen::Array3d pair_force{
                        (d_repulsive_energy +
                        fac * (d_embedding_density_i + d_embedding_density_j)) *
                        distance_vector.normalized()};

                    // sum per-atom energies
                    repulsive_energy *= 0.5;
                    energies(i) += repulsive_energy;
                    energies(j) += repulsive_energy;

                    // sum per-atom forces
                    atoms.forces.col(i) -= pair_force;
                    atoms.forces.col(j) += pair_force;

                    double factor = 1.0;
                    if (j >= n)
                    {
                        // Contribution with Ghost atoms are only counted half
                        factor = 0.5;
                    }

                    // recast
                    Eigen::Vector3d pair_stress = pair_force;

                    if (i < n)
                    {
                        stress += (factor * distance_vector * (pair_stress.transpose()));
                    }
                }
            }
        }

    }
    catch(const std::exception& e)
    {
        std::cout << e.what() << '\n';
    }
  
    // Return only energies of subset to avoid summation of ghost atoms.
    return {energies.head(n).sum(), stress};
}
