#ifndef __INITPOS_H
#define __INITPOS_H

#include <Eigen/Dense>
#include <iostream>
#include "atoms.h"

void init_cube(Atoms& atoms, double lattice_const=1.0)
{
    double cuberoot = std::pow(atoms.nb_atoms(), 1.0/3.0);
    size_t cutcb = (size_t) cuberoot;

    if (cutcb < cuberoot)
        cutcb += 1;

    for (size_t i = 0; i < cutcb; i++)
    {
        for (size_t j = 0; j < cutcb; j++)
        {
            for (size_t k = 0; k < cutcb; k++)
            {
                size_t index = i * cutcb * cutcb + j * cutcb + k;
                if (index < atoms.nb_atoms())
                {
                    atoms.positions.col(index) << 5+i*lattice_const, 5+j*lattice_const, 5+k*lattice_const;
                    std::cout << "Index: " << index << " x/i: " << 5+i*lattice_const << " y/j: " << 5+j*lattice_const << " z/k: " << 5+k*lattice_const << std::endl;
                    std::cout << "Position: " << atoms.positions.col(index) << std::endl;
                }
                else
                {
                    std::cout << "Index out of bounds" << std::endl;
                    std::cout << "x/i: " << i << " y/j: " << j << " z/k: " << k << std::endl;
                    std::cout << "x/i: " << 5+i*lattice_const << " y/j: " << 5+j*lattice_const << " z/k: " << k*lattice_const << std::endl;
                    std::cout << "This should only happen" << std::endl;
                    std::cout << "if " << cutcb*cutcb*cutcb << " > " << atoms.nb_atoms() << std::endl;
                }
            }
        }
    }
}

Atoms init_cube(size_t nb_atoms, double lattice_const=1.0)
{
    Atoms atoms(nb_atoms);

    init_cube(atoms, lattice_const);

    return atoms;
}

/*

FCC Cube done same as cube but to get the atoms inside each cubic element we assume for a lattice constant h
that the position the intermediate inner atoms lie is halfway to the next major layer of atoms.
To ensure most atoms in those intermediate layers end up in the middle of a cubic element we move them by 0.5h in always the same two axis directions.
This may mean that there is a single row of atoms at every intermeadiate step that pretrudes the main structure of the cube.

*/

void init_cube_fcc(Atoms& atoms, double lattice_const=1.0)
{
    double cuberoot = std::pow(atoms.nb_atoms(), 1.0/3.0);
    size_t cutcb = (size_t) cuberoot;

    double lattice_const_offset = lattice_const / 2.0;

    if (cutcb < cuberoot)
        cutcb += 1;

    for (size_t i = 0; i < cutcb; i++)
    {
        for (size_t j = 0; j < cutcb; j++)
        {
            for (size_t k = 0; k < cutcb; k++)
            {
                size_t index = i * cutcb * cutcb + j * cutcb + k;
                if (index < atoms.nb_atoms())
                {
                    if (i % 2 == 0)
                    {
                        atoms.positions.col(index) << 5+i*lattice_const, 5+j*lattice_const, 5+k*lattice_const;
                    }
                    else
                    {
                        atoms.positions.col(index) << 5+i*lattice_const, 5+j*lattice_const+lattice_const_offset, 5+k*lattice_const+lattice_const_offset;
                    }
                    std::cout << "Index: " << index << " x/i: " << 5+i*lattice_const << " y/j: " << 5+j*lattice_const << " z/k: " << 5+k*lattice_const << std::endl;
                    std::cout << "Position: " << atoms.positions.col(index) << std::endl;
                }
                else
                {
                    std::cout << "Index out of bounds" << std::endl;
                    std::cout << "x/i: " << i << " y/j: " << j << " z/k: " << k << std::endl;
                    if (i % 2 == 0)
                    {
                        std::cout << "x/i: " << 5+i*lattice_const << " y/j: " << 5+j*lattice_const << " z/k: " << 5+k*lattice_const << std::endl;
                    }
                    else
                    {
                        std::cout << "x/i: " << 5+i*lattice_const << " y/j: " << 5+j*lattice_const+lattice_const_offset << " z/k: " << 5+k*lattice_const+lattice_const_offset << std::endl;
                    }
                    std::cout << "x/i: " << 5+i*lattice_const << " y/j: " << 5+j*lattice_const << " z/k: " << 5+k*lattice_const << std::endl;
                    std::cout << "This should only happen" << std::endl;
                    std::cout << "if " << cutcb*cutcb*cutcb << " > " << atoms.nb_atoms() << std::endl;
                }
            }
        }
    }
}

Atoms init_cube_fcc(size_t nb_atoms, double lattice_const=1.0)
{
    Atoms atoms(nb_atoms);

    init_cube_fcc(atoms, lattice_const);

    return atoms;
}

#endif  // __INITPOS_H