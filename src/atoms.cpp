#include "atoms.h"


#include <map>
#include <string>
#include <Eigen/Dense>


using Positions_t = Eigen::Array3Xd; 
using Velocities_t = Eigen::Array3Xd; 
using Forces_t = Eigen::Array3Xd;
using Names_t = Eigen::Array<std::string, Eigen::Dynamic, 1>;
using Masses_t = Eigen::ArrayXd;
using Stress_t = Eigen::Array3Xd;

// Molar Masses for small selection of elements
std::map<std::string, double> table_of_mass =
std::map<std::string, double>({
    {"H", 1.008},
    {"He", 4.0026},
    {"O", 15.999},
    {"C", 12.011},
    {"Au", 196.96657}
});

double get_mass(const std::string &name)
{
    // check if name is in table of mass. If it isn't just return 1.0
    if (table_of_mass.find(name) == table_of_mass.end())
    {
        return 1.0; // * (1.0 / (1000/9.649));
    }
    return table_of_mass.at(name); // * (1.0 / (1000/9.649));
};

Atoms::Atoms(): 
    positions{3, 1}, velocities{3, 1}, forces{3, 1}, stress{3, 1}
{ 
    masses.resize(1);
    masses.setConstant(1.0);
    names.resize(positions.cols());
    names.setConstant("H");
    positions.setZero(); 
    velocities.setZero(); 
    forces.setZero();
    stress.setZero();
}

Atoms::Atoms(size_t nb_atoms): 
    positions{3, nb_atoms}, velocities{3, nb_atoms}, forces{3, nb_atoms}, stress{3, nb_atoms}
{ 
    masses.resize(nb_atoms);
    masses.setConstant(1.0);
    names.resize(nb_atoms);
    names.setConstant("H");
    positions.setZero(); 
    velocities.setZero(); 
    forces.setZero();
    stress.setZero();
}
 
Atoms::Atoms(const Positions_t &p): 
    positions{p}, velocities{3, p.cols()}, forces{3, p.cols()}, stress{3, p.cols()}
{
    names.resize(p.cols());
    names.setConstant("H");
    masses.resize(p.cols());
    masses.setConstant(1.0);
    velocities.setZero(); 
    forces.setZero(); 
    stress.setZero();
} 

Atoms::Atoms(const Names_t &n, const Positions_t &p): 
    names{n}, positions{p}, velocities{3, p.cols()}, forces{3, p.cols()}, stress{3, p.cols()}
{
    // Try to get the mass of the atom from the name
    masses.resize(p.cols());
    for (size_t i = 0; i < (size_t) p.cols(); i++)
    {
        masses[i] = get_mass(n[i]);
    }
    velocities.setZero(); 
    assert(p.cols() == velocities.cols());
    forces.setZero();
    stress.setZero();
} 

Atoms::Atoms(const Positions_t &p, const Velocities_t &v): 
    positions{p}, velocities{v}, forces{3, p.cols()}, stress{3, p.cols()}
{
    masses.resize(p.cols());
    masses.setConstant(1.0);
    names.resize(p.cols());
    names.setConstant("H");
    assert(p.cols() == v.cols());
    forces.setZero();
    stress.setZero();
}

Atoms::Atoms(const Names_t &n, const Positions_t &p, const Velocities_t &v): 
    names{n}, positions{p}, velocities{v}, forces{3, p.cols()}, stress{3, p.cols()}
{
    // Try to get the mass of the atom from the name
    masses.resize(p.cols());
    for (size_t i = 0; i < (size_t) p.cols(); i++)
    {
        masses[i] = get_mass(n[i]);
    }
    assert(p.cols() == v.cols());
    forces.setZero();
    stress.setZero();
} 

Atoms::Atoms(const Names_t &n, const Positions_t &p, const Velocities_t &v, const Forces_t &f, double m): 
    names{n}, positions{p}, velocities{v}, forces{f}, stress{3, p.cols()}
{ 
    masses.resize(p.cols());
    masses.setConstant(m);
    assert(p.cols() == v.cols());
    stress.setZero();
}

Atoms::Atoms(const Names_t &n, const Positions_t &p, const Velocities_t &v, const Forces_t &f, Masses_t &m_t): 
    names{n}, positions{p}, velocities{v}, forces{f}, masses{m_t}, stress{3, p.cols()}
{
    assert(p.cols() == v.cols());
    stress.setZero();
}

size_t Atoms::nb_atoms() const
{ 
    return positions.cols(); 
}

void Atoms::resize(size_t size)
{
    names.resize(size);
    positions.resize(3, size);
    velocities.resize(3, size);
    forces.resize(3, size);
    masses.resize(size);
}

void Atoms::conservativeResize(size_t size)
{
    names.conservativeResize(size);
    positions.conservativeResize(3, size);
    velocities.conservativeResize(3, size);
    forces.conservativeResize(3, size);
    masses.conservativeResize(size);
}

double Atoms::get_avg_mass()
{
    double avg_mass = 0.0;
    for (size_t i = 0; i < (size_t) masses.size(); i++)
    {
        avg_mass += masses[i];
    }
    return avg_mass / masses.size();
}

double Atoms::get_avg_mass_local(size_t n)
{
    double avg_mass = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        avg_mass += masses[i];
    }
    return avg_mass / n;
}

double Atoms::get_boltzmann()
{
    // within lennard-jones units, the boltzmann constant is 1.0
    return 1.0;
}
