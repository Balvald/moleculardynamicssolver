#ifndef __ATOMS_H
#define __ATOMS_H

#include <map>
#include <string>
#include <Eigen/Dense>


double get_mass(const std::string &name);


using Positions_t = Eigen::Array3Xd; 
using Velocities_t = Eigen::Array3Xd; 
using Forces_t = Eigen::Array3Xd;
using Names_t = Eigen::Array<std::string, Eigen::Dynamic, 1>;
using Masses_t = Eigen::ArrayXd;
using Stress_t = Eigen::Array3Xd;

class Atoms
{ 
public:
    Positions_t positions; 
    Velocities_t velocities; 
    Forces_t forces;
    Masses_t masses;
    Names_t names;
    Stress_t stress;

    Atoms();

    Atoms(size_t nb_atoms);
 
    Atoms(const Positions_t &p);

    Atoms(const Names_t &n, const Positions_t &p);

    Atoms(const Positions_t &p, const Velocities_t &v);

    Atoms(const Names_t &n, const Positions_t &p, const Velocities_t &v);
    Atoms(const Names_t &n, const Positions_t &p, const Velocities_t &v, const Forces_t &f, double m);

    Atoms(const Names_t &n, const Positions_t &p, const Velocities_t &v, const Forces_t &f, Masses_t &m_t);

    size_t nb_atoms() const;

    void resize(size_t size);

    void conservativeResize(size_t size);

    double get_avg_mass();

    double get_avg_mass_local(size_t n);
    
    double get_boltzmann();

};


#endif  // __ATOMS_H