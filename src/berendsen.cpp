#include "berendsen.h"

#include <mpi.h>
#include "mpi_support.h"

double _get_temperature(Atoms &atoms)
{
    return atoms.get_avg_mass() / atoms.get_boltzmann() * atoms.velocities.square().sum() * 1.0 / (3.0 * static_cast<double>(atoms.nb_atoms()));
}

double _get_temperature_local(Atoms &atoms, size_t n)
{
    if (n == 0)
    {
        return 0.0;
    }
    return atoms.get_avg_mass_local(n) / atoms.get_boltzmann() * atoms.velocities.leftCols(n).square().sum() * 1.0 / (3.0 * static_cast<double>(n));
}

double kin_energy(Atoms &atoms)
{
    return atoms.get_avg_mass() * atoms.velocities.square().sum() * 0.5;
}

double kin_energy_local(Atoms &atoms, int n)
{
    if (n == 0)
    {
        return 0.0;
    }
    return atoms.get_avg_mass_local((size_t) n) * atoms.velocities.leftCols(n).square().sum() * 0.5;
}

void berendsen_thermostat(Atoms &atoms, double temperature, double timestep, double relaxation_time)
{
    double tau = relaxation_time;
    // double k_b = 1.38064852e-23; // Boltzmann constant
    double T0 = temperature;
    double dt = timestep;
    double T = _get_temperature(atoms);
    double lambda = sqrt(1.0 + (T0 / T - 1.0) * (dt / tau));
    atoms.velocities *= lambda;
}

void berendsen_thermostat_local(Atoms &atoms, double temperature, double timestep, double relaxation_time, size_t n)
{
    double tau = relaxation_time;
    // double k_b = 1.38064852e-23; // Boltzmann constant
    double T0 = temperature;
    double dt = timestep;
    double T_local = _get_temperature_local(atoms, n);
    double T{MPI::allreduce(T_local, MPI_SUM, MPI_COMM_WORLD)};
    double lambda = sqrt(1.0 + (T0 / T - 1.0) * (dt / tau));
    atoms.velocities.leftCols(n) *= lambda;
}



void berendsen_thermostat_alt(Atoms &atoms, double temperature, double timestep, double relaxation_time)
{
    double tau = relaxation_time;
    double k_b = 1.38064852e-23; // Boltzmann constant
    double T0 = temperature;
    double dt = timestep;
    double T = 0.0; // atoms.velocities.
    for (size_t i = 0; i < atoms.nb_atoms(); i++)
    {
        T = T + (atoms.velocities.col(i).matrix().norm() * atoms.velocities.col(i).matrix().norm()) * atoms.masses.row(i).value(); //* atoms.mass.col(i);
    }
    T = T / (3.0 * atoms.nb_atoms()); // should I have boltzmann constant here?
    double lambda = sqrt(1.0 + ((dt / tau) * (T0 / T - 1.0)));
    atoms.velocities *= lambda;
}