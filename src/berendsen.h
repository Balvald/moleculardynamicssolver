#ifndef __BERENDSEN_H
#define __BERENDSEN_H

#include "atoms.h"


double kin_energy(Atoms &atoms);

double kin_energy_local(Atoms &atoms, int n);

void berendsen_thermostat(Atoms &atoms, double temperature, double timestep, double relaxation_time);

void berendsen_thermostat_local(Atoms &atoms, double temperature, double timestep, double relaxation_time, size_t n);

#endif  // __BERENDSEN_H