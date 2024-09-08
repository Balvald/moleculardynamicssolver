#include <gtest/gtest.h>
#include <math.h>
#include <Eigen/Dense>
#include "verlet.h"
#include "berendsen.h"


TEST(BERENDSEN_THERM_TEST, TEST1)
{
    Atoms atoms(10);
    atoms.velocities.setOnes();

    Velocities_t initial_velocities(atoms.velocities);

    berendsen_thermostat(atoms, 1.0, 256, 1234);

    for (int i = 0; i < atoms.velocities.cols(); i++)
    {
        for (int j = 0; j < atoms.velocities.rows(); j++)
        {
            EXPECT_EQ(atoms.velocities.col(i).row(j).value(), initial_velocities.col(i).row(j).value());
        }
    }
}

TEST(BERENDSEN_THERM_TEST, TEST2)
{
    Atoms atoms(10);
    atoms.velocities.setOnes();

    Velocities_t initial_velocities(atoms.velocities);

    berendsen_thermostat(atoms, 1.0, 256, 1234);

    for (int i = 0; i < atoms.velocities.cols(); i++)
    {
        for (int j = 0; j < atoms.velocities.rows(); j++)
        {
            EXPECT_EQ(atoms.velocities.col(i).row(j).value(), initial_velocities.col(i).row(j).value());
        }
    }
}
