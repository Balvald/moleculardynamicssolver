#include <gtest/gtest.h>
#include <math.h>
#include <Eigen/Dense>
#include "verlet.h"


TEST(VelovityVerletTest, MultipleAtoms)
{
  double timestep = 1;
  int nb_atoms = 5;

  double gravity = 9.81;

  Eigen::Array3Xd positions(3, nb_atoms);
  positions.setZero();
  std::cout << "Initial position: " << positions.row(0) << " " << positions.row(1) << " " << positions.row(2) << std::endl;
  positions.row(0) << 0, 1, 2, 3, 4;

  Eigen::Array3Xd velocities(3, nb_atoms);
  velocities.setZero();

  Eigen::Array3Xd forces(3, nb_atoms);
  forces.setZero();

  // std::cout << "Step: " << i << std::endl;
  verlet_step1(positions, velocities, forces, timestep);
  forces.row(1) = -gravity;
  verlet_step2(velocities, forces, timestep);

  verlet_step1(positions, velocities, forces, timestep);
  forces.row(1) = -gravity;
  verlet_step2(velocities, forces, timestep);


  std::cout << "Position: " << positions.row(0) << " " << positions.row(1) << " " << positions.row(2) << std::endl;

  for (int j = 0; j < nb_atoms; ++j)
  {
    for (int i = 0; i < 3; ++i)
    {
      std::cout << "i: " << i << " j: " << j << " positions: " << positions(i, j) << " velocities: " << velocities(i, j) << std::endl;
      if (i == 0)
      {
        EXPECT_NEAR(positions(i, j), j, 1e-6);
        EXPECT_NEAR(velocities(i, j), 0, 1e-6);
      }
      else if (i == 1)
      {
        EXPECT_NEAR(positions(i, j), -9.81, 1e-6);
        EXPECT_NEAR(velocities(i, j), -9.81*1.5, 1e-6);
      }
      else
      {
        EXPECT_NEAR(positions(i, j), 0, 1e-6);
        EXPECT_NEAR(velocities(i, j), 0, 1e-6);
      }
    }
  }

}
