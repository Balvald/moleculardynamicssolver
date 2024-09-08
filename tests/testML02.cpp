#include <gtest/gtest.h>
#include <math.h>
#include "verlet.h"

double pi = M_PI;

TEST(SinTest, IntegerMultiplesOfPi)
{
  EXPECT_NEAR(sin(0), 0, 1e-6);
  EXPECT_NEAR(sin(pi), 0, 1e-6);
  EXPECT_NEAR(sin(2*pi), 0, 1e-6);
}

TEST(VelovityVerletTest, SingleStep)
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  double vx = 0.0;
  double vy = 0.0;
  double vz = 0.0;

  double fx = 0.0;
  double fy = 0.0;
  double fz = 0.0;

  double timestep = 1;
  double gravity = 9.81;

  x = 0.0;
  y = 0.0;
  z = 0.0;

  vx = 0.0;
  vy = 0.0;
  vz = 0.0;

  fx = 0.0;
  fy = 0.0;
  fz = 0.0;

  verlet_step1(x, y, z, vx, vy, vz, fx, fy, fz, timestep);
  fy = -gravity;
  verlet_step2(vx, vy, vz, fx, fy, fz, timestep);

  verlet_step1(x, y, z, vx, vy, vz, fx, fy, fz, timestep);
  fy = -gravity;
  verlet_step2(vx, vy, vz, fx, fy, fz, timestep);

  EXPECT_NEAR(x, 0, 1e-6);
  EXPECT_NEAR(y, -9.81, 1e-6);
  EXPECT_NEAR(z, 0, 1e-6);

  EXPECT_NEAR(vx, 0, 1e-6);
  EXPECT_NEAR(vy, -9.81*1.5, 1e-6);
  EXPECT_NEAR(vz, 0, 1e-6);
}

