#include "hello.h"
#include <Eigen/Dense>
#include <iostream>
#include <filesystem>
#include "verlet.h"

#ifdef USE_MPI
#include <mpi.h>
#endif


int main(int argc, char *argv[])
{
    int rank = 0, size = 1;

    // Below is some MPI code, try compiling with `cmake -DUSE_MPI=ON ..`
#ifdef USE_MPI
    MPI_Init(&argc, &argv);

    // Retrieve process infos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    std::cout << "Hello I am rank " << rank << " of " << size << "\n";


    if (rank == 0)
    {
      hello_eigen();
    }


    auto input_path = "./simulation_test_input.txt";

    if (not std::filesystem::exists(input_path))
      std::cerr << "warning: could not find input file " << input_path << "\n";


    if (rank ==  0)
    {
        double timestep = 1;
        size_t nb_steps = 1000;

        double gravity = 9.81;

        double x = 0.0;
        double y = 0.0;
        double z = 0.0;

        double vx = 0.0;
        double vy = 0.0;
        double vz = 0.0;

        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;


        for (size_t i = 0; i < nb_steps; ++i)
        {
            std::cout << "Step: " << i << std::endl;
            verlet_step1(x, y, z, vx, vy, vz, fx, fy, fz, timestep);
            fy = -gravity;
            verlet_step2(vx, vy, vz, fx, fy, fz, timestep);

            std::cout << "Position: " << x << " " << y << " " << z << std::endl;
        }

        std::cout << "Final position: " << x << " " << y << " " << z << std::endl;
    }



#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
