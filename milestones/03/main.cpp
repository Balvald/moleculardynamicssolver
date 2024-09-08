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
        int nb_atoms = 1;

        double gravity = 9.81;

        Eigen::Array3Xd positions(3, nb_atoms);
        positions.setZero();
        std::cout << "Initial position: " << positions.row(0) << " " << positions.row(1) << " " << positions.row(2) << std::endl;
        // positions.row(0) << 0, 1, 2, 3, 4;

        Eigen::Array3Xd velocities(3, nb_atoms);
        velocities.setZero();

        Eigen::Array3Xd forces(3, nb_atoms);
        forces.setZero();

        for (size_t i = 0; i < nb_steps; ++i)
        {
            std::cout << "Step: " << i << std::endl;
            verlet_step1(positions, velocities, forces, timestep);
            forces.row(1) = -gravity;
            verlet_step2(velocities, forces, timestep);

            std::cout << "Position: " << positions.row(0) << " " << positions.row(1) << " " << positions.row(2) << std::endl;
        }

        std::cout << "Final position: " << positions.row(0) << " " << positions.row(1) << " " << positions.row(2) << std::endl;
    }



#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
