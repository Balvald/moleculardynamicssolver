#include "hello.h"
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>
#include <filesystem>
#include <getopt.h>

#include "atoms.h"
#include "lj_direct_summation.h"

#include "verlet.h"
#include "xyz.h"


using Positions_t = Eigen::Array3Xd; 
using Velocities_t = Eigen::Array3Xd; 
using Forces_t = Eigen::Array3Xd;
using Names_t = Eigen::Array<std::string, Eigen::Dynamic, 1>;


#ifdef USE_MPI
#include <mpi.h>
#endif


int main(int argc, char *argv[])
{
    bool fflag = false;
    bool tflag = false;
    bool mflag = false;
    bool eflag = false;
    bool sflag = false;
    int c;
    double tvalue;
    size_t mvalue;
    double evalue;
    double svalue;

    std::string fvalue;

    while (true)
    {

        int option_index = 0;
        static struct option long_options[] = {
            {"file",     required_argument,      0,      'f' },
            {"timestep", required_argument,      0,      't' },
            {"maxstep",  required_argument,      0,      'm' },
            {"epsilon",  required_argument,      0,      'e' },
            {"sigma",    required_argument,      0,      's' },
            {"help",     no_argument,            0,      'h' },
            {0,         0,                       0,      0   }
        };

        c = getopt_long(argc, argv, "hf:t:m:e:s:", long_options, &option_index);

        if (c == -1)
        {
            break;
        }

        switch (c)
        {
            case 'h':
                std::cout << "Usage: " << argv[0]  << " [-f --file filename]\n" << " [-t --timestep value]\n"   << " [-m --maxstep value]\n" 
                                                   << " [-e --epsilon value]\n" << " [-s --sigma value]\n";
                return 0;
            case 'f':
                fflag = true;
                std::cout << "option -f with value `" << optarg << "'\n";
                fvalue = static_cast<std::string>(optarg);
                break;
            case 't':
                tflag = true;
                std::cout << "option -t with value `" << optarg << "'\n";
                tvalue = std::stod(optarg);
                break;
            case 'm':
                mflag = true;
                std::cout << "option -m with value `" << optarg << "'\n";
                mvalue = std::stoul(optarg);
                break;
            case 'e':
                eflag = true;
                std::cout << "option -e with value `" << optarg << "'\n";
                evalue = std::stod(optarg);
                break;
            case 's':
                sflag = true;
                std::cout << "option -s with value `" << optarg << "'\n";
                svalue = std::stod(optarg);
                break;
            case '?':
                std::cerr << "Try: " << argv[0] << " [-h --help]\n";
                printf("I have no memory of this cmd line argument\n");
                std::cout <<
                "           ⢀⣀⣀⠄       \n"
                "    ⢰⡄   ⣠⣾⣿⠋         \n"
                "    ⢸⡇⠐⠾⣿⣿⣿⣦          \n"
                "     ⡇ ⢠⣿⣿⣿⣿⣧        \n"
                "    ⠐⣿⣾⣿⣿⣿⣿⣿⣿⣧       \n"
                "     ⢹⣿⣿⣿⣿⣿⣿⣿⣏      \n"
                "     ⢸⠈⠉⣿⣿⣿⣿⣿⣿⡄     \n"
                "     ⠸⡆ ⣿⣿⣿⣿⣿⣿⣿     \n"
                "      ⠇⢀⣿⣿⣿⣿⣿⣿⣿⣧⡀  \n"
                "       ⢸⣿⣿⣿⣿⣿⣿⣿⣿⣷⡄" << std::endl;
                return 1;
            default:
                printf("?? getopt returned character code 0%o ??\n", c);
                std::cerr << "Try: " << argv[0] << " [-h --help]\n";
                return 1;
        }
    }

    int rank = 0, size = 1;

    // Below is some MPI code, try compiling with `cmake -DUSE_MPI=ON ..`
#ifdef USE_MPI
    MPI_Init(&argc, &argv);

    // Retrieve process infos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    std::cout << "Hello I am rank " << rank << " of " << size << "\n";

    /*
    if (rank == 0)
    {
      hello_eigen();
    }
    */

    auto input_path = "./simulation_test_input.txt";

    if (not std::filesystem::exists(input_path))
      std::cerr << "warning: could not find input file " << input_path << "\n";

    // std::cin.get();

    std::ofstream outdata;

    if (rank == 0 && !fflag)
    {
        outdata.open("./energy-cluster_923.txt");
    }

    // std::cout << "rank: " << rank << " - fflag: " << fflag << std::endl;

    if (rank ==  0)
    {
        auto [names, positions, velocities]{read_xyz_with_velocities("./lj54.xyz")};

        if (fflag)
        {
            auto [names, positions, velocities]{read_xyz_with_velocities(fvalue)};
            fvalue = fvalue.substr(0, fvalue.find(".xyz"));
            // outdata.close();
            outdata.open("./energy" + fvalue + ".txt");
        }
        

        Atoms atoms = Atoms(names, positions, velocities);

        double timestep = 0.1;
        size_t nb_steps = 1000; // (int)(10.0*(1.0/timestep));

        double potential_energy = 0.0;
        double kinetic_energy = 0.0;

        double epsilon = 1.0;
        double sigma = 1.0;

        if (tflag)
        {
            timestep = tvalue;
        }
        if (mflag)
        {
            nb_steps = mvalue;
        }
        if (eflag)
        {
            epsilon = evalue;
        }
        if (sflag)
        {
            sigma = svalue;
        }

        outdata << timestep << " " << nb_steps << " " << epsilon << " " << sigma << " " << std::endl;




        std::stringstream ss;
        std::string output_path;

        for (size_t i = 0; i < nb_steps; ++i)
        {
            std::cout << "Step: " << i << std::endl;
            verlet_step1(atoms, timestep);
            // forces.row(1) = -gravity;
            // atoms.forces.row(1) = -gravity;
            potential_energy = lj_direct_summation(atoms, epsilon, sigma);

            verlet_step2(atoms, timestep);

            //std::cout << "E_pot: " << potential_energy << std::endl;
            kinetic_energy = atoms.get_avg_mass() * atoms.velocities.square().sum() * 0.5;


            //std::cout << "E_kin: " << kinetic_energy << std::endl;
            //std::cout << "E_tot: " << potential_energy + kinetic_energy << std::endl;

            outdata << "[" << i << "," << potential_energy << "," << kinetic_energy << "," << potential_energy + kinetic_energy << "]" << std::endl;

            if (i % 100 == 0)
            {
                //std::cout << std::setprecision(9) << "E_kin: " << kinetic_energy << std::endl;
                //std::cout << std::setprecision(9) << "E_tot: " << potential_energy + kinetic_energy << std::endl;

                if (fflag)
                {
                    ss.str(std::string());
                    ss << std::setw(8) << std::setfill('0') << std::to_string(i);
                    output_path = "./traj" + fvalue + "-" + ss.str() + ".xyz";
                    // std::cout << "output_path: " << output_path << std::endl;
                    std::ofstream file;
                    file.open(output_path);
                    write_xyz(file, atoms);
                    file.close();
                }
                else
                {
                    ss.str(std::string());
                    ss << std::setw(8) << std::setfill('0') << std::to_string(i);
                    output_path = "./traj" + ss.str() + ".xyz";
                    // std::cout << "output_path: " << output_path << std::endl;
                    std::ofstream file;
                    file.open(output_path);
                    write_xyz(file, atoms);
                    file.close();
                }
            }
            // std::cin.get();
        }

        outdata.close();

        std::cout << "Final E_pot: " << potential_energy << std::endl;
        std::cout << "Final E_kin: " << kinetic_energy << std::endl;
        std::cout << "Final E_tot: " << potential_energy + kinetic_energy << std::endl;
    }



#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
