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
#include "berendsen.h"
#include "ducastelle.h"

#include "verlet.h"
#include "xyz.h"
#include "initpos.h"

using Positions_t = Eigen::Array3Xd; 
using Velocities_t = Eigen::Array3Xd; 
using Forces_t = Eigen::Array3Xd;
using Names_t = Eigen::Array<std::string, Eigen::Dynamic, 1>;


#ifdef USE_MPI
#include <mpi.h>
#endif


int main(int argc, char *argv[])
{
    Positions_t positions;
    Velocities_t velocities;
    Names_t names;

    bool fflag = false;
    bool tflag = false;
    bool mflag = false;
    bool eflag = false;
    bool sflag = false;
    bool cflag = false;
    bool qflag = false;
    bool rflag = false;
    bool eqflag = false;
    int c;
    double tvalue;
    size_t mvalue;
    double evalue;
    double svalue;
    double cvalue;
    double qvalue;
    double rvalue;
    size_t eqvalue;

    std::string fvalue;

    while (true)
    {
        int this_option_optind = optind ? optind : 1;
        int option_index = 0;
        static struct option long_options[] = {
            {"file",     required_argument,      0,      'f' },
            {"timestep", required_argument,      0,      't' },
            {"maxstep",  required_argument,      0,      'm' },
            {"epsilon",  required_argument,      0,      'e' },
            {"sigma",    required_argument,      0,      's' },
            {"cutoff",   required_argument,      0,      'c' },
            {"qd",       required_argument,      0,      'q' },
            {"relax",    required_argument,      0,      'r' },
            {"eqsteps",  required_argument,      0,      'eq'},
            {"help",     no_argument,            0,      'h' },
            {0,         0,                       0,      0   }
        };

        c = getopt_long(argc, argv, "hf:t:m:e:s:c:q:r:", long_options, &option_index);

        if (c == -1)
        {
            break;
        }

        switch (c)
        {
            case 'h':
                std::cout << "Usage: " << argv[0] << " [-f value]\n";
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
            case 'c':
                cflag = true;
                std::cout << "option -c with value `" << optarg << "'\n";
                cvalue = std::stod(optarg);
                break;
            case 'q':
                qflag = true;
                std::cout << "option -q with value `" << optarg << "'\n";
                qvalue = std::stod(optarg);
                break;
            case 'r':
                rflag = true;
                std::cout << "option -r with value `" << optarg << "'\n";
                rvalue = std::stod(optarg);
                break;
            case 'eq':
                eqflag = true;
                std::cout << "option --eqsteps with value `" << optarg << "'\n";
                eqvalue = std::stoul(optarg);
                break;
            case '?':
                std::cerr << "Usage: " << argv[0] << " [-f value]\n";
                return 1;
            default:
                printf("?? getopt returned character code 0%o ??\n", c);
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

    if (rank ==  0)
    {
        // auto [names, positions, velocities]{read_xyz_with_velocities("./lj54.xyz")};

        // Atoms atoms = Atoms(names, positions, velocities);

        outdata.open("./energy-cluster_923.txt");

        if (fflag)
        {
            auto readdata{read_xyz_with_velocities(fvalue)};
            names = std::get<0>(readdata);
            positions = std::get<1>(readdata);
            velocities = std::get<2>(readdata);
            fvalue = fvalue.substr(0, fvalue.find(".xyz"));

            outdata.close();
            outdata.open("./energy" + fvalue + ".txt");
        }
        else
        {
            auto fallbackdata{read_xyz_with_velocities("./cluster_923.xyz")};
            names = std::get<0>(fallbackdata);
            positions = std::get<1>(fallbackdata);
            velocities = std::get<2>(fallbackdata);
        }
        

        Eigen::Array3Xd forces(3, positions.cols());
        forces.setZero();
        velocities.setZero();


        Atoms atoms = Atoms(names, positions, velocities);

        // Atoms atoms = init_cube(100, 1.0);

        // write_xyz_filename("./lj100cube.xyz", atoms);

        // atoms = init_cube_fcc(100, 1.0);

        // write_xyz_filename("./lj100cubefcc.xyz", atoms);


        NeighborList neighbors = NeighborList();


        double timestep = 0.01;
        size_t nb_steps = 10000; // (int)(10.0*(1.0/timestep));
        size_t eq_steps = 1000;

        double potential_energy = 0.0;
        double kinetic_energy = 0.0;
        double total_energy = 0.0;

        double q_delta = 1.0;
        int relaxation_time = 10;
        
        double epsilon = 1.0;
        double sigma = 1.0;
        double cutoff = 10.0;

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
        if (cflag)
        {
            cutoff = cvalue;
        }
        if (qflag)
        {
            q_delta = qvalue;
        }
        if (rflag)
        {
            relaxation_time = rvalue;
        }
        if (eqflag)
        {
            eq_steps = eqvalue;
        }

        std::stringstream ss;
        std::string output_path;

        neighbors.update(atoms, cutoff);

        for (size_t i = 0; i < eq_steps; ++i)
        {
            if (i % 100 == 0) {std::cout << "EqStep: " << i << std::endl;}
            verlet_step1(atoms, timestep);
            neighbors.update(atoms, cutoff);
            ducastelle(atoms, neighbors, cutoff);
            verlet_step2(atoms, timestep);
            std::cout << std::setprecision(9) << "E_kin: " << kinetic_energy << std::endl;
            std::cout << std::setprecision(9) << "E_tot: " << potential_energy + kinetic_energy << std::endl;

        }

        double avg_temp = 2.0 * kinetic_energy / (3.0 * atoms.positions.cols());

        for (size_t i = 0; i < nb_steps; ++i)
        {
            if (i % 100 == 0) {std::cout << "Step: " << i << std::endl;}
            

            // using lj.h instead of lj_direct_summation.h means we use a cutoff of 0.5
            verlet_step1(atoms, timestep);
            // forces.row(1) = -gravity;
            // atoms.forces.row(1) = -gravity;
            neighbors.update(atoms, cutoff);


            potential_energy = ducastelle(atoms, neighbors, cutoff);
            
            // lj_direct_summation(atoms, neighbors, 1.0, 1.0, cutoff);

            verlet_step2(atoms, timestep);

            // rescale velocities with berendsen thermostat

            // berendsen_thermostat(atoms, 0.0, timestep, 1.0);

            // Deposit energy into the system


            if (i % 100 == 0) {std::cout << std::setprecision(9) << "E_pot: " << potential_energy << std::endl;}

            kinetic_energy = kin_energy(atoms);
            total_energy = potential_energy + kinetic_energy;

            // outdata << "[" << potential_energy << " , " << kinetic_energy
            //        << " , " << total_energy << " , " << avg_temp << "]," << std::endl;

            double temp = 2.0 * kinetic_energy / (3.0 * atoms.positions.cols());

            if (i % relaxation_time == 0)
            {
                avg_temp = 0.0;
                atoms.velocities *= sqrt(1.0 + (q_delta/kinetic_energy));
            }

            if ((i % (relaxation_time)) > (relaxation_time/2)  || i == 0)
            {
                avg_temp += temp;
            }

            if ((i+1) % (relaxation_time) == 0 || i == 0)
            {
                std::cout << std::setprecision(9) << "E_kin: " << kinetic_energy << std::endl;
                std::cout << std::setprecision(9) << "E_tot: " << potential_energy + kinetic_energy << std::endl;

                avg_temp /= (relaxation_time/2);


                outdata << "[" << i << "," << potential_energy << "," << kinetic_energy << "," << potential_energy + kinetic_energy << "," << temp  << "," << avg_temp  << "]" << std::endl;


                ss.str(std::string());
                ss << std::setw(8) << std::setfill('0') << std::to_string(i);
                output_path = "./traj-923-" + ss.str() + ".xyz";
                if (fflag)
                {
                    // preferring several files with a sequence number as they are not too big to be in their own repo
                    output_path = "./traj-" + fvalue + "-" + ss.str() + ".xyz";
                }
                // std::cout << "output_path: " << output_path << std::endl;
                std::ofstream file;
                file.open(output_path);
                write_xyz(file, atoms);
                file.close();
            }
            else
            {
                outdata << "[" << i << "," << potential_energy << "," << kinetic_energy << "," << potential_energy + kinetic_energy << "," << temp  << "]," << std::endl;
            }
            
        }


        std::cout << std::setprecision(9) << "Final E_pot: " << potential_energy << std::endl;
        std::cout << std::setprecision(9) << "Final E_kin: " << kinetic_energy << std::endl;
        std::cout << std::setprecision(9) << "Final E_tot: " << potential_energy + kinetic_energy << std::endl;

        
        outdata.close();
    }


#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
