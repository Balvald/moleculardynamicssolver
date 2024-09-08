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
#include "mpi_support.h"
#include "domain.h"

#include "verlet.h"
#include "xyz.h"
#include "initpos.h"

using Positions_t = Eigen::Array3Xd; 
using Velocities_t = Eigen::Array3Xd; 
using Forces_t = Eigen::Array3Xd;
using Names_t = Eigen::Array<std::string, Eigen::Dynamic, 1>;

#ifndef USE_MPI
#define USE_MPI
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif


int main(int argc, char *argv[])
{

#pragma region CMDLINE_ARGS

    bool fflag = false;
    bool tflag = false;
    bool mflag = false;
    bool eflag = false;
    bool sflag = false;
    bool cflag = false;
    bool qflag = false;
    bool rflag = false;
    bool eqflag = false;
    bool berendsen = false;
    bool tauflag = false;
    bool ttflag = false;
    bool srflag = false;
    int c;
    double tvalue;
    size_t mvalue;
    double evalue;
    double svalue;
    double cvalue;
    double qvalue;
    double rvalue;
    size_t eqvalue;
    double btvalue;
    double ttvalue;
    double srvalue;

    std::string fvalue;

    while (true)
    {

        int option_index = 0;
        static struct option long_options[] = {
            {"file",        required_argument,      0,      'f' },
            {"timestep",    required_argument,      0,      't' },
            {"maxstep",     required_argument,      0,      'm' },
            {"epsilon",     required_argument,      0,      'e' },
            {"sigma",       required_argument,      0,      's' },
            {"cutoff",      required_argument,      0,      'c' },
            {"qd",          required_argument,      0,      'q' },
            {"relax",       required_argument,      0,      'r' },
            {"eqsteps",     required_argument,      0,      'eq'},
            {"tau",         required_argument,      0,      'bt'},
            {"targettemp",  required_argument,      0,      'tt'},
            {"berendsen",   no_argument,            0,      'b' },
            {"strainrate",  required_argument,      0,      'sr'},
            {"help",        no_argument,            0,      'h' },
            {0,             0,                      0,       0  }
        };

        c = getopt_long(argc, argv, "hf:t:m:e:s:c:q:r:b", long_options, &option_index);

        if (c == -1)
        {
            break;
        }

        switch (c)
        {
            case 'h':
                std::cout << "Usage: " << argv[0]  << " [-f --file filename]\n" << " [-t --timestep value]\n"   << " [-m --maxstep value]\n" 
                                                   << " [-e --epsilon value]\n" << " [-s --sigma value]\n"      << " [-c --cutoff value]\n"
                                                   << " [-q --qd value]\n"      << " [-r --relax value]\n"      << " [--eqsteps value]\n" << " [-b --berendsen]\n";
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
            case 'b':
                std::cout << "option -b\n";
                berendsen = true;
                break;
            case 'tt':
                std::cout << "option -tt with value `" << optarg << "'\n";
                ttflag = true;
                ttvalue = std::stod(optarg);
                break;
            case 'bt':
                std::cout << "option -bt with value `" << optarg << "'\n";
                tauflag = true;
                btvalue = std::stod(optarg);
                break;
            case 'sr':
                std::cout << "option -sr with value `" << optarg << "'\n";
                srflag = true;
                srvalue = std::stod(optarg);
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

#pragma endregion CMDLINE_ARGS

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

    // get prime factorization of size
    // define array of prime factors

    
#pragma region DOMAIN_DECOMPOSITION


    Eigen::ArrayXi prime_factors;

    int sizecopy = size;

    int prime_factor_index = 0;

    for (int i = 2; i <= sizecopy; ++i)
    {
        while (sizecopy % i == 0)
        {
            prime_factors.conservativeResize(prime_factor_index + 1);
            prime_factors[prime_factor_index] = i;
            sizecopy /= i;
            prime_factor_index++;
        }
    }

    int x = 1;
    int y = 1;

    for (int i = 0; i < prime_factors.size(); ++i)
    {
        if (x <= y)
        {
            x *= prime_factors[i];
        }
        else
        {
            y *= prime_factors[i];
        }
    }
    


    
#pragma endregion DOMAIN_DECOMPOSITION


#pragma region INITIALIZATION

    std::ofstream outdata;

    // auto [names, positions, velocities]{read_xyz_with_velocities("./lj54.xyz")};

    // Atoms atoms = Atoms(names, positions, velocities);

    auto [names, positions]{read_xyz("whisker_small.xyz")};

    if (rank == 0)
    {
        outdata.open("./energy-cluster.txt");
        
        if (fflag)
        {
            auto [names, positions]{read_xyz(fvalue)};

            fvalue = fvalue.substr(0, fvalue.find(".xyz"));

            outdata.close();
            outdata.open("./energy" + fvalue + ".txt");
        }
    }

    Eigen::Array3Xd forces(3, positions.cols());
    forces.setZero();
    Eigen::Array3Xd velocities(3, positions.cols());
    velocities.setZero();


    Atoms atoms = Atoms(positions, velocities);

    // Atoms atoms = init_cube(100, 1.0);

    // write_xyz_filename("./lj100cube.xyz", atoms);

    // atoms = init_cube_fcc(100, 1.0);

    // write_xyz_filename("./lj100cubefcc.xyz", atoms);


    // Initialize standard values.
    double timestep = 0.01;
    size_t nb_steps = 10000; // (int)(10.0*(1.0/timestep));        
    size_t eq_steps = 1000;

    //double potential_energy = 0.0;
    //double kinetic_energy = 0.0;
    double total_energy = 0.0;

    double q_delta = 1.0;
    int relaxation_time = 10;
    
    double tau = 1.0;
    double target_temp = 1.0;

    double epsilon = 1.0;
    double sigma = 1.0;
    double cutoff = 10.0;

    double avg_temp = 0.0;

    double strain_rate = 0.0;

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
    if (tauflag)
    {
        tau = btvalue;
    }
    if (ttflag)
    {
        target_temp = ttvalue;
    }
    if (srflag)
    {
        strain_rate = srvalue;
    }

    outdata << timestep << " " << nb_steps << " " << epsilon << " " << sigma << " " << cutoff << " " << q_delta << " " << relaxation_time << " " << eq_steps << " " << strain_rate << std::endl;


    std::stringstream ss;
    std::string output_path;

    NeighborList neighbor_list;
    neighbor_list.update(atoms, cutoff);
    
    std::cout << "maxcoeff" << atoms.positions.rowwise().maxCoeff().maxCoeff() << std::endl;

    double d_rim = 2.0 * cutoff;
    double min_x = atoms.positions.row(0).minCoeff();
    double min_y = atoms.positions.row(1).minCoeff();
    double min_z = atoms.positions.row(2).minCoeff();
    double max_x = atoms.positions.row(0).maxCoeff();
    double max_y = atoms.positions.row(1).maxCoeff();
    double max_z = atoms.positions.row(2).maxCoeff();
    double coeff_x = max_x - min_x;
    double coeff_y = max_y - min_y;
    double coeff_z = max_z - min_z;
    double d_size_x = (coeff_x + 3 * d_rim);
    double d_size_y = (coeff_y + 3 * d_rim);
    double d_size_z = (coeff_z + 3 * d_rim);

    Eigen::Array3d d_size = {d_size_x , d_size_y, d_size_z};

    //Eigen::Array3d d_size = {atoms.positions.rowwise().maxCoeff() -
    //              atoms.positions.rowwise().minCoeff()};

    std::cout << "d_size: " << d_size << std::endl;

    Eigen::Array3i d_decomposition = {x, y, 1};

    // periodicity breaks the program.
    Eigen::Array3i d_periodicity = {0, 0, 0};

    Domain domain = Domain(MPI_COMM_WORLD, d_size, d_decomposition, d_periodicity);


    std::string filename_new = "cluster_923_new.xyz";

    
    // write_xyz_filename(filename_new, atoms);

    // domain.enable(atoms);
    // domain.disable(atoms);

    std::string filename_new_sec = "cluster_923_new_sec.xyz";

    
    // write_xyz_filename(filename_new_sec, atoms);
    if (rank == 0) std::cout << "rank: " << rank <<" - pos of first atom-1: " << atoms.positions.col(0) << std::endl;
    if (rank == 0) std::cout << "rank: " << rank <<" - velocity of first atom-1: " << atoms.velocities.col(0) << std::endl;


    // NeighborList neighbor_list = NeighborList();
    // neighbor_list.update(atoms, cutoff);

    domain.enable(atoms);
    
    if (rank == 0) std::cout << "rank: " << rank <<" - pos of first atom0: " << atoms.positions.col(0) << std::endl;
    if (rank == 0) std::cout << "rank: " << rank <<" - velocity of first atom0: " << atoms.velocities.col(0) << std::endl;

    domain.update_ghosts(atoms, cutoff * 2);
    
    if (rank == 0) std::cout << "rank: " << rank <<" - pos of first atom1: " << atoms.positions.col(0) << std::endl;
    if (rank == 0) std::cout << "rank: " << rank <<" - velocity of first atom1: " << atoms.velocities.col(0) << std::endl;

    neighbor_list.update(atoms, cutoff);
    // ducastelle(atoms, neighbor_list);
    std::cout << "after neighbor_list update" << std::endl;


    // print velocity of first atom
    if (rank == 0) std::cout << "rank: " << rank <<" - pos of first atom2: " << atoms.positions.col(0) << std::endl;
    if (rank == 0) std::cout << "rank: " << rank <<" - velocity of first atom2: " << atoms.velocities.col(0) << std::endl;

#pragma endregion INITIALIZATION

    
    for (size_t i = 0; i < eq_steps; ++i)
    {
        
        // std::cout << "DoingEqStep" << std::endl;
        verlet_step1(atoms.positions, atoms.velocities, atoms.forces, timestep);

        domain.exchange_atoms(atoms);
        domain.update_ghosts(atoms, cutoff * 2);

        neighbor_list.update(atoms, cutoff);

        // std::cout << "after neighbor_list update" << std::endl;

        double potential_energy{MPI::allreduce(ducastelle_local(atoms, neighbor_list, domain.nb_local(), cutoff), MPI_SUM, MPI_COMM_WORLD)};

        verlet_step2(atoms.velocities, atoms.forces, timestep);

        double kinetic_energy{MPI::allreduce(kin_energy_local(atoms, domain.nb_local()), MPI_SUM, MPI_COMM_WORLD)};

        total_energy = potential_energy + kinetic_energy;

        domain.disable(atoms);

        if (rank == 0)
        {
            std::cout << "EqStep: " << i << std::endl;
            std::cout << std::setprecision(9) << "E_pot: " << potential_energy << std::endl;
            std::cout << std::setprecision(9) << "E_kin: " << kinetic_energy << std::endl;
            std::cout << std::setprecision(9) << "E_tot: " << total_energy << std::endl;
        }

        domain.enable(atoms);


    }
    
    if (eq_steps > 0)
    {
        domain.update_ghosts(atoms, cutoff * 2);
        neighbor_list.update(atoms, cutoff);
    }
    

    for (size_t i = 0; i < nb_steps; ++i)
    {

        if (i % 1 == 100) {std::cout << "rank: " << rank << " Step: " << i << std::endl;}

        // using lj.h instead of lj_direct_summation.h means we use a cutoff of 0.5
        verlet_step1(atoms.positions, atoms.velocities, atoms.forces, timestep);

        domain.exchange_atoms(atoms);

        domain.update_ghosts(atoms, cutoff * 2);

        neighbor_list.update(atoms, cutoff);

        double potential_local = ducastelle_local(atoms, neighbor_list, domain.nb_local(), cutoff);

        double potential_energy{MPI::allreduce(potential_local, MPI_SUM, MPI_COMM_WORLD)};

        verlet_step2(atoms.velocities, atoms.forces, timestep);

        if (berendsen)
        {
            berendsen_thermostat_local(atoms, target_temp, timestep, tau, domain.nb_local());
        }


        double local_stress{1 / (2 * d_size.prod()) * atoms.stress.leftCols(domain.nb_local()).colwise().sum().sum()};
        double stress{MPI::allreduce(local_stress, MPI_SUM, MPI_COMM_WORLD)};

        if (i % 100 == 0) {std::cout << std::setprecision(9) << "E_pot: " << potential_energy << std::endl;}

        double kin_energy_l{kin_energy_local(atoms, domain.nb_local())};
        
        //if (rank == 0) std::cout << "rank: " << rank <<" - kinetic_energy: " << kin_energy_l << std::endl;
        //if (rank == 1) std::cout << "rank: " << rank <<" - kinetic_energy: " << kin_energy_l << std::endl;

        double kinetic_energy{MPI::allreduce(kin_energy_l, MPI_SUM, MPI_COMM_WORLD)};

        //if (rank == 0) std::cout << "rank: " << rank <<" - kinetic_energy: " << kinetic_energy << std::endl;

    
        //if (rank == 0) std::cout << "rank: " << rank <<" - velocity of first atom11: " << atoms.velocities.col(0) << std::endl;

        total_energy = potential_energy + kinetic_energy;

        // std::cout << "rank: " << rank << "[" << potential_energy << " , " << kinetic_energy << " , " << total_energy << "]," << std::endl;

        //if (i % relaxation_time == 0 && i != 0)
        //{
        //    atoms.velocities *= sqrt(1.0 + (q_delta/kinetic_energy));
        //}

        d_size.row(2) += strain_rate * timestep;
        domain.scale(atoms, d_size);
        domain.exchange_atoms(atoms);
        domain.update_ghosts(atoms, cutoff * 2);
        neighbor_list.update(atoms, cutoff);
        
        potential_local = ducastelle_local(atoms, neighbor_list, domain.nb_local(), cutoff);

        domain.disable(atoms);

        if (rank == 0)
        {
            if (i % relaxation_time == 0)
            {
                // Deposit energy into the system
                avg_temp = 0.0;
                // auto kinetic_abs = std::abs(kinetic_energy);
                // atoms.velocities *= sqrt(1.0 + std::abs((q_delta+kinetic_abs)/(kinetic_abs-1.0)));
            }

            if ((i % (relaxation_time)) > (relaxation_time/2)  || i == 0)
            {
                avg_temp += 2.0 * kinetic_energy / (3.0 * atoms.positions.cols());
            }

            if ((i+1) % (relaxation_time) == 0 || i == 0)
            {
                std::cout << std::setprecision(9) << "E_kin: " << kinetic_energy << std::endl;
                std::cout << std::setprecision(9) << "E_tot: " << potential_energy + kinetic_energy << std::endl;

                avg_temp /= (relaxation_time/2);


                outdata << "[" << i << "," << potential_energy << "," << kinetic_energy << "," << potential_energy + kinetic_energy << "," << avg_temp  << "," << stress <<"]" << std::endl;


                ss.str(std::string());
                ss << std::setw(8) << std::setfill('0') << std::to_string(i);
                output_path = "./traj-923-" + ss.str() + ".xyz";
                if (fflag)
                {
                    // preferring several files with a sequence number as they are not too big to be in their own repo
                    output_path = "./traj-" + fvalue + "-" + ss.str() + ".xyz";
                }
                // std::cout << "output_path: " << output_path << std::endl;

                write_xyz_filename(output_path, atoms);

            }
        }

        domain.enable(atoms);

    }


    //std::cout << std::setprecision(9) << "Final E_pot: " << potential_energy << std::endl;
    //std::cout << std::setprecision(9) << "Final E_kin: " << kinetic_energy << std::endl;
    std::cout << std::setprecision(9) << "Final E_tot: " << total_energy << std::endl;

    if (rank == 0)
    {
        outdata.close();
    }

    domain.disable(atoms);

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
