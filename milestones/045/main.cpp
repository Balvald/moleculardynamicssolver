#include "hello.h"
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>
#include <filesystem>
#include <getopt.h>


#ifndef USE_MPI
#define USE_MPI
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "atoms.h"
#include "berendsen.h"
#include "ducastelle.h"
#include "domain.h"
#include "mpi_support.h"

#include "verlet.h"
#include "xyz.h"
#include "initpos.h"

using Positions_t = Eigen::Array3Xd; 
using Velocities_t = Eigen::Array3Xd; 
using Forces_t = Eigen::Array3Xd;
using Names_t = Eigen::Array<std::string, Eigen::Dynamic, 1>;

int main(int argc, char *argv[])
{

#pragma region CMDLINE_ARGS

    Names_t names;
    Positions_t positions;
    Velocities_t velocities;

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
            {"berendsen",no_argument,            0,      'b' },
            {"help",     no_argument,            0,      'h' },
            {0,         0,                       0,      0   }
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
                                                   << " [-q --qd value]\n"      << " [-r --relax value]\n"      << " [--eqsteps value]\n";
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

    if (rank == 0)
    {
        outdata.open("./energy-cluster_923.txt");
    }

    // std::cout << "rank: " << rank << " - fflag: " << fflag << std::endl;

    if (fflag)
    {
        auto data{read_xyz(fvalue)};
        names.resize(std::get<0>(data).size());
        positions.resize(3, std::get<1>(data).cols());
        names = std::get<0>(data);
        positions = std::get<1>(data);

        fvalue = fvalue.substr(0, fvalue.find(".xyz"));


        if (rank == 0)
        {
            outdata.close();
            outdata.open("./energy" + fvalue + ".txt");
        }
        
    }
    else
    {
        auto fallbackdata{read_xyz("cluster_923.xyz")};
        names.resize(std::get<0>(fallbackdata).size());
        positions.resize(3, std::get<1>(fallbackdata).cols());
        names = std::get<0>(fallbackdata);
        positions = std::get<1>(fallbackdata);
    }

    Eigen::Array3Xd forces(3, positions.cols());
    forces.setZero();
    velocities.resize(3, positions.cols());
    velocities.setZero();


    Atoms atoms = Atoms(names, positions, velocities);

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
    
    double epsilon = 1.0;
    double sigma = 1.0;
    double cutoff = 10.0;

    double avg_temp = 0.0;

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

    if (rank == 0)
    {

        outdata << timestep << " " << nb_steps << " " << epsilon << " " << sigma << " " << cutoff << " " << q_delta << " " << relaxation_time << " " << eq_steps << std::endl;

    }


    std::stringstream ss;
    std::string output_path;

    NeighborList neighbor_list;
    neighbor_list.update(atoms, cutoff);
    
    // std::cout << "maxcoeff" << atoms.positions.rowwise().maxCoeff().maxCoeff() << std::endl;

    // Eigen::Array3d d_size{atoms.positions.rowwise().maxCoeff() -
    //                       atoms.positions.rowwise().minCoeff()};

    // double d_rim = cutoff;
    //double min_x = atoms.positions.row(0).minCoeff();
    //double min_y = atoms.positions.row(1).minCoeff();
    //double min_z = atoms.positions.row(2).minCoeff();
    double max_x = atoms.positions.row(0).maxCoeff();
    double max_y = atoms.positions.row(1).maxCoeff();
    double max_z = atoms.positions.row(2).maxCoeff();
    double coeff_x = max_x;
    double coeff_y = max_y;
    double coeff_z = max_z;
    double d_size_x = (coeff_x) + 10;
    double d_size_y = (coeff_y) + 10;
    double d_size_z = (coeff_z) + 10;

    Eigen::Array3d d_size = {d_size_x , d_size_y, d_size_z};


    // std::cout << "d_size: " << d_size << std::endl;

    Eigen::Array3i d_decomposition = {x, y, 1};
    Eigen::Array3i d_decomposition2 = {size, 1, 1};

    assert(x * y * 1 == size);

    // std::cout << "d_decomposition: " << d_decomposition << std::endl;

    // periodicity breaks the program.
    Eigen::Array3i d_periodicity = {1, 1, 1};

    // std::cout << "d_periodicity: " << d_periodicity << std::endl;

    Domain domain(MPI_COMM_WORLD, d_size, d_decomposition, d_periodicity);


    size_t nb_global{atoms.nb_atoms()};
    Atoms total_atoms = Atoms(nb_global);

    size_t nb_local{atoms.nb_atoms()};

    // std::cout << atoms.positions.col(0) << std::endl;
    // std::cout << atoms.positions.col(1) << std::endl;

    domain.enable(atoms);

    //std::cout << atoms.positions.col(0) << std::endl;
    //std::cout << atoms.positions.col(1) << std::endl;

    domain.update_ghosts(atoms, cutoff * 2);

    neighbor_list.update(atoms, cutoff);
    ducastelle(atoms, neighbor_list);

#pragma endregion INITIALIZATION

    /*
    for (size_t i = 0; i < eq_steps; ++i)
    {
        // std::cout << "DoingEqStep" << std::endl;
        verlet_step1(atoms.positions, atoms.velocities, atoms.forces, timestep);

        domain.exchange_atoms(atoms);

        domain.update_ghosts(atoms, cutoff * 2);

        neighbor_list.update(atoms, cutoff);

        // std::cout << "after neighbor_list update" << std::endl;

        nb_local = domain.nb_local();

        double potential_energy{MPI::allreduce(ducastelle_local(atoms, neighbor_list, domain.nb_local(), cutoff), MPI_SUM, MPI_COMM_WORLD)};

        verlet_step2(atoms.velocities, atoms.forces, timestep);

        double kinetic_energy{MPI::allreduce(kin_energy_local(atoms, domain.nb_local()), MPI_SUM, MPI_COMM_WORLD)};

        total_energy = potential_energy + kinetic_energy;

        if (rank == 0)
        {
            // std::cout << "EqStep: " << i << std::endl;
            // std::cout << std::setprecision(9) << "E_pot: " << potential_energy << std::endl;
            // std::cout << std::setprecision(9) << "E_kin: " << kinetic_energy << std::endl;
            // std::cout << std::setprecision(9) << "E_tot: " << total_energy << std::endl;
        }


    }
    

    //if (rank == 0) std::cout << "rank: " << rank <<" - velocity of first atom3: " << atoms.velocities.col(0) << std::endl;

    //if (rank == 0) std::cout << "rank: " << rank <<" - velocity of first atom4: " << atoms.velocities.col(0) << std::endl;
    
    if (eq_steps > 0)
    {
        domain.exchange_atoms(atoms);
        domain.update_ghosts(atoms, cutoff * 2);
        neighbor_list.update(atoms, cutoff);
    }
    */

    for (size_t i = 0; i < eq_steps+nb_steps; ++i)
    {

        // if (i % 1 == 100) {std::cout << "rank: " << rank << " Step: " << i << std::endl;}

        verlet_step1(atoms.positions, atoms.velocities, atoms.forces, timestep);

        domain.exchange_atoms(atoms);

        nb_local = domain.nb_local();

        domain.update_ghosts(atoms, cutoff * 2);

        neighbor_list.update(atoms, cutoff);

        // nb_local = domain.nb_local();

        double potential_local = ducastelle_local(atoms, neighbor_list, domain.nb_local(), cutoff);

        double potential_energy{MPI::allreduce(potential_local, MPI_SUM, MPI_COMM_WORLD)};

        verlet_step2(atoms.velocities, atoms.forces, timestep);

        if (berendsen)
        {
            berendsen_thermostat(atoms, 0.0, timestep, 1.0);
        }

        double kin_energy_l{kin_energy_local(atoms, domain.nb_local())};
        
        double kinetic_energy{MPI::allreduce(kin_energy_l, MPI_SUM, MPI_COMM_WORLD)};

        total_energy = potential_energy + kinetic_energy;


        total_atoms = Atoms(nb_global);
        Atoms local_atoms{atoms};

        local_atoms.conservativeResize(nb_local);

        auto nb_local_thing{nb_local};
        int nb_local_ = (int)nb_local_thing;

        // We first need to figure how many atoms there are in total.
        Eigen::ArrayXi recvcount(size);
        MPI_Allgather(&nb_local_, 1, MPI_INT, recvcount.data(), 1, MPI_INT, MPI_COMM_WORLD);

        Eigen::Index nb_global_atoms{recvcount.sum()};

        // Resize atoms object to fit all (global) atoms.
        local_atoms.conservativeResize(nb_global);

        // Compute where in the global array we need to place the results.
        Eigen::ArrayXi displ(size);
        displ(0) = 0;
        for (int i = 0; i < size - 1; i++)
            displ(i + 1) = displ(i) + recvcount(i);

        // Gather masses, positions, velocities and forces into their respective
        // arrays.
        MPI_Allgatherv(local_atoms.masses.data(), nb_local_, MPI_DOUBLE,
                    total_atoms.masses.data(), recvcount.data(), displ.data(),
                    MPI_DOUBLE, MPI_COMM_WORLD);
        recvcount *= 3;
        displ *= 3;
        MPI_Allgatherv(local_atoms.positions.data(), 3 * nb_local_, MPI_DOUBLE,
                    total_atoms.positions.data(), recvcount.data(), displ.data(),
                    MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_atoms.velocities.data(), 3 * nb_local_, MPI_DOUBLE,
                    total_atoms.velocities.data(), recvcount.data(), displ.data(),
                    MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_atoms.forces.data(), 3 * nb_local_, MPI_DOUBLE,
                    total_atoms.forces.data(), recvcount.data(), displ.data(),
                    MPI_DOUBLE, MPI_COMM_WORLD);

        //std::cout << rank << " th rank has total atoms: " << total_atoms.positions.cols() << std::endl;
        //std::cout << rank << " th rank has local atoms: " << domain.nb_local() << std::endl;

        if (rank == 0)
        {
            //std::cin.get();
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            //std::cout << std::setprecision(9) << "E_kin: " << kinetic_energy << std::endl;
            //std::cout << std::setprecision(9) << "E_tot: " << potential_energy + kinetic_energy << std::endl;
            //std::cin.get();
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        // domain.disable(atoms);

        if (rank == 0)
        {
            if (i % relaxation_time == 0)
            {
                avg_temp = 0.0;
                if (q_delta > 0.0 && i > eq_steps)
                {
                    auto kinetic_abs = std::abs(kinetic_energy);
                    atoms.velocities.leftCols(nb_local) *= sqrt(1.0 + std::abs((q_delta+kinetic_abs)/(kinetic_abs-1.0)));
                }
            }

            if (((int)i % (relaxation_time)) > (relaxation_time/2)  || (int) i == 0)
            {
                avg_temp += 2.0 * kinetic_energy / (3.0 * atoms.positions.cols());
            }

            // outdata << "[" << i << "," << potential_energy << "," << kinetic_energy << "," << potential_energy + kinetic_energy << "," << avg_temp  << "]" << std::endl;


            if (((int) i+1) % (relaxation_time) == 0 || (int) i == 0)
            {


                avg_temp /= (relaxation_time/2);


                outdata << "[" << i << "," << potential_energy << "," << kinetic_energy << "," << potential_energy + kinetic_energy << "," << (2.0 * kinetic_energy / (3.0 * total_atoms.positions.cols())) << "," << avg_temp  << "]" << std::endl;

                ss.str(std::string());
                ss << std::setw(8) << std::setfill('0') << std::to_string(i);
                output_path = "./traj-923-" + ss.str() + ".xyz";
                if (fflag)
                {
                    // preferring several files with a sequence number as they are not too big to be in their own repo
                    output_path = "./traj-" + fvalue + "-" + ss.str() + ".xyz";
                }
                // std::cout << "output_path: " << output_path << std::endl;

                // std::cout << "writing to file" << std::endl;

                // std::cout << "total_atoms.names.size(): " << total_atoms.names.size() << std::endl;
                // std::cout << "total_atoms.positions.cols(): " << total_atoms.positions.cols() << std::endl;
                // std::cout << "total_atoms.velocities.rows(): " << total_atoms.velocities.cols() << std::endl;
                // std::cout << "total_atoms.forces.cols(): " << total_atoms.forces.cols() << std::endl;
                // std::cout << "total_atoms.masses.size(): " << total_atoms.masses.size() << std::endl;

                total_atoms.names.conservativeResize(total_atoms.positions.cols());
                total_atoms.names.setConstant("Au");

                write_xyz_filename(output_path, total_atoms);

                // std::cout << "output_path: " << output_path << std::endl;
            }
            else
            {
                outdata << "[" << i << "," << potential_energy << "," << kinetic_energy << "," << potential_energy + kinetic_energy << "," << (2.0 * kinetic_energy / (3.0 * total_atoms.positions.cols())) << "]" << std::endl;
            }

            //std::cin.get();
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // domain.enable(atoms);

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
