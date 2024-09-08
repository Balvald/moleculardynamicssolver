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
// #include "domain.h"
#include "mpi_support.h"

#include "verlet.h"
#include "xyz.h"
#include "initpos.h"

using Positions_t = Eigen::Array3Xd; 
using Velocities_t = Eigen::Array3Xd; 
using Forces_t = Eigen::Array3Xd;
using Names_t = Eigen::Array<std::string, Eigen::Dynamic, 1>;



int modulo(int a, int b)
{
    return ((a % b) + b) % b;
}


void update_ghosts()
{
    // test
}

void find_ghosts(Atoms &atoms, double cutoff, double x_local_min, double x_local_max, double y_local_min, double y_local_max,
                 size_t &ne_size, size_t &nw_size, size_t &se_size, size_t &sw_size, size_t &n_size, size_t &s_size, size_t &e_size, size_t &w_size,
                 Atoms &NE, Atoms &NW, Atoms &SE, Atoms &SW, Atoms &N, Atoms &S, Atoms &E, Atoms &W)
{
    n_size = 0;
    s_size = 0;
    e_size = 0;
    w_size = 0;
    ne_size = 0;
    nw_size = 0;
    se_size = 0;
    sw_size = 0;

    NE.positions.resize(3, 0);
    NE.velocities.resize(3, 0);
    NE.forces.resize(3, 0);
    
    NW.positions.resize(3, 0);
    NW.velocities.resize(3, 0);
    NW.forces.resize(3, 0);

    SE.positions.resize(3, 0);
    SE.velocities.resize(3, 0);
    SE.forces.resize(3, 0);

    SW.positions.resize(3, 0);
    SW.velocities.resize(3, 0);
    SW.forces.resize(3, 0);

    N.positions.resize(3, 0);
    N.velocities.resize(3, 0);
    N.forces.resize(3, 0);

    S.positions.resize(3, 0);
    S.velocities.resize(3, 0);
    S.forces.resize(3, 0);

    E.positions.resize(3, 0);
    E.velocities.resize(3, 0);
    E.forces.resize(3, 0);

    W.positions.resize(3, 0);
    W.velocities.resize(3, 0);
    W.forces.resize(3, 0);

    // iterate over given atoms
    for (size_t i = 0; i < atoms.nb_atoms(); ++i)
    {
        // NE
        if (((atoms.positions(1, i)) > (y_local_max-cutoff)) && (atoms.positions(0, i) < (x_local_max-cutoff)))
        {
            NE.positions.resize(3, ne_size + 1);
            NE.velocities.resize(3, ne_size + 1);
            NE.forces.resize(3, ne_size + 1);
            //NE.names.resize(ne_size + 1);
            //NE.masses.resize(ne_size + 1);

            NE.positions.col(i) = atoms.positions.col(i);
            NE.velocities.col(i) = atoms.velocities.col(i);
            NE.forces.col(i) = atoms.forces.col(i);
            //NE.names[i] = atoms.names[i];
            //NE.masses[i] = atoms.masses[i];
            ne_size++;
        }
        // NW
        if (((atoms.positions(1, i)) > (y_local_max-cutoff)) && (atoms.positions(0, i) > (x_local_min+cutoff)))
        {
            NW.positions.resize(3, nw_size + 1);
            NW.velocities.resize(3, nw_size + 1);
            NW.forces.resize(3, nw_size + 1);
            //NW.names.resize(nw_size + 1);
            //NW.masses.resize(nw_size + 1);

            NW.positions.col(i) = atoms.positions.col(i);
            NW.velocities.col(i) = atoms.velocities.col(i);
            NW.forces.col(i) = atoms.forces.col(i);
            //NW.names[i] = atoms.names[i];
            //NW.masses[i] = atoms.masses[i];
            nw_size++;
        }
        // SE
        if (((atoms.positions(1, i)) < (y_local_min+cutoff)) && (atoms.positions(0, i) < (x_local_max-cutoff)))
        {
            SE.positions.resize(3, se_size + 1);
            SE.velocities.resize(3, se_size + 1);
            SE.forces.resize(3, se_size + 1);
            // SE.names.resize(se_size + 1);
            // SE.masses.resize(se_size + 1);

            SE.positions.col(i) = atoms.positions.col(i);
            SE.velocities.col(i) = atoms.velocities.col(i);
            SE.forces.col(i) = atoms.forces.col(i);
            // SE.names[i] = atoms.names[i];
            // SE.masses[i] = atoms.masses[i];
            se_size++;
        }
        // SW
        if (((atoms.positions(1, i)) < (y_local_min+cutoff)) && (atoms.positions(0, i) > (x_local_min+cutoff)))
        {
            SW.positions.resize(3, sw_size + 1);
            SW.velocities.resize(3, sw_size + 1);
            SW.forces.resize(3, sw_size + 1);
            //SW.names.resize(sw_size + 1);
            //SW.masses.resize(sw_size + 1);

            SW.positions.col(i) = atoms.positions.col(i);
            SW.velocities.col(i) = atoms.velocities.col(i);
            SW.forces.col(i) = atoms.forces.col(i);
            // SW.names[i] = atoms.names[i];
            // SW.masses[i] = atoms.masses[i];
            sw_size++;
        }
        // N
        if (atoms.positions(1, i) > (y_local_max-cutoff))
        {
            N.positions.resize(3, n_size + 1);
            N.velocities.resize(3, n_size + 1);
            N.forces.resize(3, n_size + 1);
            //N.names.resize(n_size + 1);
            //N.masses.resize(n_size + 1);

            N.positions.col(i) = atoms.positions.col(i);
            N.velocities.col(i) = atoms.velocities.col(i);
            N.forces.col(i) = atoms.forces.col(i);
            // N.names[i] = atoms.names[i];
            // N.masses[i] = atoms.masses[i];
            n_size++;
        }
        // S
        if (atoms.positions(1, i) < (y_local_min+cutoff))
        {
            S.positions.resize(3, s_size + 1);
            S.velocities.resize(3, s_size + 1);
            S.forces.resize(3, s_size + 1);
            //S.names.resize(s_size + 1);
            //S.masses.resize(s_size + 1);

            S.positions.col(i) = atoms.positions.col(i);
            S.velocities.col(i) = atoms.velocities.col(i);
            S.forces.col(i) = atoms.forces.col(i);
            // S.names[i] = atoms.names[i];
            // S.masses[i] = atoms.masses[i];
            s_size++;
        }
        // E
        if (atoms.positions(0, i) > (x_local_max-cutoff))
        {
            E.positions.resize(3, e_size + 1);
            E.velocities.resize(3, e_size + 1);
            E.forces.resize(3, e_size + 1);
            // E.names.resize(e_size + 1);
            // E.masses.resize(e_size + 1);

            E.positions.col(i) = atoms.positions.col(i);
            E.velocities.col(i) = atoms.velocities.col(i);
            E.forces.col(i) = atoms.forces.col(i);
            // E.names[i] = atoms.names[i];
            // E.masses[i] = atoms.masses[i];
            e_size++;
        }
        // W
        if (atoms.positions(0, i) < (x_local_min+cutoff))
        {
            W.positions.resize(3, w_size + 1);
            W.velocities.resize(3, w_size + 1);
            W.forces.resize(3, w_size + 1);
            // W.names.resize(w_size + 1);
            // W.masses.resize(w_size + 1);

            W.positions.col(i) = atoms.positions.col(i);
            W.velocities.col(i) = atoms.velocities.col(i);
            W.forces.col(i) = atoms.forces.col(i);
            //W.names[i] = atoms.names[i];
            //W.masses[i] = atoms.masses[i];
            w_size++;
        }
    }

    std::cout << "this rank has the following x coords in atoms: " << atoms.positions.row(0) << std::endl;

    std::cout << "this rank has the following x coords in W: " << W.positions.row(0) << std::endl;

    printf("NE: %ld NW: %ld SE: %ld SW: %ld N: %ld S: %ld E: %ld W: %ld\n", ne_size, nw_size, se_size, sw_size, n_size, s_size, e_size, w_size);

    // resizing name and mass arrays

    NE.names.resize(ne_size);
    NE.names.setConstant("Au");
    NE.masses.resize(ne_size);
    NE.masses.setConstant(1.0);

    NW.names.resize(nw_size);
    NW.names.setConstant("Au");
    NW.masses.resize(nw_size);
    NW.masses.setConstant(1.0);

    SE.names.resize(se_size);
    SE.names.setConstant("Au");
    SE.masses.resize(se_size);
    SE.masses.setConstant(1.0);

    SW.names.resize(sw_size);
    SW.names.setConstant("Au");
    SW.masses.resize(sw_size);
    SW.masses.setConstant(1.0);

    N.names.resize(n_size);
    N.names.setConstant("Au");
    N.masses.resize(n_size);
    N.masses.setConstant(1.0);

    S.names.resize(s_size);
    S.names.setConstant("Au");
    S.masses.resize(s_size);
    S.masses.setConstant(1.0);

    E.names.resize(e_size);
    E.names.setConstant("Au");
    E.masses.resize(e_size);
    E.masses.setConstant(1.0);

    W.names.resize(w_size);
    W.names.setConstant("Au");
    W.masses.resize(w_size);
    W.masses.setConstant(1.0);


    return;
}

size_t transfer_ghosts(Atoms &atoms, MPI_Comm comm, int rank, int size, MPI_Status &status,
                       int NE_rank, int NW_rank, int SE_rank, int SW_rank, int N_rank, int S_rank, int E_rank, int W_rank,
                       size_t NE_s_size, size_t NW_s_size, size_t SE_s_size, size_t SW_s_size,
                       size_t N_s_size, size_t S_s_size, size_t E_s_size, size_t W_s_size,
                       Atoms &NE_s, Atoms &NW_s, Atoms &SE_s, Atoms &SW_s, Atoms &N_s, Atoms &S_s, Atoms &E_s, Atoms &W_s,
                       Atoms &NE_r, Atoms &NW_r, Atoms &SE_r, Atoms &SW_r, Atoms &N_r, Atoms &S_r, Atoms &E_r, Atoms &W_r)
{
    /*
    size_t NE_s_size = NE_s.nb_atoms();
    size_t NW_s_size = NW_s.nb_atoms();
    size_t SE_s_size = SE_s.nb_atoms();
    size_t SW_s_size = SW_s.nb_atoms();
    size_t N_s_size = N_s.nb_atoms();
    size_t S_s_size = S_s.nb_atoms();
    size_t E_s_size = E_s.nb_atoms();
    size_t W_s_size = W_s.nb_atoms();
    */

    std::cout << "rank: " << rank << " NE_s_size: " << NE_s_size << " NW_s_size: " << NW_s_size << " SE_s_size: " << SE_s_size << " SW_s_size: " << SW_s_size << " N_s_size: " << N_s_size << " S_s_size: " << S_s_size << " E_s_size: " << E_s_size << " W_s_size: " << W_s_size << std::endl;

    // Gather Sizes for all ranks
    Eigen::ArrayXi NE_r_sizes(size);
    Eigen::ArrayXi NW_r_sizes(size);
    Eigen::ArrayXi SE_r_sizes(size);
    Eigen::ArrayXi SW_r_sizes(size);
    Eigen::ArrayXi N_r_sizes(size);
    Eigen::ArrayXi S_r_sizes(size);
    Eigen::ArrayXi E_r_sizes(size);
    Eigen::ArrayXi W_r_sizes(size);

    MPI_Allgather(&NE_s_size, 1, MPI_INT, NE_r_sizes.data(), 1, MPI_INT, comm);
    MPI_Allgather(&NW_s_size, 1, MPI_INT, NW_r_sizes.data(), 1, MPI_INT, comm);
    MPI_Allgather(&SE_s_size, 1, MPI_INT, SE_r_sizes.data(), 1, MPI_INT, comm);
    MPI_Allgather(&SW_s_size, 1, MPI_INT, SW_r_sizes.data(), 1, MPI_INT, comm);
    MPI_Allgather(&N_s_size, 1, MPI_INT, N_r_sizes.data(), 1, MPI_INT, comm);
    MPI_Allgather(&S_s_size, 1, MPI_INT, S_r_sizes.data(), 1, MPI_INT, comm);
    MPI_Allgather(&E_s_size, 1, MPI_INT, E_r_sizes.data(), 1, MPI_INT, comm);
    MPI_Allgather(&W_s_size, 1, MPI_INT, W_r_sizes.data(), 1, MPI_INT, comm);

    printf("Sizes: %d %d %d %d %d %d %d %d\n", NE_r_sizes(rank), NW_r_sizes(rank), SE_r_sizes(rank), SW_r_sizes(rank), N_r_sizes(rank), S_r_sizes(rank), E_r_sizes(rank), W_r_sizes(rank));

    bool skip_NE = false;
    bool skip_NW = false;
    bool skip_SE = false;
    bool skip_SW = false;
    bool skip_N = false;
    bool skip_S = false;
    bool skip_E = false;
    bool skip_W = false;

    if (N_rank == rank)
    {
        skip_N = true;
    }
    if (S_rank == rank)
    {
        skip_S = true;
    }
    if (E_rank == rank)
    {
        skip_E = true;
    }
    if (W_rank == rank)
    {
        skip_W = true;
    }
    if (NE_rank == rank)
    {
        skip_NE = true;
    }
    if (NW_rank == rank)
    {
        skip_NW = true;
    }
    if (SE_rank == rank)
    {
        skip_SE = true;
    }
    if (SW_rank == rank)
    {
        skip_SW = true;
    }

    if (NE_rank == SW_rank)
    {
        skip_NE = true;
        skip_SW = true;
    }

    if (SE_rank == NW_rank)
    {
        skip_SE = true;
        skip_NW = true;
    }

    std::cout << "rank: " << rank << " skip_NE: " << skip_NE << " skip_NW: " << skip_NW << " skip_SE: " << skip_SE << " skip_SW: " << skip_SW << " skip_N: " << skip_N << " skip_S: " << skip_S << " skip_E: " << skip_E << " skip_W: " << skip_W << std::endl;

    size_t nb_local = atoms.nb_atoms();

    // resize to highest possible size
    if (!skip_NE)
    {
        NE_r.positions.resize(3, NE_r_sizes[NE_rank]);
        NE_r.velocities.resize(3, NE_r_sizes[NE_rank]);
        NE_r.forces.resize(3, NE_r_sizes[NE_rank]);
        NE_r.names.resize(NE_r_sizes[NE_rank]);
        NE_r.names.setConstant("Au");
        NE_r.masses.resize(NE_r_sizes[NE_rank]);

        auto pos_recv{MPI::Eigen::sendrecv(NE_s.positions, NE_rank, rank, comm)};
        NE_r.positions = pos_recv;
        auto vel_recv{MPI::Eigen::sendrecv(NE_s.velocities, NE_rank, rank, comm)};
        NE_r.velocities = vel_recv;
        auto force_recv{MPI::Eigen::sendrecv(NE_s.forces, NE_rank, rank, comm)};
        NE_r.forces = force_recv;
        auto masses_recv{MPI::Eigen::sendrecv(NE_s.masses, NE_rank, rank, comm)};
        NE_r.masses = masses_recv;

        /*
        MPI_Sendrecv(NE_s.positions.row(0).data(), NE_s.positions.row(0).size(), MPI_DOUBLE, rank, MPI_ANY_TAG,
                     NE_r.positions.row(0).data(), NE_r_sizes[NE_rank], MPI_DOUBLE, NE_rank, MPI_ANY_TAG, comm, &status);
        MPI_Sendrecv(NE_s.positions.row(1).data(), NE_s.positions.row(1).size(), MPI_DOUBLE, rank, MPI_ANY_TAG,
                     NE_r.positions.row(1).data(), NE_r_sizes[NE_rank], MPI_DOUBLE, NE_rank, MPI_ANY_TAG, comm, &status);
        MPI_Sendrecv(NE_s.positions.row(2).data(), NE_s.positions.row(2).size(), MPI_DOUBLE, rank, MPI_ANY_TAG,
                     NE_r.positions.row(2).data(), NE_r_sizes[NE_rank], MPI_DOUBLE, NE_rank, MPI_ANY_TAG, comm, &status);

        MPI_Sendrecv(NE_s.velocities.row(0).data(), NE_s.velocities.row(0).size(), MPI_DOUBLE, rank, MPI_ANY_TAG,
                     NE_r.velocities.row(0).data(), NE_r_sizes[NE_rank], MPI_DOUBLE, NE_rank, MPI_ANY_TAG, comm, &status);
        MPI_Sendrecv(NE_s.velocities.row(1).data(), NE_s.velocities.row(1).size(), MPI_DOUBLE, rank, MPI_ANY_TAG,
                     NE_r.velocities.row(1).data(), NE_r_sizes[NE_rank], MPI_DOUBLE, NE_rank, MPI_ANY_TAG, comm, &status);
        MPI_Sendrecv(NE_s.velocities.row(2).data(), NE_s.velocities.row(2).size(), MPI_DOUBLE, rank, MPI_ANY_TAG,
                     NE_r.velocities.row(2).data(), NE_r_sizes[NE_rank], MPI_DOUBLE, NE_rank, MPI_ANY_TAG, comm, &status);
        
        MPI_Sendrecv(NE_s.forces.row(0).data(), NE_s.forces.row(0).size(), MPI_DOUBLE, rank, MPI_ANY_TAG,
                     NE_r.forces.row(0).data(), NE_r_sizes[NE_rank], MPI_DOUBLE, NE_rank, MPI_ANY_TAG, comm, &status);
        MPI_Sendrecv(NE_s.forces.row(1).data(), NE_s.forces.row(1).size(), MPI_DOUBLE, rank, MPI_ANY_TAG,
                     NE_r.forces.row(1).data(), NE_r_sizes[NE_rank], MPI_DOUBLE, NE_rank, MPI_ANY_TAG, comm, &status);
        MPI_Sendrecv(NE_s.forces.row(2).data(), NE_s.forces.row(2).size(), MPI_DOUBLE, rank, MPI_ANY_TAG,
                     NE_r.forces.row(2).data(), NE_r_sizes[NE_rank], MPI_DOUBLE, NE_rank, MPI_ANY_TAG, comm, &status);

        MPI_Sendrecv(NE_s.masses.data(), NE_s.masses.size(), MPI_DOUBLE, rank, MPI_ANY_TAG,
                     NE_r.masses.data(), NE_r_sizes[NE_rank], MPI_DOUBLE, NE_rank, MPI_ANY_TAG, comm, &status);
        */
    }
    
    std::cout << "NE done" << std::endl;

    if (!skip_NW)
    {
        NW_r.positions.resize(3, NW_r_sizes[NW_rank]);
        NW_r.velocities.resize(3, NW_r_sizes[NW_rank]);
        NW_r.forces.resize(3, NW_r_sizes[NW_rank]);
        NW_r.names.resize(NW_r_sizes[NW_rank]);
        NW_r.names.setConstant("Au");
        NW_r.masses.resize(NW_r_sizes[NW_rank]);

        auto pos_recv{MPI::Eigen::sendrecv(NW_s.positions, NW_rank, rank, comm)};
        NW_r.positions = pos_recv;
        auto vel_recv{MPI::Eigen::sendrecv(NW_s.velocities, NW_rank, rank, comm)};
        NW_r.velocities = vel_recv;
        auto force_recv{MPI::Eigen::sendrecv(NW_s.forces, NW_rank, rank, comm)};
        NW_r.forces = force_recv;
        auto masses_recv{MPI::Eigen::sendrecv(NW_s.masses, NW_rank, rank, comm)};
        NW_r.masses = masses_recv;

    }

    std::cout << "NW done" << std::endl;

    if (!skip_SE)
    {
        SE_r.positions.resize(3, SE_r_sizes[SE_rank]);
        SE_r.velocities.resize(3, SE_r_sizes[SE_rank]);
        SE_r.forces.resize(3, SE_r_sizes[SE_rank]);
        SE_r.names.resize(SE_r_sizes[SE_rank]);
        SE_r.names.setConstant("Au");
        SE_r.masses.resize(SE_r_sizes[SE_rank]);

        auto pos_recv{MPI::Eigen::sendrecv(SE_s.positions, SE_rank, rank, comm)};
        SE_r.positions = pos_recv;
        auto vel_recv{MPI::Eigen::sendrecv(SE_s.velocities, SE_rank, rank, comm)};
        SE_r.velocities = vel_recv;
        auto force_recv{MPI::Eigen::sendrecv(SE_s.forces, SE_rank, rank, comm)};
        SE_r.forces = force_recv;
        auto masses_recv{MPI::Eigen::sendrecv(SE_s.masses, SE_rank, rank, comm)};
        SE_r.masses = masses_recv;

    }

    std::cout << "SE done" << std::endl;

    if (!skip_SW)
    {
        SW_r.positions.resize(3, SW_r_sizes[SW_rank]);
        SW_r.velocities.resize(3, SW_r_sizes[SW_rank]);
        SW_r.forces.resize(3, SW_r_sizes[SW_rank]);
        SW_r.names.resize(SW_r_sizes[SW_rank]);
        SW_r.names.setConstant("Au");
        SW_r.masses.resize(SW_r_sizes[SW_rank]);
        
        auto pos_recv{MPI::Eigen::sendrecv(SW_s.positions, SW_rank, rank, comm)};
        SW_r.positions = pos_recv;
        auto vel_recv{MPI::Eigen::sendrecv(SW_s.velocities, SW_rank, rank, comm)};
        SW_r.velocities = vel_recv;
        auto force_recv{MPI::Eigen::sendrecv(SW_s.forces, SW_rank, rank, comm)};
        SW_r.forces = force_recv;
        auto masses_recv{MPI::Eigen::sendrecv(SW_s.masses, SW_rank, rank, comm)};
        SW_r.masses = masses_recv;

    }

    std::cout << "SW done" << std::endl;

    if (!skip_N)
    {
        N_r.positions.resize(3, N_r_sizes[N_rank]);
        N_r.velocities.resize(3, N_r_sizes[N_rank]);
        N_r.forces.resize(3, N_r_sizes[N_rank]);
        N_r.names.resize(N_r_sizes[N_rank]);
        N_r.names.setConstant("Au");
        N_r.masses.resize(N_r_sizes[N_rank]);

        auto pos_recv{MPI::Eigen::sendrecv(N_s.positions, N_rank, rank, comm)};
        N_r.positions = pos_recv;
        auto vel_recv{MPI::Eigen::sendrecv(N_s.velocities, N_rank, rank, comm)};
        N_r.velocities = vel_recv;
        auto force_recv{MPI::Eigen::sendrecv(N_s.forces, N_rank, rank, comm)};
        N_r.forces = force_recv;
        auto masses_recv{MPI::Eigen::sendrecv(N_s.masses, N_rank, rank, comm)};
        N_r.masses = masses_recv;

    }

    std::cout << "N done" << std::endl;

    if (!skip_S)
    {
        S_r.positions.resize(3, S_r_sizes[S_rank]);
        S_r.velocities.resize(3, S_r_sizes[S_rank]);
        S_r.forces.resize(3, S_r_sizes[S_rank]);
        S_r.names.resize(S_r_sizes[S_rank]);
        S_r.names.setConstant("Au");
        S_r.masses.resize(S_r_sizes[S_rank]);

        auto pos_recv{MPI::Eigen::sendrecv(S_s.positions, S_rank, rank, comm)};
        S_r.positions = pos_recv;
        auto vel_recv{MPI::Eigen::sendrecv(S_s.velocities, S_rank, rank, comm)};
        S_r.velocities = vel_recv;
        auto force_recv{MPI::Eigen::sendrecv(S_s.forces, S_rank, rank, comm)};
        S_r.forces = force_recv;
        auto masses_recv{MPI::Eigen::sendrecv(S_s.masses, S_rank, rank, comm)};
        S_r.masses = masses_recv;

    }

    std::cout << "S done" << std::endl;

    skip_E = true;

    if (!skip_E)
    {
        std::cout << "in E" << std::endl;

        E_r.positions.resize(3, E_r_sizes[E_rank]);
        E_r.velocities.resize(3, E_r_sizes[E_rank]);
        E_r.forces.resize(3, E_r_sizes[E_rank]);
        E_r.names.resize(E_r_sizes[E_rank]);
        E_r.names.setConstant("Au");
        E_r.masses.resize(E_r_sizes[E_rank]);

        std::cout << "in E after resize" << std::endl;

        double *recvval = new double[E_r_sizes[E_rank]];

        double *sendval;

        sendval = E_s.positions.row(0).data();

        std::cout << "sendval: " << sendval << std::endl;

        std::cout << "rank: " << rank << "Sendsize: " << E_s.positions.cols() << std::endl;

        std::cout << "rank: " << rank << "Recvsize: " << E_r_sizes[E_rank] << std::endl;

        MPI_Sendrecv(sendval, E_s.positions.cols(), MPI_DOUBLE, E_rank, 0, recvval, E_r_sizes[E_rank], MPI_DOUBLE, E_rank, 0, comm, &status);

        std::cout << "recvval: " << recvval[0] << std::endl;

        auto pos_recv{MPI::Eigen::sendrecv(E_s.positions, E_rank, E_rank, comm)};
        E_r.positions = pos_recv;
        auto vel_recv{MPI::Eigen::sendrecv(E_s.velocities, E_rank, E_rank, comm)};
        E_r.velocities = vel_recv;
        auto force_recv{MPI::Eigen::sendrecv(E_s.forces, E_rank, E_rank, comm)};
        E_r.forces = force_recv;
        auto masses_recv{MPI::Eigen::sendrecv(E_s.masses, E_rank, E_rank, comm)};
        E_r.masses = masses_recv;

    }

    std::cout << "E done" << std::endl;

    if (!skip_W)
    {
        std::cout << "in W" << std::endl;

        W_r.positions.resize(3, W_r_sizes[W_rank]);
        W_r.velocities.resize(3, W_r_sizes[W_rank]);
        W_r.forces.resize(3, W_r_sizes[W_rank]);
        W_r.names.resize(W_r_sizes[W_rank]);
        W_r.names.setConstant("Au");
        W_r.masses.resize(W_r_sizes[W_rank]);

        std::cout << "in W after resize" << std::endl;

        double *recvval = new double[W_r_sizes[W_rank]];

        double *sendval;

        sendval = W_s.positions.row(0).data();

        std::cout << "sendval: " << sendval << std::endl;

        std::cout << "rank: " << rank << "Sendsize: " << W_s.positions.cols() << std::endl;

        std::cout << "rank: " << rank << "sanity: " << W_s.positions.row(0) << std::endl;

        std::cout << "rank: " << rank << "Sendvalues:" << sendval[0] << std::endl;

        std::cout << "rank: " << rank << "Recvsize: " << W_r_sizes[W_rank] << std::endl;

        MPI_Sendrecv(sendval, W_s.positions.cols(), MPI_DOUBLE, W_rank, 0,
                     recvval, W_r_sizes[W_rank], MPI_DOUBLE, W_rank, 0, comm, &status);

        std::cout << "recvval: " << recvval[0] << std::endl;

        std::cin.get();

        auto pos_recv{MPI::Eigen::sendrecv(W_s.positions, W_rank, W_rank, comm)};
        W_r.positions = pos_recv;
        auto vel_recv{MPI::Eigen::sendrecv(W_s.velocities, W_rank, W_rank, comm)};
        W_r.velocities = vel_recv;
        auto force_recv{MPI::Eigen::sendrecv(W_s.forces, W_rank, W_rank, comm)};
        W_r.forces = force_recv;
        auto masses_recv{MPI::Eigen::sendrecv(W_s.masses, W_rank, W_rank, comm)};
        W_r.masses = masses_recv;

    }

    std::cout << "W done" << std::endl;

    size_t nb_ghosts = 0;

    if (!skip_NE)
    {
        nb_ghosts += NE_r_sizes[NE_rank];
    }
    if (!skip_NW)
    {
        nb_ghosts += NW_r_sizes[NW_rank];
    }
    if (!skip_SE)
    {
        nb_ghosts += SE_r_sizes[SE_rank];
    }
    if (!skip_SW)
    {
        nb_ghosts += SW_r_sizes[SW_rank];
    }
    if (!skip_N)
    {
        nb_ghosts += N_r_sizes[N_rank];
    }
    if (!skip_S)
    {
        nb_ghosts += S_r_sizes[S_rank];
    }
    if (!skip_E)
    {
        nb_ghosts += E_r_sizes[E_rank];
    }
    if (!skip_W)
    {
        nb_ghosts += W_r_sizes[W_rank];
    }

    size_t size_with_ghosts = nb_local + nb_ghosts;

    return size_with_ghosts;
}



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

    MPI_Request request;
    MPI_Status status;

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

    if (rank == 0)
    {
        outdata.open("./energy-cluster_923.txt");
    }

    if (fflag)
    {
        auto data{read_xyz(fvalue)};
        names.resize(std::get<0>(data).size());
        positions.resize(3, std::get<1>(data).cols());
        names = std::get<0>(data);
        positions = std::get<1>(data);

        fvalue = fvalue.substr(0, fvalue.find(".xyz"));

        outdata.close();
        outdata.open("./energy" + fvalue + ".txt");
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

    auto global_atom_count = positions.cols();

    Atoms C_staying(global_atom_count);
    Atoms N_recv(global_atom_count);
    Atoms S_recv(global_atom_count);
    Atoms E_recv(global_atom_count);
    Atoms W_recv(global_atom_count);
    Atoms NE_recv(global_atom_count);
    Atoms NW_recv(global_atom_count);
    Atoms SE_recv(global_atom_count);
    Atoms SW_recv(global_atom_count);

    Atoms N_send(global_atom_count);
    Atoms S_send(global_atom_count);
    Atoms E_send(global_atom_count);
    Atoms W_send(global_atom_count);
    Atoms NE_send(global_atom_count);
    Atoms NW_send(global_atom_count);
    Atoms SE_send(global_atom_count);
    Atoms SW_send(global_atom_count);

    Atoms N_ghost_send(global_atom_count);
    Atoms S_ghost_send(global_atom_count);
    Atoms E_ghost_send(global_atom_count);
    Atoms W_ghost_send(global_atom_count);
    Atoms NE_ghost_send(global_atom_count);
    Atoms NW_ghost_send(global_atom_count);
    Atoms SE_ghost_send(global_atom_count);
    Atoms SW_ghost_send(global_atom_count);

    Atoms N_ghost_recv(global_atom_count);
    Atoms S_ghost_recv(global_atom_count);
    Atoms E_ghost_recv(global_atom_count);
    Atoms W_ghost_recv(global_atom_count);
    Atoms NE_ghost_recv(global_atom_count);
    Atoms NW_ghost_recv(global_atom_count);
    Atoms SE_ghost_recv(global_atom_count);
    Atoms SW_ghost_recv(global_atom_count);

    Atoms atoms = Atoms(positions, velocities);

    // Atoms atoms = init_cube(100, 1.0);

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


    outdata << timestep << " " << nb_steps << " " << epsilon << " " << sigma << " " << cutoff << " " << q_delta << " " << relaxation_time << " " << eq_steps << std::endl;


    std::stringstream ss;
    std::string output_path;

    NeighborList neighbor_list;
    neighbor_list.update(atoms, cutoff);
    
    //std::cout << "maxcoeff" << atoms.positions.rowwise().maxCoeff().maxCoeff() << std::endl;

    //Eigen::Array3d d_size{atoms.positions.rowwise().maxCoeff() -
    //                      atoms.positions.rowwise().minCoeff()};

    // std::cout << "d_size: " << d_size << std::endl;

    Eigen::Array3i d_decomposition = {x, y, 1};

    assert(x * y * 1 == size);

    // std::cout << "d_decomposition: " << d_decomposition << std::endl;

    // periodicity breaks the program.
    Eigen::Array3i d_periodicity = {0, 0, 0};

    // Create Cartesian communicator and get cartesian coordinates of the current rank.

    Eigen::Array3i rank_cart_coordinate;

    MPI_Comm MPI_COMM_CART;

    MPI_Cart_create(MPI_COMM_WORLD, 3, d_decomposition.data(), d_periodicity.data(), 1, &MPI_COMM_CART);
    MPI_Cart_coords(MPI_COMM_CART, rank, 3, rank_cart_coordinate.data());

    std::cout << "rank: " << rank << " has rank_cart_coordinate: " << rank_cart_coordinate << std::endl;

    Eigen::ArrayXi coords_of_rank(3*size);

    for (int i = 0; i < 3*size; i+=3)
    {
        Eigen::Array3i temp_rank_cart_coordinate;
        MPI_Cart_coords(MPI_COMM_CART, (int) i/3, 3, temp_rank_cart_coordinate.data());
        coords_of_rank[i] = temp_rank_cart_coordinate[0];
        coords_of_rank[i+1] = temp_rank_cart_coordinate[1];
        coords_of_rank[i+2] = temp_rank_cart_coordinate[2];
    }

     printf("Cart Coord: ");
    std::cout << coords_of_rank << std::endl;

    auto cart_size_x = d_decomposition[0];
    auto cart_size_y = d_decomposition[1];
    auto cart_size_z = d_decomposition[2];

    Eigen::Array3i our_coords(coords_of_rank[rank*3], coords_of_rank[rank*3], coords_of_rank[rank*3]);

    Eigen::Array3i NE_coords = our_coords + Eigen::Array3i(1, 1, 0);
    Eigen::Array3i NW_coords = our_coords + Eigen::Array3i(-1, 1, 0);
    Eigen::Array3i SE_coords = our_coords + Eigen::Array3i(1, -1, 0);
    Eigen::Array3i SW_coords = our_coords + Eigen::Array3i(-1, -1, 0);
    Eigen::Array3i N_coords = our_coords + Eigen::Array3i(0, 1, 0);
    Eigen::Array3i S_coords = our_coords + Eigen::Array3i(0, -1, 0);
    Eigen::Array3i E_coords = our_coords + Eigen::Array3i(1, 0, 0);
    Eigen::Array3i W_coords = our_coords + Eigen::Array3i(-1, 0, 0);

    // set each coord back into the range of the decomposition

    std::cout << "rank: " << rank << " has cart_size_x: " << cart_size_x << std::endl;
    std::cout << "rank: " << rank << " has cart_size_y: " << cart_size_y << std::endl;

    NE_coords[0] = modulo(NE_coords[0], cart_size_x);
    NE_coords[1] = modulo(NE_coords[1], cart_size_y);
    NW_coords[0] = modulo(NW_coords[0], cart_size_x);
    NW_coords[1] = modulo(NW_coords[1], cart_size_y);
    SE_coords[0] = modulo(SE_coords[0], cart_size_x);
    SE_coords[1] = modulo(SE_coords[1], cart_size_y);
    SW_coords[0] = modulo(SW_coords[0], cart_size_x);
    SW_coords[1] = modulo(SW_coords[1], cart_size_y);
    N_coords[0] = modulo(N_coords[0], cart_size_x);
    N_coords[1] = modulo(N_coords[1], cart_size_y);
    S_coords[0] = modulo(S_coords[0], cart_size_x);
    S_coords[1] = modulo(S_coords[1], cart_size_y);
    E_coords[0] = modulo(E_coords[0], cart_size_x);
    E_coords[1] = modulo(E_coords[1], cart_size_y);
    W_coords[0] = modulo(W_coords[0], cart_size_x);
    W_coords[1] = modulo(W_coords[1], cart_size_y);

    std::cout << "rank: " << rank << " has NE_coords: " << NE_coords << std::endl;
    std::cout << "rank: " << rank << " has NW_coords: " << NW_coords << std::endl;
    std::cout << "rank: " << rank << " has SE_coords: " << SE_coords << std::endl;
    std::cout << "rank: " << rank << " has SW_coords: " << SW_coords << std::endl;
    std::cout << "rank: " << rank << " has N_coords: " << N_coords << std::endl;
    std::cout << "rank: " << rank << " has S_coords: " << S_coords << std::endl;
    std::cout << "rank: " << rank << " has E_coords: " << E_coords << std::endl;
    std::cout << "rank: " << rank << " has W_coords: " << W_coords << std::endl;

    int NE_rank = 0;
    int NW_rank = 0;
    int SE_rank = 0;
    int SW_rank = 0;
    int N_rank = 0;
    int S_rank = 0;
    int E_rank = 0;
    int W_rank = 0;

    for (int i = 0; i < size; i++)
    {
        if (coords_of_rank[i*3] == NE_coords[0] && coords_of_rank[i*3+1] == NE_coords[1])
        {
            NE_rank = i;
        }
        if (coords_of_rank[i*3] == NW_coords[0] && coords_of_rank[i*3+1] == NW_coords[1])
        {
            NW_rank = i;
        }
        if (coords_of_rank[i*3] == SE_coords[0] && coords_of_rank[i*3+1] == SE_coords[1])
        {
            SE_rank = i;
        }
        if (coords_of_rank[i*3] == SW_coords[0] && coords_of_rank[i*3+1] == SW_coords[1])
        {
            SW_rank = i;
        }
        if (coords_of_rank[i*3] == N_coords[0] && coords_of_rank[i*3+1] == N_coords[1])
        {
            N_rank = i;
        }
        if (coords_of_rank[i*3] == S_coords[0] && coords_of_rank[i*3+1] == S_coords[1])
        {
            S_rank = i;
        }
        if (coords_of_rank[i*3] == E_coords[0] && coords_of_rank[i*3+1] == E_coords[1])
        {
            E_rank = i;
        }
        if (coords_of_rank[i*3] == W_coords[0] && coords_of_rank[i*3+1] == W_coords[1])
        {
            W_rank = i;
        }
    }

    std::cout << "rank: " << rank << " has NE_rank: " << NE_rank << std::endl;
    std::cout << "rank: " << rank << " has NW_rank: " << NW_rank << std::endl;
    std::cout << "rank: " << rank << " has SE_rank: " << SE_rank << std::endl;
    std::cout << "rank: " << rank << " has SW_rank: " << SW_rank << std::endl;
    std::cout << "rank: " << rank << " has N_rank: " << N_rank << std::endl;
    std::cout << "rank: " << rank << " has S_rank: " << S_rank << std::endl;
    std::cout << "rank: " << rank << " has E_rank: " << E_rank << std::endl;
    std::cout << "rank: " << rank << " has W_rank: " << W_rank << std::endl;


    std::cout << "rank: " << rank << " has coords_of_rank: " << coords_of_rank << std::endl;

    // std::cout << "rank: " << rank << " following positions: " << atoms.positions << std::endl;

    auto min_pos = -atoms.positions.rowwise().maxCoeff().maxCoeff();
    auto max_pos = atoms.positions.rowwise().maxCoeff().maxCoeff();
    
    std::cout << "min_pos: " << min_pos << std::endl;
    std::cout << "max_pos: " << max_pos << std::endl;

    double max_x = max_pos + 2.0 * cutoff;
    double max_y = max_pos + 2.0 * cutoff;
    double min_x = min_pos - 2.0 * cutoff;
    double min_y = min_pos - 2.0 * cutoff;

    double dx = (max_x - min_x) / d_decomposition[0];
    double dy = (max_y - min_y) / d_decomposition[1];

    std::cout << "dx: " << dx << std::endl;
    std::cout << "dy: " << dy << std::endl;


    size_t nb_global{atoms.nb_atoms()};
    Atoms total_atoms = Atoms(nb_global);

    Atoms domain_atoms;

    double x_local_min = min_x + rank_cart_coordinate[0] * dx;
    double x_local_max = min_x + (rank_cart_coordinate[0] + 1) * dx;
    double y_local_min = min_y + rank_cart_coordinate[1] * dy;
    double y_local_max = min_y + (rank_cart_coordinate[1] + 1) * dy;

    std::cout << "rank: " << rank << " x_local_min: " << x_local_min << std::endl;
    std::cout << "rank: " << rank << " x_local_max: " << x_local_max << std::endl;
    std::cout << "rank: " << rank << " y_local_min: " << y_local_min << std::endl;
    std::cout << "rank: " << rank << " y_local_max: " << y_local_max << std::endl;

    std::cout << "rank: " << rank << " has positions: " << atoms.positions.row(0) << std::endl;
    std::cout << "rank: " << rank << " has positions: " << atoms.positions.row(1) << std::endl;
    std::cout << "rank: " << rank << " has positions: " << atoms.positions.row(2) << std::endl;
    
    size_t nb_local = 0;

    // Distribute atoms to their domains
    for (size_t i = 0; i < nb_global; ++i)
    {
        // if atom is in the local domain
        if (atoms.positions(0, i) >= x_local_min && atoms.positions(0, i) < x_local_max &&
            atoms.positions(1, i) >= y_local_min && atoms.positions(1, i) < y_local_max)
        {
            std::cout << "rank: " << rank << " atom: " << i << " is in the local domain" << std::endl;

            // resize domain_atoms
            domain_atoms.positions.conservativeResize(3, nb_local + 1);
            domain_atoms.velocities.conservativeResize(3, nb_local + 1);
            domain_atoms.forces.conservativeResize(3, nb_local + 1);
            //domain_atoms.names.resize(nb_local + 1);
            // domain_atoms.masses.resize(nb_local + 1);
            
            std::cout << "rank: " << rank << " has positions: " << atoms.positions.row(0) << std::endl;
            std::cout << "rank: " << rank << " has positions: " << atoms.positions.row(1) << std::endl;
            std::cout << "rank: " << rank << " has positions: " << atoms.positions.row(2) << std::endl;

            // transfer full atom info to domain_atoms
            auto temp0{atoms.positions(0, i)};
            auto temp1{atoms.positions(1, i)};
            auto temp2{atoms.positions(2, i)};

            std::cout << "rank: " << rank << " has temp0: " << temp0 << std::endl;
            std::cout << "rank: " << rank << " has temp1: " << temp1 << std::endl;
            std::cout << "rank: " << rank << " has temp2: " << temp2 << std::endl;

            std::cin.get();

            domain_atoms.positions.row(0).col(nb_local) = temp0;
            domain_atoms.positions.row(1).col(nb_local) = temp1;
            domain_atoms.positions.row(2).col(nb_local) = temp2;

            std::cout << "rank: " << rank << " has positions in local: " << domain_atoms.positions.row(0) << std::endl;
            std::cout << "rank: " << rank << " has positions in local: " << domain_atoms.positions.row(1) << std::endl;
            std::cout << "rank: " << rank << " has positions in local: " << domain_atoms.positions.row(2) << std::endl;

            std::cin.get();

            domain_atoms.velocities(0, nb_local) = atoms.velocities(0, i);
            domain_atoms.velocities(1, nb_local) = atoms.velocities(1, i);
            domain_atoms.velocities(2, nb_local) = atoms.velocities(2, i);
            domain_atoms.forces(0, nb_local) = atoms.forces(0, i);
            domain_atoms.forces(1, nb_local) = atoms.forces(1, i);
            domain_atoms.forces(2, nb_local) = atoms.forces(2, i);
            // domain_atoms.names[nb_local] = atoms.names[i];
            // domain_atoms.masses[nb_local] = atoms.masses[i];
            nb_local++;
        }
    }

    std::cout << "rank: " << rank << " has positions: " << domain_atoms.positions.row(0) << std::endl;
    std::cout << "rank: " << rank << " has positions: " << domain_atoms.positions.row(1) << std::endl;
    std::cout << "rank: " << rank << " has positions: " << domain_atoms.positions.row(2) << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    domain_atoms.names.resize(domain_atoms.nb_atoms());
    domain_atoms.names.setConstant("Au");
    domain_atoms.masses.resize(domain_atoms.nb_atoms());
    domain_atoms.masses.setConstant(1.0);

    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "distibuted atoms to their domains" << std::endl;

    size_t NE_s_size = 0;
    size_t NW_s_size = 0;
    size_t SE_s_size = 0;
    size_t SW_s_size = 0;
    size_t N_s_size = 0;
    size_t S_s_size = 0;
    size_t E_s_size = 0;
    size_t W_s_size = 0;

    size_t nb_local_with_ghosts = 0 + nb_local;

    find_ghosts(domain_atoms, cutoff, x_local_min, x_local_max, y_local_min, y_local_max,
                NE_s_size, NW_s_size, SE_s_size, SW_s_size, N_s_size, S_s_size, E_s_size, W_s_size,
                NE_ghost_send, NW_ghost_send, SE_ghost_send, SW_ghost_send,
                N_ghost_send, S_ghost_send, E_ghost_send, W_ghost_send);

    printf("NE: %ld NW: %ld SE: %ld SW: %ld N: %ld S: %ld E: %ld W: %ld\n", NE_s_size, NW_s_size, SE_s_size, SW_s_size, N_s_size, S_s_size, E_s_size, W_s_size);

    std::cout << "rank: " << rank << " has particles with these x coords: " << W_ghost_send.positions.row(0) << std::endl;

    nb_local_with_ghosts = transfer_ghosts(domain_atoms, MPI_COMM_WORLD, rank, size, status,
                    NE_rank, NW_rank, SE_rank, SW_rank, N_rank, S_rank, E_rank, W_rank,
                    NE_s_size, NW_s_size, SE_s_size, SW_s_size,
                    N_s_size, S_s_size, E_s_size, W_s_size,
                    NE_ghost_send, NW_ghost_send, SE_ghost_send, SW_ghost_send,
                    N_ghost_send, S_ghost_send, E_ghost_send, W_ghost_send,
                    NE_ghost_recv, NW_ghost_recv, SE_ghost_recv, SW_ghost_recv,
                    N_ghost_recv, S_ghost_recv, E_ghost_recv, W_ghost_recv);

    MPI_Barrier(MPI_COMM_WORLD);

    domain_atoms.names.resize(domain_atoms.nb_atoms());
    domain_atoms.names.setConstant("Au");
    domain_atoms.masses.resize(domain_atoms.nb_atoms());
    domain_atoms.masses.setConstant(1.0);

    MPI_Barrier(MPI_COMM_WORLD);

    // after this every rank has its own domain_atoms (supposedly like after using domain.enable(atoms))

    neighbor_list.update(domain_atoms, cutoff);
    ducastelle(domain_atoms, neighbor_list);

#pragma endregion INITIALIZATION

    for (size_t i = 0; i < eq_steps; ++i)
    {
        // std::cout << "DoingEqStep" << std::endl;
        verlet_step1(domain_atoms.positions, domain_atoms.velocities, domain_atoms.forces, timestep);

        // domain.exchange_atoms(atoms);

        // domain.update_ghosts(atoms, cutoff * 2);

        neighbor_list.update(domain_atoms, cutoff);

        std::cout << "after neighbor_list update" << std::endl;

        double potential_energy{MPI::allreduce(ducastelle_local(domain_atoms, neighbor_list, nb_local, cutoff), MPI_SUM, MPI_COMM_WORLD)};

        verlet_step2(domain_atoms.velocities, atoms.forces, timestep);

        double kinetic_energy{MPI::allreduce(kin_energy_local(domain_atoms, nb_local), MPI_SUM, MPI_COMM_WORLD)};

        total_energy = potential_energy + kinetic_energy;

        if (rank == 0)
        {
            std::cout << "EqStep: " << i << std::endl;
            std::cout << std::setprecision(9) << "E_pot: " << potential_energy << std::endl;
            std::cout << std::setprecision(9) << "E_kin: " << kinetic_energy << std::endl;
            std::cout << std::setprecision(9) << "E_tot: " << total_energy << std::endl;
        }


    }

    
    if (eq_steps > 0)
    {

        // domain.exchange_atoms(atoms);
        // domain.update_ghosts(atoms, cutoff * 2);

        neighbor_list.update(domain_atoms, cutoff);
    }
    

    for (size_t i = 0; i < nb_steps; ++i)
    {

        if (i % 1 == 100) {std::cout << "rank: " << rank << " Step: " << i << std::endl;}

        verlet_step1(domain_atoms.positions, domain_atoms.velocities, domain_atoms.forces, timestep);

        // domain.exchange_atoms(atoms);

        // domain.update_ghosts(atoms, cutoff * 2);

        neighbor_list.update(domain_atoms, cutoff);

        double potential_local = ducastelle_local(domain_atoms, neighbor_list, nb_local, cutoff);

        double potential_energy{MPI::allreduce(potential_local, MPI_SUM, MPI_COMM_WORLD)};

        verlet_step2(domain_atoms.velocities, domain_atoms.forces, timestep);

        if (berendsen)
        {
            berendsen_thermostat(domain_atoms, 0.0, timestep, 1.0);
        }

        double kin_energy_l{kin_energy_local(domain_atoms, nb_local)};
        
        double kinetic_energy{MPI::allreduce(kin_energy_l, MPI_SUM, MPI_COMM_WORLD)};

        total_energy = potential_energy + kinetic_energy;


        if (rank == 0)
        {
            if (i % relaxation_time == 0)
            {
                avg_temp = 0.0;
                auto kinetic_abs = std::abs(kinetic_energy);
                atoms.velocities *= sqrt(1.0 + std::abs((q_delta+kinetic_abs)/(kinetic_abs-1.0)));
            }

            if (((int)i % (relaxation_time)) > (relaxation_time/2)  || (int) i == 0)
            {
                avg_temp += 2.0 * kinetic_energy / (3.0 * atoms.positions.cols());
            }

            outdata << "[" << i << "," << potential_energy << "," << kinetic_energy << "," << potential_energy + kinetic_energy << "," << avg_temp  << "]" << std::endl;


            if (((int) i+1) % (relaxation_time) == 0 || (int) i == 0)
            {
                std::cout << std::setprecision(9) << "E_kin: " << kinetic_energy << std::endl;
                std::cout << std::setprecision(9) << "E_tot: " << potential_energy + kinetic_energy << std::endl;

                avg_temp /= (relaxation_time/2);


                outdata << "[" << i << "," << potential_energy << "," << kinetic_energy << "," << potential_energy + kinetic_energy << "," << (2.0 * kinetic_energy / (3.0 * atoms.positions.cols())) << "," << avg_temp  << "]" << std::endl;

                ss.str(std::string());
                ss << std::setw(8) << std::setfill('0') << std::to_string(i);
                output_path = "./traj-923-" + ss.str() + ".xyz";
                if (fflag)
                {
                    // preferring several files with a sequence number as they are not too big to be in their own repo
                    output_path = "./traj-" + fvalue + "-" + ss.str() + ".xyz";
                }
                std::cout << "output_path: " << output_path << std::endl;

                std::cout << "writing to file" << std::endl;

                std::cout << "total_atoms.names.size(): " << total_atoms.names.size() << std::endl;
                std::cout << "total_atoms.positions.cols(): " << total_atoms.positions.cols() << std::endl;
                std::cout << "total_atoms.velocities.rows(): " << total_atoms.velocities.cols() << std::endl;
                std::cout << "total_atoms.forces.cols(): " << total_atoms.forces.cols() << std::endl;
                std::cout << "total_atoms.masses.size(): " << total_atoms.masses.size() << std::endl;

                total_atoms.names.resize(total_atoms.positions.cols());
                total_atoms.names.setConstant("Au");

                write_xyz_filename(output_path, total_atoms);

                std::cout << "output_path: " << output_path << std::endl;
            }
            else
            {
                outdata << "[" << i << "," << potential_energy << "," << kinetic_energy << "," << potential_energy + kinetic_energy << "," << (2.0 * kinetic_energy / (3.0 * atoms.positions.cols())) << "]" << std::endl;
            }
        }

    }


    //std::cout << std::setprecision(9) << "Final E_pot: " << potential_energy << std::endl;
    //std::cout << std::setprecision(9) << "Final E_kin: " << kinetic_energy << std::endl;
    //std::cout << std::setprecision(9) << "Final E_tot: " << total_energy << std::endl;

    if (rank == 0)
    {
        outdata.close();
    }

    // domain.disable(atoms);

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
