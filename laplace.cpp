// Solves Laplace's equation using Jacobi's iteration
// mpic++ -O3 -Wall laplace.cpp -o laplace
// mpirun -np <nprocs> ./laplace

#include <iostream>
#include <cmath>
#include <chrono>
#include "mpi.h"

int nrows, ncols;

enum {W, E, S, N};

void set_bc(int neighbors[4], double *xlocal, double *xnew)
{
    // West
    if (neighbors[W] < 0)
        for (int i = 0; i < nrows; ++i)
        {
            xlocal[i * ncols] = xlocal[i * ncols + 1] = 1.0;
            xnew[i * ncols] = xnew[i * ncols + 1] = 1.0;
        }
    // East
    if (neighbors[E] < 0)
        for (int i = 0; i < nrows; ++i)
        {
            xlocal[i * ncols + ncols - 2] = xlocal[i * ncols + ncols - 1] = 1.0;
            xnew[i * ncols + ncols - 2] = xnew[i * ncols + ncols - 1] = 1.0;
        }
    // South
    if (neighbors[S] < 0)
        for (int i = 0; i < ncols; ++i)
        {
            xlocal[(nrows - 2) * ncols + i] = xlocal[(nrows - 1) * ncols + i] = 1.0;
            xnew[(nrows - 2) * ncols + i] = xnew[(nrows - 1) * ncols + i] = 1.0;
        }
    // North
    if (neighbors[N] < 0)
        for (int i = 0; i < ncols; ++i)
        {
            xlocal[i] = xlocal[ncols + i] = 1.0;
            xnew[i] = xnew[ncols + i] = 1.0;
        }
}

void exchange_halo(int neighbors[4], double *xlocal, MPI_Datatype column, MPI_Comm comm2D)
{
    // Send row to NORTH neighbor, and receive row from SOUTH neighbor
    MPI_Sendrecv(&xlocal[ncols], ncols, MPI_DOUBLE, neighbors[N], 0, &xlocal[(nrows - 1) * ncols], ncols, MPI_DOUBLE, neighbors[S], 0, comm2D, MPI_STATUS_IGNORE);
    // Send row to SOUTH neighbor, and receive row from NORTH neighbor
    MPI_Sendrecv(&xlocal[(nrows - 2) * ncols], ncols, MPI_DOUBLE, neighbors[S], 0, &xlocal[0], ncols, MPI_DOUBLE, neighbors[N], 0, comm2D, MPI_STATUS_IGNORE);
    // Send column to WEST neighbor, and receive column from EAST neighbor
    MPI_Sendrecv(&xlocal[1], 1, column, neighbors[W], 1, &xlocal[ncols - 1], 1, column, neighbors[E], 1, comm2D, MPI_STATUS_IGNORE);
    // Send column to EAST neighbor, and receive column from WEST neighbor
    MPI_Sendrecv(&xlocal[ncols - 2], 1, column, neighbors[E], 1, &xlocal[0], 1, column, neighbors[W], 1, comm2D, MPI_STATUS_IGNORE);
}

void compute_next(double *xlocal, double *xnew, double *diff_norm)
{
    *diff_norm = 0.0;
    for (int i = 1; i < nrows - 1; ++i)
        for (int j = 1; j < ncols - 1; ++j)
        {   
            int idx = i * ncols + j;
            // Method of relaxation
            xnew[idx] = 0.25 * (xlocal[idx + ncols] +
                                 xlocal[idx - ncols] +
                                 xlocal[idx + 1] +
                                 xlocal[idx - 1]);
            *diff_norm += (xnew[idx] - xlocal[idx]) *
                          (xnew[idx] - xlocal[idx]);
        }
}

int main(int argc, char *argv[])
{
    int my_rank, nprocs, npts, npts_x, npts_y, np_x, np_y, max_iter, count = 0, dims[2], periodicity[2], neighbors[4];
    double *xlocal, *xnew, diff_norm, gdiff_norm = 0.0, tolerance = 0.01;
    std::chrono::time_point<std::chrono::steady_clock> tic, toc;
    MPI_Comm comm2D;
    MPI_Datatype column;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 6 && !my_rank)
    {
        std::cerr << "Five command line arguments were expected!\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    npts_x = atoi(argv[1]);
    npts_y = atoi(argv[2]);
    np_x = atoi(argv[3]);
    np_y = atoi(argv[4]);
    max_iter = atoi(argv[5]);
    
    if (nprocs != np_x * np_y && !my_rank)
    {
        std::cerr << "Number of processes not equal to number of subdomains!\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (((npts_x % np_x) || (npts_y % np_y)) && !my_rank)
    {
        std::cerr << "Number of points not evenly divisible by number of subdomains!\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    nrows = npts_y / np_y + 2; // Height of subdomain (expanded)
    ncols = npts_x / np_x + 2; // Width of subdomain (expanded)
    npts = nrows * ncols; // Total number of points (including the bdry)

    xlocal = new double[npts]();
    xnew = new double[npts];

    dims[0] = np_x;
    dims[1] = np_y;
    periodicity[0] = 0;
    periodicity[1] = 0;                             // reorder = 0
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodicity, 0, &comm2D);

    // Who are my neighbors?
    MPI_Cart_shift(comm2D, 0, 1, &neighbors[W], &neighbors[E]);
    MPI_Cart_shift(comm2D, 1, 1, &neighbors[S], &neighbors[N]);
    
    // Strided datatype for exchange of columns
    MPI_Type_vector(nrows, 1, ncols, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    // Set boundary conditions
    set_bc(neighbors, xlocal, xnew);

    MPI_Barrier(MPI_COMM_WORLD); // For timing purposes
    
    if (!my_rank)
    {
        std::cout << "Computing...\n";
        tic = std::chrono::steady_clock::now();
    }

    while (count < max_iter)
    {
        exchange_halo(neighbors, xlocal, column, comm2D);

        compute_next(xlocal, xnew, &diff_norm);

        MPI_Allreduce(&diff_norm, &gdiff_norm, 1, MPI_DOUBLE, MPI_SUM, comm2D);

        gdiff_norm = sqrt(gdiff_norm);

        ++count;

        if (gdiff_norm < tolerance)
            break;

        std::swap(xlocal, xnew);
    }

    MPI_Barrier(MPI_COMM_WORLD); // For timing purposes

    if (!my_rank)
    {
        toc = std::chrono::steady_clock::now();
        std::cout << "At iteration #" << count << ", norm is " << gdiff_norm << std::endl;
        std::cout << "Elapsed time = " << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() << " ms\n";
    }

#if 0
    // MPI I/O
    MPI_File fh;
    MPI_File_open(comm2D, "output.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    // If reorder = 0 then rank(MPI_COMM_WORLD) == rank(comm2D)
    MPI_Offset offset = my_rank * npts * sizeof(double);
    MPI_File_write_at(fh, offset, xlocal, npts, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
#endif

    delete[] xlocal;
    delete[] xnew;

    MPI_Type_free(&column);
    MPI_Finalize();

    return 0;
}
