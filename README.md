[![DOI](https://zenodo.org/badge/191792781.svg)](https://zenodo.org/badge/latestdoi/191792781)

benchParanumal
=======

`benchParanumal` contains several benchmark problems set out, or inspired by, the [Center for Efficient Exascale Discretizations](https://ceed.exascaleproject.org/) (CEED) [Bake-off Problems](https://ceed.exascaleproject.org/bps/).


How to compile `benchParanumal`
------------------------

There are a couple of prerequisites for building `benchParanumal`;

- MPI
- OpenBlas

Installing `MPI` and `OpenBlas` can be done using whatever package manager your
operating system provides.

`OCCA` is packaged with `benchParanumal` in a git submodule. Either clone with `--recursive` or run
```
git submodule init
git submoduel update
```

To build `benchParanumal`:

    $ git clone --recursive <CEED repo>
    $ cd /path/to/CEED
    $ export LIBP_BLAS_DIR=/path/to/openblas
    $ make -j `nproc`

If your MPI supports GPU-aware RDMA functionality, you can optionally build `benchParanumal` with this support via:

    $ make -j `nproc` --gpu-aware-mpi=true

How to run `benchParanumal`
--------------------

`benchParanumal` contains two distinct types of tests:

2. BK - Benchmark Kernels
3. BP - Benchmark Problems

The usage of each benchmark, outside of the provide run scripts, can be found with the `-h` option. For example:

    $ mpirun -np 1 ./BK/BK1/BK1 -h

    Name:     [THREAD MODEL]
    CL keys:  [-m, --mode]
    Description: OCCA's Parallel execution platform
    Possible values: { Serial, OpenMP, CUDA, HIP, OpenCL }

    Name:     [PLATFORM NUMBER]
    CL keys:  [-pl, --platform]
    Description: Parallel platform number (used in OpenCL mode)

    Name:     [DEVICE NUMBER]
    CL keys:  [-d, --device]
    Description: Parallel device number

    Name:     [ELEMENT TYPE]
    CL keys:  [-e, --elements]
    Description: Type of mesh elements
    Possible values: { Tri, Quad, Tet, Hex }

    Name:     [BOX NX]
    CL keys:  [-nx, --dimx]
    Description: Number of elements in X-dimension per rank

    Name:     [BOX NY]
    CL keys:  [-ny, --dimy]
    Description: Number of elements in Y-dimension per rank

    Name:     [BOX NZ]
    CL keys:  [-nz, --dimz]
    Description: Number of elements in Z-dimension per rank

    Name:     [POLYNOMIAL DEGREE]
    CL keys:  [-p, --degree]
    Description: Degree of polynomial finite element space
    Possible values: { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }

    Name:     [VERBOSE]
    CL keys:  [-v, --verbose]
    Description: Enable verbose output
    Possible values: { TRUE, FALSE }

    Name:     [HELP]
    CL keys:  [-h, --help]
    Description: Print this help message


Here is an example large problem size that you can run on one GPU:

    $ mpirun -np 1 ./BP/BP5/BP5 -m HIP -nx 24 -ny 24 -nz 24 -p 15 -v

Running on multiple GPUs can by done by passing a larger argument to `np`:

    $ mpirun -np 4 ./BP/BP5/BP5 -m HIP -nx 24 -ny 24 -nz 24 -p 15 -v


Verifying correctness
---------------------

To verify that the computation is correct, add the `-v` option to the command
line.  Example output towards the end of the run may look like this:

    CG: it 96, r norm 1.405229334496e-04, alpha = 2.686587e+00
    CG: it 97, r norm 1.375460859099e-04, alpha = 2.540830e+00
    CG: it 98, r norm 1.198097786957e-04, alpha = 2.780510e+00
    CG: it 99, r norm 1.108821042895e-04, alpha = 2.907639e+00
    CG: it 100, r norm 9.086922290200e-05, alpha = 2.946219e+00
    BP5: N, DOFs, elapsed, iterations, time per DOF, avg BW (GB/s), avg GFLOPs, DOFs*iterations/ranks*time
    4, 68921, 0.0216, 100, 3.13e-07, 81.5, 49.8, 3.20e+08

The printed value of `r norm` at the end of 100 CG iterations should be small.

How to clean build objects
--------------------------

To clean the `benchParanumal` build objects:

    $ cd /path/to/CEED/repo
    $ make realclean

Please invoke `make help` for more supported options.
