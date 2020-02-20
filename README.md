# benchparanumal
Standalone benchmarks related to libparanumal capabilities

To clone, build, and run:

```
 mkdir Tmp
 cd Tmp/
 git clone https://github.com/paranumal/benchparanumal
 git clone https://github.com/libocca/occa
 cd occa/
 export OCCA_DIR=`pwd`
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OCCA_DIR/lib
 make -j
 cd ../benchparanumal/CEED/BP/occa
 make -j
 ./BP setups/setupBP1.rc
```

The final output should be something like (disregard the Bandwidth estimates as they seem too high for the NVIDIA Titan V I ran on just now):

```
Warning: Failed to find [CUBATURE DEGREE].
Failed to find [CUBATURE DEGREE].
Failed to find [CUBATURE DEGREE].
TARGET NODES = 3000000, ACTUAL NODES = 2999808, NELEMENTS = [31,27,7=>5859]
Rank 0 initially has 5859 elements
r: 00 [ 0000] (Nelements=5859, Nmessages=0, Ncomms=0)
J in range [0.000170678,0.000170678] and max Skew = 4.42857
Local nodes=2999808, Localized nodes=2009637
device_id = 0
mode: 'CUDA', device_id: 0
BP0:10=1,0,0,0,0,0,0,0
after occa setup: bytes allocated = 1900591540
ENTERING BPSOLVESETUP
allNeumann = 0 
Compiling GatherScatter Kernels...done.
ENTERING KERNEL BUILDS
useGlobal=0
BUILDING LOCAL STORAGE KERNELS
Loaded: BP1_v0 from /home/tcew/Tmp/benchparanumal/CEED/BP/occa/okl/BP1.okl
after BP setup: bytes allocated = 2308992448
Elapsed: overall: 0.0820965, PCG Update 0.0222753, Pupdate: 0.0094207, Copy: 0.00692614, dot: 0.0105949, op: 0.031517
Bandwidth (GB/s): PCG update: 550.529, Copy: 252.938, Op: 667.027, Dot: 496.054, Pupdate: 557.884
CG: 1.563257e-04, 1.918947e+01 ; % (OP(x): elapsed, GNodes/s)
Elapsed: overall: 0.0811244, PCG Update 0.0220635, Pupdate: 0.00918595, Copy: 0.00671158, dot: 0.0103675, op: 0.0314526
Bandwidth (GB/s): PCG update: 555.815, Copy: 261.025, Op: 668.391, Dot: 506.939, Pupdate: 572.141
CG: 1.553473e-04, 1.931033e+01 ; % (OP(x): elapsed, GNodes/s)
elapsed = 0.081916, globalElapsed = 0.081916, globalNelements = 5859
7, 5859, 2999808, 2009637, 0.0819158, 73, 5.58376e-10, 1.79091e+09, 331.323, 0, 0, 1.29364e+10, 1; % global: N, Nelements, dofs, globalDofs, elapsed, iterations, time per global node, fields*global nodes*iterations/time, BW GFLOPS/s, kernel Id, combineDot, fields*nodes*iterations/opElapsed, BPid
globalMaxError = 2.9811e-05
```

You may need to adjust the DEVICE, PLATFORM, MODEL in the setups/setupBP1.rc file to match your system. Currently it is set to use CUDA on DEVICE 0 as indicated below.

```
[THREAD MODEL]
CUDA
#HIP
#OpenCL

[PLATFORM NUMBER]
0

[DEVICE NUMBER]
0
```
