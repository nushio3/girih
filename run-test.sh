OMP_NUM_THREADS=16 mpiexec -n 2 ./build/mwd_kernel --npy 2 --ny 1024 --nx 512 --nz 512 --nt 500 --target-kernel 1 --mwd-type 2 --target-ts 2
