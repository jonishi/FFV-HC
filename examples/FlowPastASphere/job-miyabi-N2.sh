#PBS -q special-g
#PBS -l select=2:mpiprocs=1
#PBS -l walltime=00:10:00
#PBS -W group_list=gz06
#PBS -j oe

cd ${PBS_O_WORKDIR}
mpirun -np 2 ./ffv sphere.tpp

