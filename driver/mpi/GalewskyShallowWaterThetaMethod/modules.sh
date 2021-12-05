# .bashrc
module load pbs
module load openmpi/4.1.0
module load intel-mkl/2020.0.166
module load python3-as-python
export HCOLL_ML_DISABLE_SCATTERV=1; export HCOLL_ML_DISABLE_BCAST=1
MKLDIRPATH=$MKLROOT/lib/intel64_lin/
#export LD_PRELOAD=$MKLDIRPATH/libmkl_avx2.so:$MKLDIRPATH/libmkl_def.so:$MKLDIRPATH/libmkl_sequential.so:$MKLDIRPATH/libmkl_core.so
export PATH=~/julia-1.6.4/bin:$PATH
alias cd_scratch="cd /scratch/bt62/am6349/"
alias interactive="qsub -I -X -lwalltime=1:00:00,mem=180GB,ncpus=48,jobfs=1GB"
export GRIDAP_PARDISO_LIBGOMP_DIR=/half-root/lib/gcc/x86_64-redhat-linux/8/
export P4EST_ROOT_DIR=/home/565/am6349/p4est-install/
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export JULIA_MPI_BINARY="system"
export JULIA_MPI_PATH=/apps/openmpi/4.1.0
export JULIA_PETSC_LIBRARY="/home/565/am6349/petsc-install/lib/libpetsc"
export PATH=$PATH:/home/565/am6349/.julia/bin
