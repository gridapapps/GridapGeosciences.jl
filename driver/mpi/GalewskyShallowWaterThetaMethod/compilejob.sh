#!/bin/bash
#PBS -q normal
#PBS -l walltime=00:20:00
#PBS -l ncpus=1
#PBS -l mem=6gb
#PBS -N makejl
#PBS -l wd

source modules.sh
julia --project=. -O3 --check-bounds=no --color=yes compile/compile.jl
