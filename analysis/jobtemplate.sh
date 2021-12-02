#!/bin/bash
#PBS -P bt62
#PBS -q {{q}} 
#PBS -l walltime={{walltime}}
#PBS -l ncpus={{ncpus}}
#PBS -l mem={{mem}}
#PBS -N {{{name}}}
#PBS -l wd
#PBS -o {{{o}}}
#PBS -e {{{e}}} 
#PBS -l software=GridapGeosciences.jl

PERIOD=0.1
top -b -d $PERIOD -u am6349 > {{{title}}}.log &

source {{{modules}}}

$HOME/.julia/bin/mpiexecjl --project={{{projectdir}}} -n {{n}}\
    julia -J {{{sysimage}}} -O3 --check-bounds=no -e\
      'using GalewskyShallowWaterThetaMethod; GalewskyShallowWaterThetaMethod.main(np={{n}},numrefs={{numrefs}},dt={{dt}},write_solution={{write_solution}},write_solution_freq={{write_solution_freq}},title="{{{title}}}",k={{k}},degree={{degree}},mumps_relaxation={{mumps_relaxation}})'

