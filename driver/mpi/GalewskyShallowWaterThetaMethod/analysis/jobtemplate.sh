#!/bin/bash
#PBS -P bt62
#PBS -q {{q}} 
#PBS -l walltime={{walltime}}
#PBS -l ncpus={{ncpus}}
#PBS -l jobfs=40GB
#PBS -l mem={{mem}}
#PBS -N {{{name}}}
#PBS -l wd
#PBS -o {{{o}}}
#PBS -e {{{e}}} 
#PBS -l software=GridapGeosciences.jl

PERIOD=1
top -b -d $PERIOD -u am6349 > {{{title}}}.log &

source {{{modules}}}

$HOME/.julia/bin/mpiexecjl --project={{{projectdir}}} -n {{n}}\
    julia -J {{{sysimage}}} -O3 --check-bounds=no -e\
      'using GalewskyShallowWaterThetaMethod; GalewskyShallowWaterThetaMethod.main(np={{n}},numrefs={{numrefs}},nr={{nr}},dt={{dt}},write_solution=false,write_solution_freq={{write_solution_freq}},title="{{{title}}}",k={{k}},degree={{degree}},mumps_relaxation={{mumps_relaxation}},nstep={{nstep}})' > {{{title}}}.stdout

