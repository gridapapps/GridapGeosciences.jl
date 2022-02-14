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

export OMP_NUM_THREADS={{nthreads}}
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_DISPLAY_ENV=true
mpirun -np {{ncpus}} hostname > {{{title}}}.hostfile
hosts=$(sort -u {{{title}}}.hostfile)
echo $hosts > {{{title}}}.hostfile
let nodes={{nnodes}}
let TPN={{tpn}}
i=1
rm -f {{{title}}}.machinefile
while [ $i -le $nodes ]
do
  host=`echo $hosts | sed s/" "" "*/#/g | cut -f $i -d#`
  echo "$host slots=$TPN max_slots=$TPN" >> {{{title}}}.machinefile
  let i=i+1
done

$HOME/.julia/bin/mpiexecjl --project={{{projectdir}}} --report-bindings -x OMP_NUM_THREADS  -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DISPLAY_ENV --hostfile {{{title}}}.machinefile\
 -n {{n}} \
    julia -O3 --check-bounds=no -e\
      'using GalewskyShallowWaterThetaMethod; GalewskyShallowWaterThetaMethod.main(np={{n}},numrefs={{numrefs}},nr={{nr}},dt={{dt}},write_solution=false,write_solution_freq={{write_solution_freq}},title="{{{title}}}",k={{k}},degree={{degree}},mumps_relaxation={{mrelax}},nstep={{nstep}})' > {{{title}}}.stdout

