#!/bin/bash

                                                         # Set the working directory of the batch 
                                                         # script before it is executed.
#SBATCH --job-name=m1e12b0Job                               # Job name
#SBATCH --mail-type=ALL                                  # Valid event values are: BEGIN, END, FAIL, REQUEUE, 
                                                         # ALL (equivalent to BEGIN, END, FAIL, REQUEUE, and STAGE_OUT), 
                                                         # STAGE_OUT (burst buffer stage out completed), 
                                                         # TIME_LIMIT, TIME_LIMIT_90 (reached 90 percent of time limit), 
                                                         # TIME_LIMIT_80 (reached 80 percent of time limit), 
                                                         # and TIME_LIMIT_50 (reached 50 percent of time limit). 
#SBATCH --mail-user=22226851@student.uwa.edu.au          # Where to send mail.  Set this to your email address
#SBATCH --ntasks-per-node=32                             # Number of processes/threads per node
#SBATCH --nodes=1                                        # Maximum number of nodes to be allocated
#SBATCH --mem-per-cpu=4GB                                # Memory (i.e. RAM) per CPU
#SBATCH --time=2-00:00:00                                # Wall time limit (days-hrs:min:sec)
#SBATCH --output=m1e12b0.log              # Path to the standard output and error files 
                                                         # relative to the working directory

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Script Directory  = $(pwd)"
echo ""

nodelist=$SLURM_NODELIST
numnodes=$SLURM_NNODES
numtasks=$SLURM_NTASKS
numprocs=$SLURM_NPROCS
cpuspertask=$SLURM_CPUS_PER_TASK
mempernode=$SLURM_MEM_PER_NODE
mempercpu=$SLURM_MEM_PER_CPU

echo "Nodes Allocated                = $nodelist"
echo "Number of Nodes Allocated      = $numnodes"
echo "Number of Tasks Allocated      = $numtasks"
echo "Number of Processes Allocated  = $numprocs"
echo "Number of Cores/Task Allocated = $cpuspertask"
echo "Memory per Node                = $mempernode"
echo "Memory per CPU                 = $mempercpu"
echo ""

# rundir=/fred/oz009/jwinstan/cleanisolated/m1e12b0_2r200
rundir=${PWD}
cd "$rundir" || exit -1
echo "Run Directory  = $(pwd)"
export OMP_NUM_THREADS=$cpuspertask
module load gcc/12.2.0 openmpi/4.1.4 fftw/3.3.10 hdf5/1.14.0 gsl/2.7 

srun ../../Gadget4 ./param.txt
