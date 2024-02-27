# Tell the system to call the sh command line interpreter (shell) for interpreting the subsequent lines
#!/bin/sh

### General options 
# The lines starting with #BSUB are interpreted by the Resource Manager (RM) as lines that contain options for the RM
### -- specify queue --   # different queues have different defaults 
#BSUB -q gpua100

### -- set the job Name --  # to easily check the status of your job 
#BSUB -J ThesisApp

### -- ask for number of cores (default: 1) --  # ask to reserve n cores (processors). The number is the total number of cores, that could be on one or more than one node
#BSUB -n 1

### -- Select the resources: 1 gpu in exclusive process mode --
### GPU: Always request the GPU with the clause mode=exclusive_process
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that we need xGB of memory per core/slot --  # means that your job will be run on a machine that has AT LEAST 4GB per core (slot) of memory available. So in our case with -n 4 and -R "span[hosts=1], the job will be dispatched to a machine with at least 16 GB or RAM available.
#BSUB -R "rusage[mem=350GB]"

#BSUB -R "select[gpu80gb]"

### -- set walltime limit: hh:mm --  # specifies that you want your job to run AT MOST hh:mm 
#BSUB -W 24:00 

### -- set the email address -- 
# please uncomment the following line and put in your e-mail address if you want to receive e-mail notifications on a non-default address
#BSUB -u s210289@dtu.dk

### -- send notification at start -- 
#BSUB -B 

### -- send notification at completion -- 
#BSUB -N 

### -- send notification if the job fails
### #BSUB -Ne

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo Output_%J.out 
#BSUB -ee Output_%J.err 

nvidia-smi
# Load the cuda module
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

# Load modules needed by myapplication.x
module load python3/3.10.7
. .venv/bin/activate

# here follow the commands you want to execute with input.in as the input file
#python3 -m cProfile -s 'cumulative' -o prof_gpu_new train.py #input.in > output.out
python3 pred_noaug_comb.py
