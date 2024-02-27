# Tell the system to call the sh command line interpreter (shell) for interpreting the subsequent lines
#!/bin/sh

### General options 
# The lines starting with #BSUB are interpreted by the Resource Manager (RM) as lines that contain options for the RM
### -- specify queue --   # different queues have different defaults 
#BSUB -q hpc

### -- set the job Name --  # to easily check the status of your job 
#BSUB -J ThesisApp

### -- ask for number of cores (default: 1) --  # ask to reserve n cores (processors). The number is the total number of cores, that could be on one or more than one node
#BSUB -n 4

### -- specify that the cores must be on the same host --  # specify how the users wants the cores to be distributed across nodes. Therefore it is not possible to request more cores than the number of physical cores present on a machine.
#BSUB -R "span[hosts=1]"

### -- specify that we need 4GB of memory per core/slot --  # means that your job will be run on a machine that has AT LEAST 4GB per core (slot) of memory available. So in our case with -n 4 and -R "span[hosts=1], the job will be dispatched to a machine with at least 16 GB or RAM available.
#BSUB -R "rusage[mem=250GB]"

### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --  # specifies the per-process memory limit for all the process of our job. In our case, with -n 4 and -R "span[hosts=1], the job will be killed when it exceeds 20 GB of RAM.
###BSUB -M 128GB   ### commented or same as rusage

### -- set walltime limit: hh:mm --  # specifies that you want your job to run AT MOST hh:mm 
#BSUB -W 24:00 

### -- set the email address -- 
# please uncomment the following line and put in your e-mail address if you want to receive e-mail notifications on a non-default address
#BSUB -u s210289@dtu.dk

### -- send notification at start -- 
###BSUB -B 

### -- send notification at completion -- 
#BSUB -N 

### -- send notification if the job fails
### #BSUB -Ne

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo Output_%J.out 
#BSUB -ee Output_%J.err 

# Load modules needed by myapplication.x
module load python3/3.10.7
. .venv/bin/activate

# here follow the commands you want to execute with input.in as the input file
#mprof run --multiprocess -o memory_profile2.dat python3 -m cProfile -s 'cumulative' -o prof try2.py #input.in > output.out
python3 train64.py