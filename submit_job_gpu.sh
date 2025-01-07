#!/bin/bash

# Load the email from the .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found!"
    exit 1
fi

# Replace the placeholder in the template and create the job script
sed -e "s|{{EMAIL}}|$EMAIL|g" jobfile_gpu.sh > jobfile.sh

# Suppress both stdout and stderr
#BSUB -o /dev/null
#BSUB -e /dev/null

# Submit the generated job script
bsub < jobfile.sh
