#!/bin/sh
#$ -N YOLO
#$ -wd /home/s2062378/Synthetic-Radio-Frequency-Data-Generator
#$ -l h_rt=00:59:00
#$ -l h_vmem=20G

#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 1

#$ -o /home/s2062378/OutputFiles
#$ -e /home/s2062378/OutputFiles
#$ -m beas
#$ -M "s2062378@ed.ac.uk"

export LD_LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LD_LIBRARY_PATH
export PATH=$HOME/liquid-dsp-install/bin:$PATH
export C_INCLUDE_PATH=$HOME/liquid-dsp-install/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LIBRARY_PATH

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load cuda

# Load anaconda
module load anaconda

# Activate the anaconda environment
source activate mypython

if false ; then
    # Remove the old dataset
    rm -rf /exports/eddie/scratch/s2062378/data/*
    rmdir /exports/eddie/scratch/s2062378/data/training
    rmdir /exports/eddie/scratch/s2062378/data/validation
    rmdir /exports/eddie/scratch/s2062378/data/testing

    # Create a new dataset
    python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/generator.py /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/configs/training_set.json 2025
    python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/generator.py /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/configs/validation_set.json 2026
    python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/generator.py /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/configs/testing_set.json 2027
fi

# Run the program
python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/YOLO-Model/main.py