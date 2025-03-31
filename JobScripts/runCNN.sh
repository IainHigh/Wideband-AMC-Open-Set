#!/bin/sh
#$ -N CNN-Model
#$ -wd /home/s2062378/Synthetic-Radio-Frequency-Data-Generator
#$ -l h_rt=47:59:00
#$ -l h_vmem=20G

#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 4

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

# Run the program
python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/ML-Models/main.py