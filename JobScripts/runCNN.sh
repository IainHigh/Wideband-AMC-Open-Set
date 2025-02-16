#!/bin/sh
export LD_LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LD_LIBRARY_PATH
export PATH=$HOME/liquid-dsp-install/bin:$PATH
export C_INCLUDE_PATH=$HOME/liquid-dsp-install/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LIBRARY_PATH

#$ -N CNN-Model         
#$ -wd /home/s2062378/Synthetic-Radio-Frequency-Data-Generator
#$ -q gpu
#$ -pe gpu-a100 1           
#$ -l h_rt=47:59:00
#$ -l h_vmem=20G 

#$ -o /home/s2062378/OutputFiles
#$ -e /home/s2062378/OutputFiles
#$ -m beas
#$ -M "s2062378@ed.ac.uk"
#$ -R y

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load cuda

# Load anaconda
module load anaconda

# Activate the anaconda environment
source activate mypython

# Run the program
python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/ML-Models/main.py