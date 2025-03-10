#!/bin/sh
export LD_LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LD_LIBRARY_PATH
export PATH=$HOME/liquid-dsp-install/bin:$PATH
export C_INCLUDE_PATH=$HOME/liquid-dsp-install/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LIBRARY_PATH

#$ -N DatasetAnalysis
#$ -wd /home/s2062378/Synthetic-Radio-Frequency-Data-Generator
#$ -l h_rt=00:59:00 
#$ -l h_vmem=10G
#$ -o /home/s2062378/OutputFiles
#$ -e /home/s2062378/OutputFiles
#$ -m beas
#$ -M "s2062378@ed.ac.uk"

# Initialise the environment modules and cuda
. /etc/profile.d/modules.sh
module load cuda

# Load anaconda
module load anaconda

# Activate the anaconda environment
source activate mypython

# Run the dataset generation program
python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/generator.py ./configs/BER_Tests.json

# Run the dataset analysis program
# python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/tests/plot_dataset.py

# Remove the default dataset generated
rm -rf "/exports/eddie/scratch/s2062378/data/default"