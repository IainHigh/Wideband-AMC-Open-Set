#!/bin/sh
#$ -N DatasetAnalysis
#$ -wd /home/s2062378/Synthetic-Radio-Frequency-Data-Generator
#$ -l h_rt=00:30:00 
#$ -l h_vmem=1G

#$ -o /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/OutputFiles
#$ -e /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/OutputFiles
# -m beas
# -M "s2062378@ed.ac.uk"

export LD_LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LD_LIBRARY_PATH
export PATH=$HOME/liquid-dsp-install/bin:$PATH
export C_INCLUDE_PATH=$HOME/liquid-dsp-install/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LIBRARY_PATH

# Initialise the environment modules and cuda
. /etc/profile.d/modules.sh
module load cuda

# Load anaconda
module load anaconda

# Activate the anaconda environment
source activate mypython

# Run the dataset generation program
# python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/generator.py ./configs/testing_set.json

# Run the dataset analysis program
# python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/DatasetPlotter/plot_dataset.py

# Remove the default dataset generated
# rm -rf /exports/eddie/scratch/s2062378/data

# Plot all of the modulation schemes
rm -rf /exports/eddie/scratch/s2062378/data/all_mod_schemes
python3 /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/DatasetPlotter/plot_all_constellations.py
