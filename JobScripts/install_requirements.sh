#!/bin/sh
#$ -N Install-Requirements
#$ -wd /home/s2062378
#$ -l h_rt=00:59:00 
#$ -l h_vmem=10G

#$ -o /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/OutputFiles
#$ -e /home/s2062378/Synthetic-Radio-Frequency-Data-Generator/OutputFiles
#$ -m beas
#$ -M "s2062378@ed.ac.uk"

export LD_LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LD_LIBRARY_PATH
export PATH=$HOME/liquid-dsp-install/bin:$PATH
export C_INCLUDE_PATH=$HOME/liquid-dsp-install/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LIBRARY_PATH

# Initialise the environment modules and cuda
. /etc/profile.d/modules.sh
module load cuda

# Load anaconda
module load anaconda

# Clear scratch space
rm -rf /exports/eddie/scratch/s2062378
mkdir /exports/eddie/scratch/s2062378/anaconda
mkdir /exports/eddie/scratch/s2062378/data
mkdir /exports/eddie/scratch/s2062378/anaconda/envs
mkdir /exports/eddie/scratch/s2062378/anaconda/pkgs

# Install the requirements
conda create -n mypython python=3.7 matplotlib numpy tqdm scipy -y
source activate mypython
conda install pip -y
pip3 install seaborn
pip3 install torch torchvision torchaudio
pip3 install SigMF==1.1.1
pip3 install scikit-learn
pip3 install adjustText

echo "Installation complete. Please check the output files for any errors."