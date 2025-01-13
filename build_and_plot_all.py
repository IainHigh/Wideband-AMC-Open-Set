import os
import sys
import shutil

# Empty out the existing data directory
shutil.rmtree('data')
        
# Run the generator script:
# $ python3 generator.py ./configs/defaults.json
os.system('python3 generator.py ./configs/defaults.json')


# Run the plotter scripts:
    # $ cd tests; python3 plot_constellation_diagram.py; cd ..
    # $ cd tests; python3 plot_time_domain_diagram.py; cd ..
    # $ cd tests; python3 plot_frequency_domain_diagram.py; cd ..
    
os.system('cd tests; python3 plot_constellation_diagram.py; cd ..')
os.system('cd tests; python3 plot_time_domain_diagram.py; cd ..')
os.system('cd tests; python3 plot_frequency_domain_diagram.py; cd ..')