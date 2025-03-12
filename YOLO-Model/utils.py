#############################################
# utils.py
#############################################

import os
import glob

def create_dataset(data_dir, rng_seed):
    """
    Removes existing training/validation/testing sets, then invokes generator.py
    to create new sets using the specified JSON config files and seeds.
    """
    for set_name in ["training", "validation", "testing"]:
        full_path = os.path.join(data_dir, set_name)
        if os.path.exists(full_path):
            for f in glob.glob(f"{full_path}/*"):
                os.remove(f)
            os.removedirs(full_path)

    # Generate the new dataset using the generator.py script.
    # Different rng seeds so each set is unique.
    os.system(f"python3 generator.py ./configs/training_set.json {rng_seed + 1}")
    os.system(f"python3 generator.py ./configs/validation_set.json {rng_seed + 2}")
    os.system(f"python3 generator.py ./configs/testing_set.json {rng_seed + 3}")