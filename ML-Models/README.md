This is the "omnipotent" center_frequency CNN pipeline approach discussed in weekly report 5 (and partly 6). It works by using three seperate steps:
1) Read the center_frequency from the metadata file (omnipotent step).
2) Filter the signal to receive the baseband and filter out all other signals other than the one identified in the step before.
3) Use narrowband AMC to categorise that narrowband signal.

The CNNs sub-directory contains all narrowband CNN models designed and implemented for narrowband AMC.

This will no longer be worked on, as opposed to the pipeline approach, we are instead interested in a single over-arching ML approach for detecting transmissions and categorising in a single step (using methods such as computer vision single-pass detection e.g. YOLO approach).