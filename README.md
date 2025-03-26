# STORM2.0
An updated version of STORM by NBloemendaal (https://github.com/NBloemendaal/STORM and https://github.com/NBloemendaal/STORM-preprocessing)
See https://doi.org/10.1038/s41597-020-0381-2 for details

This version combines two repositories, STORM_preprocessing and STORM, to create one dual module repository.

STORM_preprocessing:
Conducts the data preprocessing necessary for STORM.

STORM:
Generates synthetic cyclone tracks.

A user guide for the model is in progress!

Summary of the changes compared to the original version:

- More user friendly (more comments, better able to tweak some things from just the master script, generated files separated into folders by type, all necessary scripts called from the master files)
- Added datetime compatibility (the storm tracks have datetime timestamps instead of just month and 3-hour timestep number)
- Changed basin boundaries for the North Atlantic (Eastern boundary now 30 degrees instead of 0) to allow for European impacts, and the South Indian/South Pacific border (from 135 E to 105 E) to stop splitting Australia in half. All basins also extended to 90 degrees north/south from 60.
- Wind threshold updated to match 10-minute threshold.
- Fixed an issue where some tracks were being double-called when being written leading to storms that would complete their track, jump back to the beginning, and repeat their track.
- Implemented haversine function when determining distance to land, no longer just using pythagorean (flat plane) distance
  
