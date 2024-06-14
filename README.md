# n-body-problem

![nbody](output.gif)

Implementation of a numeric simulation of the n-body problem. Mostly meant as a playground to explore various speed up approaches like compiler intrinsics or CUDA. 
Made in MSVC C++20 and needs SFML to compile. The paths for SFML within the solution are set to my own, so you'll need to change those to where the SFML headers and lib files are located on your system (info [here](https://www.sfml-dev.org/tutorials/2.6/start-vc.php)).
SIMD (AVX-based) and CUDA versions are included as configurations in the VS solution file. Obviously, CUDA needs to be installed for the latter to work.

Currently, using SIMD and 800 bodies, the simulation runs at ~700 FPS on my system. The release contains a binary for x64 system built with SIMD support.


## Controls

The setup provides onscreen instructions. During simulation:
- mouse wheel zooms in and out
- clicking and dragging moves the camera view around
- pressing and holding L-CTRL makes zooming and dragging faster
- pressing R resets zoom to default and focuses the camera on 0,0
- pressing F toggles the display of the FPS counter, the processing type, gravity and the timestep
- pressing Numpad+/- (+ L-CTRL) increases/decreases gravity (the timestep)
- pressing S flips gravity around
- pressing T sends you back to the setup menu

## Acknowledgements

Uses [SFML](https://www.sfml-dev.org/index.php) for rendering & user input and the [Routed Gothic Typeface](https://github.com/dse/routed-gothic) by Darren Emby for texts.