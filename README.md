# n-body-problem

![nbody](output.gif)

Implementation of a numeric simulation of the n-body problem. Mostly meant as a playground to explore various optimization approaches like compiler intrinsics. 
Made in MSVC C++20 and needs SFML to compile. The paths for SFML within the solution are set to my own, so you'll need to change those to where the SFML headers and lib files are located on your system (info [here](https://www.sfml-dev.org/tutorials/2.6/start-vc.php)).
There is currently no way to change most settings without recompiling, will add functionality for that later.
On my system, SIMD processing runs about 1.6 times faster than the SISD approach (both using the naive O(n²) algorithm and 800 bodies).

## Controls

- mouse wheel zooms in and out
- clicking and dragging moves the camera view around
- pressing and holding L-CTRL makes zooming and dragging faster
- pressing R resets zoom to default and focuses the camera on 0,0
- pressing F toggles the display of the FPS counter, the processing type, gravity and the timestep
- pressing Numpad+/- (+ L-CTRL) increases/decreases gravity (the timestep)

## Acknowledgements

Uses [SFML](https://www.sfml-dev.org/index.php) for rendering & user input and the [Routed Gothic Typeface](https://github.com/dse/routed-gothic) by Darren Emby for texts.