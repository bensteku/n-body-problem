# n-body-problem

![nbody](output.gif)

Implementation of a numeric simulation of the n-body problem. Mostly meant as a playground to explore various speed up approaches like compiler intrinsics or CUDA. 
Made in MSVC C++20 and needs SFML to compile. The paths for SFML within the solution are set to my own, so you'll need to change those to where the SFML headers and lib files are located on your system (info [here](https://www.sfml-dev.org/tutorials/2.6/start-vc.php)).
SIMD (AVX-based) and CUDA versions are included as configurations in the VS solution file. Obviously, CUDA needs to be installed for the latter to work. The release to the side is a binary for x64 systems with AVX support.

Small FPS benchmark on my system (Ryzen 5950x, RTX3080Ti):

| num. bodies | SISD | SIMD | CUDA |
|:-------------:|:------:|:------:|:------:|
| 80          |   4900   |   4800   |   2700   |
| 800         |   460   |   790   |   870   |
| 8000        |   8   |   18   |    92  |

| num. bodies @ 30 FPS | SISD | SIMD | CUDA |
|:-------------:|:------:|:------:|:------:|
| -         |   3400   |   5900   |   19500   |



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