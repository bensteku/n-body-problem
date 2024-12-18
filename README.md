# n-body-problem

![nbody](output.gif)

Implementation of a numeric simulation of the n-body problem. Mostly meant as a playground to explore various speed up approaches like compiler intrinsics or CUDA. 
Build it with CMake. Currently, there are 3 build types, all for MSVC 20: SISD, SIMD and CUDA. All three need the SFML library, so make sure that the SFML_DIR env variable is set correctly (or edit the CMakeLists.txt with your personal path). The CUDA variant obviously also needs the CUDA toolkit to be installed on your PC.
The release version is the SIMD variant built for Windows x64 systems.

Small FPS benchmark on my system (Ryzen 5950x, RTX3080Ti):

| num. bodies | SISD <br> (multi-threaded) | SIMD | CUDA |
|:-------------:|:------:|:------:|:------:|
| 80          |   4900 <br> (3300)  |   4800   |   2700   |
| 800         |   460 <br> (530)  |   790   |   870   |
| 8000        |   8 <br> (20)  |   18   |    92  |

| num. bodies @ 30 FPS | SISD | SIMD | CUDA |
|:-------------:|:------:|:------:|:------:|
| -         |   3400 <br> (7000)  |   5900   |   19500   |



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