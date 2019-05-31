# OptiXPrimeTemplate
A template project for OptiXPrime with CMake that just works out of the box.

## Usage

This works out of the box with CMake, granted that you have CUDA and the Nvidia OptiX SDK installed somewhere CMake can find. Just run the CMake GUI in this directory, or just `cmake .` from the command-line.

Feel free to edit the `CMakeLists.txt` file to change your project's name or its destination directory.

I also added a couple `throw` statements where your action is really needed, namely before loading vertices and indices (you need to provide them) and building rays (you need to provide an origin and a view-projection matrix).

Mail comments at mattias at refeyton dot fr. Happy coding !