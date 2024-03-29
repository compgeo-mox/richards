# Numerical simulations of the water table evolution in a Darcy-Richards system
This repository contains the code developed for my master thesis, titled 'Numerical simulations of the water table evolution in a Darcy-Richards system.'


### Installation
To install the project, we advise you to install and run the project through Docker (the Dockerfile of the project is available in the directory .devcontainer).
The steps that must be performed to install the project with Docker are:
1. Clone the repository:
```bash
git clone https://github.com/compgeo-mox/richards --recurse-submodules
```
or
```bash
git clone git@github.com:compgeo-mox/richards.git --recurse-submodules
```
2. Enter in the repository directory and build the Docker image:
```bash
docker build . -t richards -f .devcontainer/Dockerfile
```
3. Run the container and enter the shell:
```bash
docker run --name nn --rm -v $(pwd):/richards -it richards
```

Notice that the directory containing the project itself must be included in the Python PATH for the code to work. 
This process is performed automatically by Docker.

In the case you prefer to run the code without using Docker, the installation steps are:

1. Clone the repository:
```bash
git clone https://github.com/compgeo-mox/richards --recurse-submodules
```
or
```bash
git clone git@github.com:compgeo-mox/richards.git --recurse-submodules
```
2. Follow the installation instructions contained in the directories 'porepy' and 'pygeon', and install the two libraries
3. Install the following pip packages:
```bash
python3 -m pip install cython numba shapely ipython ipykernel jupyterlab
```
4. Add to the Python PATH porepy, pygeon, and the directory containing this project.

The installing procedure is greatly simplified if Visual Studio Code is employed.
In fact, after cloning the directory, Visual Studio Code will automatically detect the Dockerfile when opening the project and ask the user to run the project inside a developing container.
In this way, the only requirement is to clone the repository.

### Code Structure

The code is divided into several different directories.
In particular:
- .devcontainer: It contains the Dockerfile that can be employed to run all the simulations
- final_tests: It's the directory containing the tests that made an appearance in the thesis
- matlab_scripts: It contains some helper Matlab scripts that can be used to post-process the obtained solution
- plotters: It's the directory containing the notebooks used to generate the plots contained in the thesis
- porepy and pygeon: Those are the two main dependencies of this project. They can be installed by following their installation instructions (the Dockerfile automatically installs them and all the required dependencies)
- richards: It's the directory containing part of the library that we developed to solve Richards's problem
- utilities: The directory contains a collection of simple modules the library mentioned above uses to solve the different problems.
