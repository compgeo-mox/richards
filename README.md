# Numerical simulations of the water table evolution in a Darcy-Richards system
This repository contains the code developed throught for my master thesis, titled 'Numerical simulations of the water table evolution in a Darcy-Richards system'.


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

Notice that, in order for the code to work, the directory containing the project itself must be included in the Python PATH. 
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
3. Add to the Python PATH porepy, pygeon, and the directory containing this project.

The installing procedure is greatly simplified if Visual Studio Code is employed.
In fact, after having cloned the directory, Visual Studio Code will automatically detect when opening the project, the Dockerfile and ask the user to run the project inside a develop container.
In this way, the only requirement is to clone the repository.

### Code Structure

The code is divided in several different directories.
In particular:
- .devcontainer: It contains the Dockerfile that can be employed to run all the simulations
- final_tests: It's directory containing the tests that made an appearance in the thesis
- matlab_scripts: It contains some helper Matlab scripts that can be used to post-process the obtained solution
- plotters: It's the directory containing the notebooks used to generate the plots contained in the thesis
- porepy and pygeon: Those are the two main dependencies of this project. They can be installed by following their own installation instructions (the Dockerfile automatically installs them and all the required dependencies)
- richards: It's the directory containing part of the library that we developed to solve Richards's problem
- utilities: It's the directory containing a collection of simple modules used by the aformentioned library to solve the different problems.
