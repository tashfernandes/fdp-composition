# f-Differential Privacy - Composition Results

This code accompanies the paper "Composition Theorems for f-Differential Privacy".

It includes all code to generate the figures from the paper.

## Getting Started

An artifact for evaluation has been provided as a Docker image.
Note that the image has been built for linux/amd64 but should run fine on Windows or MacOS. 
The artifact was tested on MacOS with ARM processor.

Alternatively, to run the code locally you'll need Python3 and pip.
Required packages are specified in requirements.txt.

### Quick start - Install via Docker

1. Download the artifact here: https://doi.org/10.5281/zenodo.18185391 and extract it

   `gunzip fdp-artifact_1.0.tar.gz`

2. Load the image into Docker

   `docker load -i fdp-artifact_1.0.tar`

3. Run the container (NOTE: this will create a figures/ directory on the local machine)

- For Windows (Powershell)

    `docker run --rm -v "${PWD}\figures:/artifact/figures" fdp-artifact:1.0`

- For Linux/MacOS

    `docker run --rm -v "$(pwd)/figures:/artifact/figures" fdp-artifact:1.0`

Generated figures will be written to the `figures/` directory.

## Step-By-Step Instructions

The following instructions are for a local install of the code for custom testing
of the supplied methods.

### Local install - Instructions for MacOS / Linux

You need to have Python3 already installed; in addition, the plotting
code uses latex to render labels, so you also need some texlive packages.

- On Linux this can be accomplished with:

   `apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    cm-super \
    dvipng`

- On MacOS, install the MacTex distribution from here: https://www.tug.org/mactex/

Once these are installed, you can run the code using the following:

1. Create a virtual environment and activate

   `python3 -m venv env; source env/bin/activate`

2. Install required packages

   `pip install -r requirements.txt`

3. Run code to generate the figures

   `python3 scripts/generate_plots.py`

This will generate the figures from the paper.

NOTE: You may need to set your python path to find the python modules included in src.

    `export PYTHONPATH=.:$PYTHONPATH`

## Code structure

- `src/qif.py` — QIF methods supporting computations on channels
- `src/core.py` - Methods supporting f-differential privacy.
- `src/alg.py` — Algorithms from the paper.
- `src/graphs.py` — Code for plotting each of the figures.
- `scripts/generate_plots.py` — Configure and generate Figures 2–9.

## Instructions for modifying code

Most of the configurable variables are in scripts/generate_plots.py.
eg. epsilon, delta, alpha values. These can be modified for testing.
There is also a show_plot() function that can be used in place of
save_plot() to display the graph using matplotlib.

Useful helper functions are included in src/core.py and src/alg.py.
These support the main operations of building channels and converting
these to trade-off functions for graphing.
Examples of how to use these can be found in src/graphs.py.

QIF helper functions can be found in src/qif.py. These are used to
implement basic QIF functionality used in the paper, such as computing
prior and posterior vulnerabilities, and building Geometric and random
response mechanisms. Examples of how to use these can be found in 
src/graphs.py

