#!/bin/bash

# Set path for local install
PATH=$PATH:/miniconda2/bin;

# Create new env
conda create -n pyceratOpsRecs python=2.7;
source activate pyceratOpsRecs;

# run conda updates
conda update -y conda;

# Install deps.
conda install -q -y -c conda-forge tensorflow;