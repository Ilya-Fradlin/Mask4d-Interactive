#!/bin/bash

# Install MinkowskiEngine on GPU node
cd third_party/MinkowskiEngine 
pip install .
cd ../..

# run the main training script
python juwels_test.py
