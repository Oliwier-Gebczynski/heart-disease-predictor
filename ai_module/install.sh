#!/bin/bash

# Aktualizacja listy pakietów
sudo apt-get update

# CUDA-toolkit
sudo apt install nvidia-cuda-toolkit

# Instalacja Pythona 3 i pip (jeśli nie są zainstalowane)
sudo apt-get install -y python3 python3-pip

# Instalacja wymaganych bibliotek Pythona
pip3 install numpy scikit-learn imbalanced-learn joblib tqdm dask dask-ml

# CUDA nvidia
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==24.12.*" "dask-cudf-cu12==24.12.*" "cuml-cu12==24.12.*" \
    "cugraph-cu12==24.12.*" "nx-cugraph-cu12==24.12.*" "cuspatial-cu12==24.12.*" \
    "cuproj-cu12==24.12.*" "cuxfilter-cu12==24.12.*" "cucim-cu12==24.12.*" \
    "pylibraft-cu12==24.12.*" "raft-dask-cu12==24.12.*" "cuvs-cu12==24.12.*" \
    "nx-cugraph-cu12==24.12.*"

# Potwierdzenie zakończenia instalacji
echo "All required libraries have been installed."