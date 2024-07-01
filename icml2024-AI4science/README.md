# Quantum Code Discovery

This project contains the code to reproduce the results relating to the paper "Scaling Automated Quantum Error Correction Discovery with Reinforcement Learning" submitted to the AI4Science workshop of ICML 2024.

Use the `requirements.txt` to install all necessary libraries.

Use `run_d3.sh`, `run_d4.sh`, `run_d5_0.sh`, `run_d5_1.sh` file to run all the experiments appearing in Table 1 in the paper.

Inside the `notebooks` folder there are two notebooks: one for the simulation time comparison between our framework and Stim called `ours_vs_stim.ipynb`, and another one to analyze results generated with the previous scripts called `results.ipynb`.

* The training and testing files are `experiments_d3.py`, `experiments_d4.py` and `experiments_d5.py`.
* Results of all training (circuits, plot, etc.) is in the `results/` folder.