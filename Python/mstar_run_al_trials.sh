# Script for running 10 trials of active learning on MSTAR data.
# Run this script through command line (terminal).
# To make this script executable, first run chmod 700 mstar_run_al_trials.sh 

python mstar_run_al.py --iters 500 --M 300 --seed 1
python mstar_run_al.py --iters 500 --M 300 --seed 2
python mstar_run_al.py --iters 500 --M 300 --seed 3
python mstar_run_al.py --iters 500 --M 300 --seed 4
python mstar_run_al.py --iters 500 --M 300 --seed 5
python mstar_run_al.py --iters 500 --M 300 --seed 6
python mstar_run_al.py --iters 500 --M 300 --seed 7
python mstar_run_al.py --iters 500 --M 300 --seed 8
python mstar_run_al.py --iters 500 --M 300 --seed 9
python mstar_run_al.py --iters 500 --M 300 --seed 10
