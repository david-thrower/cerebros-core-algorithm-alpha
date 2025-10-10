# Run in home folder ... 
mkdir 2025-10-10--more-optimizations-br-254-single-machine-jit
cd 2025-10-10--more-optimizations-br-254-single-machine-jit
git clone https://github.com/david-thrower/cerebros-core-algorithm-alpha.git
cd cerebros-core-algorithm-alpha
git checkout 254-more-optimizations-to-notgpt-hpo-script
git pull origin 254-more-optimizations-to-notgpt-hpo-script
screen -S hpo_study_40_850_jit -dm bash -c "python3 generative-proof-of-concept-CPU-preprocessing-in-memory.py &> 2025-10-10--more-optimizations-br-254-single-machine-jit.txt; exit"
