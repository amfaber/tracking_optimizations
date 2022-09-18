#conda install -c conda-forge graphviz gprof2dot -y
script=$1
shift
python -m cProfile -o profile.pstats $script $@
gprof2dot -f pstats profile.pstats | dot -Tpng -o profile.png