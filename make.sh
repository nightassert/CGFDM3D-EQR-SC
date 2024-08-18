Method=${1}

make clean -f Makefile_${Method}
make -j -f Makefile_${Method}
