Method=${1}

mkdir -p bin obj

make clean -f Makefile_${Method}
make -j -f Makefile_${Method}
