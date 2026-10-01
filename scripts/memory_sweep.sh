#!/bin/bash
# Memory sweep for the mpi_memory_profile example.
# Usage: scripts/memory_sweep.sh <gismo build dir> > sweep.log
#        python3 scripts/memory_tables.py sweep.log
cd "${1:-.}" || exit 1
R="mpirun --oversubscribe"
[ "$(id -u)" = 0 ] && R="$R --allow-run-as-root"
run() { echo "### $*"; $R "$@" --legacy --csv 2>&1 | grep -v "^#" ; }
# Fixed N, growing P (2D ~1M dofs, 3D ~275k dofs)
for P in 1 2 4 8; do run -np $P ./bin/mpi_memory_profile -d 2 -r 9; done
for P in 1 2 4 8; do run -np $P ./bin/mpi_memory_profile -d 3 -r 5; done
# Fixed P, growing N
for r in 6 7 8 10; do run -np 4 ./bin/mpi_memory_profile -d 2 -r $r; done
for r in 3 4; do run -np 4 ./bin/mpi_memory_profile -d 3 -r $r; done
# Variants at P=4: refined geometry, many patches, higher degree, no fiber reservation
run -np 4 ./bin/mpi_memory_profile -d 2 -r 9 --geo
run -np 4 ./bin/mpi_memory_profile -d 2 -s 5 -r 5
run -np 4 ./bin/mpi_memory_profile -d 2 -s 5 -r 5 --geo
run -np 4 ./bin/mpi_memory_profile -d 2 -r 8 -p 4
for P in 1 4; do run -np $P ./bin/mpi_memory_profile -d 2 -r 9 --noreserve; done
run -np 4 ./bin/mpi_memory_profile -d 3 -r 5 --noreserve
