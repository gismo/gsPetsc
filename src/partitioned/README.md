# Partitioned objects for MPI assembly with G+Smo

This folder holds the building blocks for element-partitioned assembly
with `gsExprAssembler`, and the measurements that show which G+Smo objects
stop scaling when every rank holds them in full.

| File | Content |
|---|---|
| `gsElementRangeDomain.h` | A `gsDomain` that is a contiguous range `[first,last)` of the elements of another domain (e.g. `gsMultiBasis::domain()`), plus a block partition helper. Element ids are local, so OpenMP chunking in `gsDomain::allElements()` still works. Pass it to `gsExprAssembler::setIntegrationDomain()`. |
| `gsPetscCOO.h` | Rank-local assembly results → distributed PETSc `Mat`/`Vec` (COO interface, off-rank rows are summed by PETSc), plus gathers of selected entries (local + ghost) or of the full vector. |
| `gsPetscSink.h` | Assembler sinks that write directly into PETSc objects. `gsPetscPatternSink` collects the sparsity pattern in `computePattern_into` and preallocates an AIJ matrix exactly from it. `gsPetscSystemSink` receives `assemble_into` through `MatSetValues`, with per-thread buffers that are flushed under an OpenMP critical section. |
| `gsMemoryProbe.h` | Heap probes (`mallinfo2`), peak RSS, size estimators for G+Smo objects, and a ledger that reduces min/max/sum over ranks. |
| `../../examples/mpi_memory_profile.cpp` | Toy problem: Poisson on a grid of B-spline patches, element-partitioned assembly, PETSc KSP solve, error check. |
| `../../scripts/memory_sweep.sh`, `memory_tables.py` | The sweep used below and the table generator. |

```
mpirun -np 4 ./bin/mpi_memory_profile -d 2 -r 9 [--legacy] [--noreserve] [--geo] [-s 5] [--csv]
              [--partition block|rcb|hilbert|morton] [--local [--rendezvous]]
              [--serial-partition] [--check-partition]
              [--sink] [--sparse-mapper [--check-mapper]]
```

`--legacy` excludes `--sink` and `--local`, `--rendezvous` needs `--local`, and `--check-mapper`
needs `--sparse-mapper`. `--serial-partition` and `--check-partition` need a geometric `--partition`.

### Ledger output (`--csv`)

One line per measurement, written by rank 0, values reduced over ranks:

```
CSV,P,d,p,N,patches,geo,noreserve,v=<variant>,<name>,<kind>,<min>,<max>,<sum>
```

`<name>` may contain commas: it is everything between the `v=` field and the last four fields.

| kind | value | unit |
|---|---|---|
| `stage` | heap the stage still holds at its end (`mallinfo2` delta) | B |
| `time` | wall time of the stage (barrier to barrier; one value in all three fields) | µs |
| `rss` | `VmRSS` at the end of the stage | B |
| `peakrss` | `VmHWM` over the stage; it is reset after every stage, so transient allocations count | B |
| `object` | size estimate of an object, or a start-up figure (`VmRSS` after `MPI_Init`, and `VmRSS` with its `RssAnon` / `RssFile` / `RssShmem` parts after `PetscInitialize`) | B |
| `count` | a per-rank number: local elements, owned rows, local + ghost dofs, nonzeros, off-rank entries | – |

The last line `peak RSS (whole run),peakrss,...` is the run's peak. A line `VmHWM reset failed,count,...`
means the kernel refused the reset on some rank, so `peakrss` is cumulative there. glibc keeps freed
memory resident. A stage's own peak is therefore `peakrss(stage) − rss(previous stage)`, not `peakrss`.
For a count, `max · P / sum` is the load imbalance (1 is perfectly balanced).

PETSc's `-log_view` and `-memory_view` have to be set before `PetscInitialize`, i.e. with
`PETSC_OPTIONS=...` in the environment. `-o` inserts its options after initialization.

The L2 error does not depend on the number of ranks (e.g. 1.573e-9 / 1.575e-9 /
1.575e-9 / 1.573e-9 for P = 1/2/4/8, and identical in 3D). This confirms that the
element partition and the COO summation are consistent.

## Measurements

Each number is the **maximum per-rank heap growth** in MiB, measured with
`mallinfo2` around each stage, or an object size estimate. All runs use
the default reservation of `gsExprAssembler` unless marked `noreserve`.
Machine: 4 cores, 15 GB, OpenMPI 4.1, PETSc 3.19, CG + GAMG.

### 2D, p = 2, 4 patches, N = 1 050 625 dofs, growing P

| object | P=1 | P=2 | P=4 | P=8 | behaviour |
|---|---:|---:|---:|---:|---|
| `gsFiberMatrix` after `initSystem` | 444.9 | 444.9 | 444.9 | 444.9 | **replicated** |
| rhs vector (`m_rhs`) | 8.0 | 8.0 | 8.0 | 8.0 | replicated |
| `gsDofMapper` (dense storage, the default) | 4.0 | 4.0 | 4.0 | 4.0 | replicated |
| full solution vector (for `gsFeSolution`) | 8.0 | 8.0 | 8.0 | 8.0 | replicated |
| solution as `gsMultiPatch` (`extract`) | 8.1 | 8.1 | 8.1 | 8.1 | replicated |
| geometry `gsMultiPatch`, `gsMultiBasis`, BCs | 0.1 | 0.1 | 0.1 | 0.1 | replicated, but coarse |
| PETSc Mat/Vec via COO | 723.3 | 371.5 | 187.7 | 95.9 | distributed |
| KSP (GAMG) setup + solve | 132.6 | 167.7 | 85.2 | 44.1 | distributed |
| local + ghost solution entries (what is actually needed) | 12.0 | 6.0 | 3.0 | 1.5 | distributed |
| legacy path: global `gsSparseMatrix` CSC + CSR copy | 607.3 | 307.7 | 157.9 | 83.2 | local nnz, but global outer index |

### 3D, p = 2, 8 patches, N = 274 625 dofs, growing P

| object | P=1 | P=2 | P=4 | P=8 |
|---|---:|---:|---:|---:|
| `gsFiberMatrix` after `initSystem` | 534.3 | 534.3 | 534.3 | 534.3 |
| PETSc Mat/Vec via COO | 857.0 | 475.8 | 284.1 | 189.3 |
| KSP (GAMG) | 40.3 | 66.7 | 35.4 | 21.0 |
| rhs / solution / solution multipatch / mapper (dense storage) | 2.1 / 2.1 / 2.4 / 1.2 | same | same | same |

### Replicated cost per global dof, per rank

| object | bytes per global dof |
|---|---:|
| `gsFiberMatrix`, default reservation, 2D p=2 | 444 |
| `gsFiberMatrix`, default reservation, 2D p=4 | 1 332 |
| `gsFiberMatrix`, default reservation, 3D p=2 | 2 040 |
| `gsFiberMatrix`, `noreserve` (one heap `gsSparseVector` + pointer per column) | 48 |
| rhs vector | 8 × #rhs |
| full solution vector | 8 |
| solution `gsMultiPatch` | 8 × #components |
| `gsDofMapper`, dense storage (the default) | 4 × #components |
| refined geometry (`--geo`) | 8 × d |
| OpenMP pattern locks (`std::vector<omp_lock_t>(numDofs())`, only with OpenMP) | 4 |

The `gsDofMapper` rows here and below are for dense storage; the opt-in sparse storage is described in
[Sparse `gsDofMapper` storage](#sparse-gsdofmapper-storage).

Even with the reservation fixed, roughly 76 B × N stays on every rank (2D
scalar). With 2 GB per rank, that caps N at about 25 M dofs, however many
ranks are used.

### Variants (P = 4)

* **N = 4.2 M (2D, `-r 10`) with the default reservation was OOM-killed** in a 15 GB
  container. With `--noreserve` it runs: fiber matrix 553 MiB per rank (192 MiB of it replicated
  column objects), peak RSS 2.6 GB per rank.
* `--noreserve`: fiber matrix 445 → 138 MiB per rank in 2D and 534 → 113 MiB in 3D.
  Assembly time does not get worse (5.1 s → 4.6 s in 2D, 11.1 s → 10.0 s in 3D), because
  `computePattern` sizes the fibers anyway.
* Refined geometry (`--geo`, 1 M dofs): geometry `gsMultiPatch` 16.2 MiB per rank, replicated.
* 1024 patches (`-s 5 -r 5`, 1.1 M dofs): geometry 1.0, multibasis 1.8, composite domain 0.2 MiB.
  The patch count itself is cheap; the dof-proportional objects still dominate.
* Off-rank matrix entries (rows owned by another rank under PETSc's equal split):
  0.4 % (2D, 4 patches) but 13 % (2D, 1024 patches) and 20 % (3D, P = 8). The global
  patch-wise dof numbering and the element block partition only line up by accident.
* PETSc COO keeps permutation arrays (`PetscCount`, 8 B per entry) next to the AIJ data.
  That is about 2.3× the CSR size (723 MiB vs. ~315 MiB of AIJ at P = 1). It is distributed,
  but it is the largest single item once the fiber matrix is fixed.

## Findings, ranked by impact

1. **`gsFiberMatrix` in `gsExprAssembler` is the dominant non-scaling object.**
   `clearMatrix()` creates one heap-allocated `gsSparseVector` per *global*
   column and calls `reservePerColumn(nz)`. Here `nz = Π(bdA·p_i + bdB)·(1+bdO)` (33
   entries for 2D p=2, 166 for 3D p=2). Only the columns touched by the rank's
   elements are ever used.
2. **Global dense vectors**: `m_rhs` (assembler), the full solution vector
   required by `gsFeSolution` (`_Sv->at(ii)` with global `ii`), and
   `gsFeSolution::extract(gsMultiPatch&)`. Each costs 8 B × N per rank.
3. **`gsDofMapper` is global**: `m_dofs` has one entry per basis function per
   component on every patch, 4 B × N × components with the default dense
   storage. It is small next to (1) but still O(N). Sparse storage
   (see below) is O(marked positions) instead.
4. **Geometry `gsMultiPatch`** costs O(N) only when the geometry is fine
   (8·d B/dof). Coarse CAD multipatches, even with 1024 patches, are cheap
   (topology is O(#patches)).
5. **Correctness trap**: an element subset restricts `allElements()` only.
   `assembleBdr` / `assembleIfc` / `computePatternBdr` iterate
   `domain().subdomain(patch)->beginBdr(side)`, i.e. all boundary elements. With
   an element-partitioned domain, every rank would add the full Neumann /
   interface terms, so they would be counted P times.
6. **Legacy helpers in `PETScSupport.h`**:
   * `petsc_copySparseMat` assumes that each rank already holds *complete* rows for
     its PETSc-owned range, in a global-size `gsSparseMatrix<RowMajor>`. With element
     partitioning, rows on partition boundaries are incomplete on every rank, so the
     copy would drop contributions. It also builds two global-size index maps.
   * `petsc_copyVecToGismo` declares `index_t rowIDs[M]; real_t vals[M];` on the
     stack. With the default 8 MB stack, that overflows at about M ≈ 700 k (found by
     reading the code, not by running it), and it always gathers the full vector.
7. `gsDirichletValues` (interpolation / L2 projection) is computed for the whole
   boundary on every rank. It is O(N^{(d-1)/d}) and minor.

## Recommendations

**Short term (small gismo patches)**

* Do not reserve per global column in `gsExprAssembler::clearMatrix()` when
  `computePattern` is going to run anyway (or reserve after the pattern is
  known). Allocate fibers lazily (`nullptr` until first touched) to remove the
  remaining 48 B per column.
* Restrict boundary/interface iteration in partitioned domains. Rule: a boundary or
  interface element belongs to the rank that owns the adjacent volume element (on
  interfaces, the left-hand one).
* Replace the VLAs in `petsc_copyVecToGismo` with `VecGetArrayRead`.

**Assembler back end (your templating question)**

Prefer member templates over making `gsExprAssembler` a class template:

```cpp
template<class Sink, class... expr> void assemble_into(Sink & sink, const expr &... args);
template<class Sink, class... expr> void computePattern_into(Sink & sink, const expr &... args);
// assemble(args...) == assemble_into(m_fiberSink, args...)
```

* `_eval::push` becomes `sink.add(rowIdx, colIdx, localMat)` per element block (with
  the elimination / rhs-correction logic kept in the assembler). This replaces the
  per-entry `coeffRef` + `omp atomic`. A PETSc sink can then call `MatSetValues` once
  per element, with negative indices for eliminated dofs (PETSc ignores them).
* `computePattern_into` into `gsPetscPatternSink` gives exact AIJ preallocation without
  COO's ~2.3× overhead, without any gismo-side matrix and without a PETSc hash table.
  `addPattern` stores the row and column index blocks of every element after mapping them
  through the permutation and dropping negative indices (thread-safe). `createMatrix` counts
  the distinct columns of each owned row and splits them into a diagonal and an off-diagonal
  block. The (row, column) pairs of rows owned by other ranks go to their owners in one
  `MPI_Alltoall` plus one `MPI_Alltoallv`, and the matrix is then preallocated with
  `MatXAIJSetPreallocation`. The preallocated slots equal the matrix nonzeros and no malloc
  occurs during assembly; `MAT_NEW_NONZERO_ALLOCATION_ERR` is set, so an entry outside the
  pattern aborts. Time is O(E·k²·log k) + O(received pairs), with E the number of `addPattern`
  calls and k the block size. Memory is O(E·k) stored indices, released before the matrix is
  allocated. The sink is single-use and supports AIJ matrix types only. The driver prints
  `preallocated slots (sum over ranks)` and `mallocs (max over ranks)` as evidence of exactness.
* Advantages over `gsExprAssembler<T, Backend = gsFiberMatrix<T>>`: no new
  type that leaks into every module and the Python bindings; several back ends per
  assembler (e.g. a mass matrix in gismo and the stiffness matrix in PETSc); the rhs
  sink can be a PETSc `Vec` as well, which removes `m_rhs`. The existing
  `fiberMatrix()`, `matrix()` and `rhs()` API keeps working.
* OpenMP + MPI: `MatSetValues` is not thread-safe, so a sink needs per-thread
  staging buffers or a critical section. The concept should declare whether it is
  thread-safe.

**Partitioned objects (this folder, next steps)**

* `gsPartitionedDofMapper`: built only for the patches touched by local elements
  (+ interface matching with neighbouring patches). Ownership is a contiguous global
  range per rank (`MPI_Exscan`), and shared dofs go to the lowest touching rank. It
  exposes a local→global map (`ISLocalToGlobalMapping`, so the sink can use
  `MatSetValuesLocal`). Then the PETSc layout equals the ownership layout, and
  off-rank entries are limited to the partition interfaces.
* Ghosted solution: `gsFeSolution` should read coefficients through an index
  map instead of a raw global `gsMatrix*`, e.g. a `VecGhost` local form + the
  partitioned mapper. That needs only owned + ghost entries: 25 % of N at P = 4 here,
  instead of 100 %. Nonlinear assembly then calls `VecGhostUpdate` once per Newton step.
* `gsPartitionedMultiPatch`: owned + halo patches (via the replicated topology),
  only needed when geometries are fine or the patch count is very large. Output
  of the solution per rank (`.pvtu`) instead of `extract(gsMultiPatch&)`.

## Status with `features/petsc-support-partitioned-dm` + sinks

Variants of `mpi_memory_profile` (all give identical L2 errors):
`block` (element block partition, PETSc equal split), `lazy` (`lazyMatrix`),
`rcb` (`gsGeometricPartitioner` + `gsPartitionedDofMapper` ownership),
`sink` (`computePattern_into` into `gsPetscPatternSink`, `assemble_into` into `gsPetscSystemSink` with `MatSetValues`).

Peak RSS per rank, P = 4:

| case | block | block+lazy | rcb+lazy | block+sink | rcb+sink |
|---|---:|---:|---:|---:|---:|
| 2D p=2, N = 1.05 M | 922 MiB | 649 | 577 | 445 | 450 |
| 3D p=2, N = 275 k | 1106 MiB | 868 | 760 | 505 | 483 |

The peak RSS tables in this section were measured when the pattern stage was still based on
PETSc's `MATPREALLOCATOR`.

rcb+sink, 2D N = 1.05 M, peak RSS per rank for P = 1/2/4/8: 1467 / 846 / 450 / 251 MiB.
rcb+sink, 2D N = 4.2 M, P = 4: 1671 MiB per rank. With the default `initSystem` reservation, the same run was OOM-killed.

Matrix entries assembled in rows owned by another rank (3D, P = 4): block 52 % of element-block
entries, causing 161 MiB of PETSc stash per rank. With rcb it is 6 % (20 MiB of stash).

### Replicated cost per global (scalar) dof, rcb+sink

| object | B/dof | lifetime |
|---|---:|---|
| `gsDofMapper`, dense storage (the default) | 4 | whole run |
| `gsDofMapper`, sparse storage (opt-in) | O(marked positions), not O(N): `nBytes()` 43 732 B after `localize`, max over ranks (2D, p = 2, r = 9, P = 4, rcb; dense 4 227 608 B); a container-capacity estimate, not a heap measurement | whole run |
| permutation (`gsPartitionedDofMapper::permutation()`) | 4 | whole run |
| partitioner labels/weights + ownership tables | ~20 | transient (partitioning) |
| full solution vector for `gsFeSolution` (+ unpermuted copy) | 8 (+8) | post-processing |
| solution `gsMultiPatch` (`extract`) | 8 | optional |
| `lazy` without sink: fiber pointers + `m_rhs` | 16 | assembly |
| fine (deformed) geometry | 8·d | if used |

### Distributed cost per owned dof (p = 2)

| | 2D | 3D |
|---|---:|---:|
| final AIJ matrix | ~340 B | ~1.6 kB |
| GAMG setup + CG | ~320 B | ~470 B |

The `MATPREALLOCATOR` transient (~520 B in 2D, ~2.2 kB in 3D) belonged to the earlier pattern stage.
`gsPetscPatternSink` stores the element index blocks until `createMatrix`: 33 538 128 B on the
most loaded rank and 1.2159002e+08 B summed over ranks (2D, p = 2, r = 9, P = 4, rcb, dense
mapper; median of 3 repeats), i.e. 4.644 B per matrix nonzero and 116 B per element. The transient
peak inside `createMatrix` (stored blocks, the row→call CSR and the pair buffers) was not measured,
and nothing was measured in 3D.
### Pattern sink: measured effect (r = 9, P = 4)

2D, p = 2, 4 patches, `--partition rcb --local --rendezvous --sink`, `OMP_NUM_THREADS=1`. PETSc is the
`-O0` debug build and `GISMO_ASSERT` is active in G+Smo, so absolute times are inflated. The earlier
`MATPREALLOCATOR`-based sink is one run on an earlier tree at load 3.9 with 13 GiB of swap in use; the new sink is the median
(min..max) of 3 repeats at load 3.2–5.2, with 3 GiB of swap in use. The within-run ratio of pattern
time to `assemble_into` time is therefore the time figure to read.

| quantity | `MATPREALLOCATOR` sink | `gsPetscPatternSink` |
|---|---:|---:|
| pattern-stage heap growth, max over ranks [MiB] | 136.93 | 31.99 (31.99..31.99) |
| pattern-stage heap growth, min over ranks [MiB] | 136.47 | 25.99 (25.99..25.99) |
| `computePattern_into` time [s] | 3.173 | 0.1978 (0.1974..0.1994) |
| `assemble_into` time [s] | 2.906 | 2.951 (2.917..3.176) |
| pattern time / `assemble_into` time | 1.092 | 0.06688 (0.06278..0.06782) |

Both sinks give the same system: nnz 26 183 689, 19 KSP iterations, L2 error 1.57399e-09 and
55 251 off-rank element-block entries. The new sink's preallocated slots equal the nonzeros
(26 183 689) with 0 mallocs. For the old sink "slots = nnz" holds by construction. The
dense and the sparse mapper (next section) also give identical nnz, KSP iterations and L2 error.

### Sparse `gsDofMapper` storage

Dense storage, the default, keeps one `index_t` per basis function per component (O(N_c) for a
component with N_c positions), which is a replicated O(N) cost on every rank. The opt-in
`gsDofMapper::storage::sparse` reduces it to O(marked positions).

**What is stored.** Only the marked positions: interface/coupled, eliminated (Dirichlet) and
collapsed dofs. Regular dofs are numbered by counting. The numbering, and the answer to every query,
are identical to the dense mapper built by the same calls.

**Memory and lookup cost** (M_c = number of marked positions of component c):

| phase | stored | memory | lookup |
|---|---|---|---|
| setup (before `finalize()`) | one hash map per component | O(M_c) | expected O(1) |
| after `finalize()` | sorted marked positions, their values, one base id per component | O(M_c) | O(log M_c) |
| after `localize()` | table of maximal runs of consecutive positions with consecutive local ids | O(#runs), #runs ≤ (maximal runs of local regular dofs) + M_c | O(log #runs) |

No pass over all positions is needed.

**Enabling it.** The storage argument comes after `finalize` in every overload:

```cpp
// factory; the driver uses this form
gsDofMapper m = createMapper(mb, bc, nComp, unk, /*conforming=*/true,
                             /*finalize=*/false, gsDofMapper::storage::sparse);
// from patch dof sizes
gsDofMapper m2(patchDofSizes, nComp, gsDofMapper::storage::sparse);
// identity mapping: setIdentity(..., gsDofMapper::storage::sparse)
// expression space, before setup()
u.setMapperStorage(gsDofMapper::storage::sparse);
u.setup(bc, dirichlet::interpolation, 0);
```

The `setMapperStorage` setting also applies when the assembler rebuilds the mapper.

**Limits.** `permuteFreeDofs()` converts the mapper to dense storage. The storage argument is not
exposed in the Python bindings.

**Driver.** `--sparse-mapper` sets sparse storage on the space and prints
`gsDofMapper nBytes (max over ranks, …): setup …, finalized …, localized …` next to the dense table
`mapSize()*sizeof(index_t)`. "setup" is a separate, unfinalized mapper built by the same
`createMapper` call. "finalized" and "localized" are the space's mapper right after `u.setup` and after
localization. `--check-mapper` (needs `--sparse-mapper`) compares the mapper query by query with a
dense twin built by the same calls, and once more after an extra localization. It aborts on the first
difference.

**Measured** (2D, p = 2, r = 9, P = 4, `--partition rcb --local --rendezvous --sink`,
`OMP_NUM_THREADS=1`; `-O0` debug PETSc, `GISMO_ASSERT` active, load 3.2–5.2 and 3 GiB of swap in use,
dense and sparse runs interleaved). The byte counts are `gsDofMapper::nBytes()`, the capacity of the
mapper's containers with the hash maps estimated from buckets and stored pairs, not heap measurements.
Maximum over ranks:

| `nBytes()` after | dense | sparse | sparse / dense |
|---|---:|---:|---:|
| setup (separate unfinalized mapper) | 4 228 088 | 148 856 | 0.0352 |
| `finalize()` | 4 227 608 | 66 188 | 0.0157 |
| `localize()` | 4 227 608 | 43 732 | 0.0103 |

The dense table `mapSize()*sizeof(index_t)` is 4 227 136 B. The localized sparse size depends on the
rank count and the partition.

Times, median (min..max) of 3 repeats, dense / sparse [s]: space setup (includes the Dirichlet
interpolation) 0.01202 (0.01187..0.01619) / 0.008362 (0.005857..0.008542); the "localize mapper" stage
(spans `localFreeDofs`, the rendezvous numbering and `localizeSpace`, not `gsDofMapper::localize` alone)
0.2581 (0.2572..0.2621) / 0.2673 (0.2663..0.2688); `assemble_into` 2.951 (2.917..3.176) / 3.019
(3.002..3.032). Sparse setup is a few milliseconds shorter, the localize stage 1–7 % longer over the
configurations measured, and `assemble_into` differs by a few percent in either direction.

## Weak scaling (P = 1, 2, 4; 4-core container)

These tables were measured with the `MATPREALLOCATOR`-based pattern stage and dense mapper storage.

About 262k dofs per rank (2D, p=2) and 275k (3D, p=2): the patch grid grows with `--aspect P`.
All variants give the same L2 error per size. GAMG needs 20–22 CG iterations in every run.

Peak RSS per rank [MiB]:

| variant | 2D P=1 | 2 | 4 | 3D P=1 | 2 | 4 |
|---|---:|---:|---:|---:|---:|---:|
| block (stable `initSystem`/`assemble`) | 518 | 681 | 929 | 2295 | 3097 | OOM-killed |
| rcb+sink | 397 | 428 | 450 | 1574 | 1823 | 1845 |
| rcb+local+rendezvous+sink | 417 | 445 | 448 | 1609 | 1717 | 1755 |
| block+local+rendezvous+sink | 416 | 444 | 447 | 1608 | 1812 | 1826 |

Stage wall time [s], rcb+local+rendezvous+sink:

| stage | 2D P=1 | 2 | 4 | 3D P=1 | 2 | 4 |
|---|---:|---:|---:|---:|---:|---:|
| computePattern_into | 2.05 | 2.15 | 2.07 | 18.98 | 19.06 | 19.40 |
| assemble_into | 2.89 | 3.03 | 3.11 | 29.08 | 30.56 | 31.99 |
| KSP setup + solve | 1.70 | 2.51 | 2.78 | 10.16 | 11.59 | 14.39 |
| error evaluation | 0.93 | 0.79 | 0.95 | 3.43 | 3.89 | 4.28 |
| RCB partition (replicated) | 0.10 | 0.20 | 0.44 | 0.15 | 0.36 | 0.73 |
| localize (scans the global mapper) | 0.35 | 0.42 | 0.43 | 0.63 | 0.79 | 0.91 |

What still grows with the global size on every rank:

- `gsDofMapper`: 1 → 2 → 4 MiB in 2D, i.e. 4 B per global dof with the default dense storage.
  Sparse storage is O(marked positions) and does not grow with N this way.
- RCB labels and weights: 3 → 5 → 9 MiB while partitioning, plus O(N) time on every rank.
  The block partition with rendezvous ownership avoids the partitioner entirely.
  In 3D it has more ghost dofs and more stash than RCB.
- `localize`: one pass over the global mapper with dense storage. With sparse storage no loop runs
  over all positions: the cost is O(n_c + M_c + M_c log n) for the first localization (n the number
  of local dofs, n_c the part of it in the regular id range of component c, M_c the number of
  marked positions of component c).

The growth in KSP time comes from the cost per iteration (communication, and the
memory bandwidth of 4 ranks on 4 cores); the iteration counts are constant.
