# AMD hipVS HNSW construction

Reproduction tooling for the AMD HNSW construction contribution, based on official
`2026-competition` commit `5bce9e06d1281347ad61344e4b01fab9158c58a2`.
Measured results and validation records are in [REPORT.md](REPORT.md).

This implementation accelerates bulk HNSW **construction** inside seekdb. It
does not require GPU search or the demo image's `lib=cuvs` SQL extensions.

## Build

Use Linux x86_64, a supported AMD GPU, ROCm, and a hipVS installation exporting
the CAGRA and HNSW C APIs. The development environment uses ROCm 7.2.4 and
`/opt/hipvs` on gfx1100. Initialize ordinary seekdb dependencies as usual, then:

```bash
bash build.sh release --make -j8
```

On Linux x86_64, the normal build detects hipVS under `/opt/hipvs` and enables
its worker automatically when the C API and ROCm headers are available. Override
`HIPVS_ROOT` for another installation, or pass `-DOB_BUILD_HIPVS=OFF` for a CPU-only
build. The bridge calls the host C API; it does not compile GPU kernels or fix a
GPU architecture. Kernels come from the installed hipVS library.

The complete product build produces `build_release/src/observer/seekdb` and
`build_release/libseekdb_hipvs_build.so`, together with
`build_release/seekdb_hipvs_build_worker`. Without `OB_BUILD_HIPVS`, normal CPU
builds do not need ROCm or hipVS.

## Enable construction

The server discovers its GPU worker beside its own executable. With no GPU
environment variables, it automatically uses the worker when `/dev/kfd` is
accessible; otherwise it records CPU fallback. No path depends on the shell's
current directory. The build places the worker and bridge beside the server,
and standard installation/deploy commands include them.

For controlled comparisons, set `SEEKDB_HIPVS_BUILD=0` for CPU, or
`SEEKDB_HIPVS_BUILD=1` to require GPU construction and fail if unavailable.
`SEEKDB_HIPVS_BUILD=auto` is the default. An explicit
`SEEKDB_HIPVS_BUILD_WORKER=/absolute/path/to/worker` remains supported.

```bash
export OMP_NUM_THREADS=8
export TMPDIR=/path/to/private-build-temporaries
mkdir -p -m 700 "$TMPDIR"
# Optional: require GPU instead of automatic selection.
export SEEKDB_HIPVS_BUILD=1
```

Use the ordinary SQL syntax after loading the raw data into an unindexed table:

```sql
CREATE VECTOR INDEX hnsw_idx ON vectors(embedding)
WITH (distance=l2, type=hnsw, lib=vsag, m=16,
      ef_construction=200, ef_search=1000);
```

The switch applies to bulk snapshot construction, including post-create and
rebuild paths. Mutable incremental indexes continue to use VSAG. Starting with
`SEEKDB_HIPVS_BUILD=0` explicitly retains CPU construction.

## Construction path

1. Collect every snapshot vector and its original 64-bit vector ID. The input
   buffers use the index's existing tenant allocator and support concurrent
   scanner batches.
2. hipVS constructs the CAGRA graph on the AMD GPU. For snapshots with at least
   16,384 rows,
   GPU IVF-PQ proposes candidates and refines them against the complete original
   FP32 vectors, with FP32 lookup and distance accumulation. PQ codes are never
   stored in the resulting HNSW or used for SQL search. Smaller snapshots use
   NN-Descent with a larger candidate set to preserve low-degree graph recall.
   Its base-layer degree is `2*m`; the intermediate graph degree is at least
   `max(4*m, ef_construction)`, capped by the available rows. The small-snapshot
   NN-Descent path uses at least 256 candidates when the row count permits it.
3. hipVS constructs the **full HNSW hierarchy on the GPU**, synchronizes, and
   serializes its hnswlib-compatible index. This is not a base-layer-only index.
4. Validate the serialized layout, row count and neighbor IDs, then replace
   temporary row labels with the original vector IDs. This does not change the
   vectors, graph edges, precision or dataset size.
5. Import the complete graph through VSAG's existing HNSW deserializer. Queries,
   filtering, distance-by-ID operations, and seekdb's persistent serialization
   use the original VSAG implementation. A persisted index can be read with GPU
   construction disabled.

The server invokes a fresh GPU worker using `posix_spawn` and waits for its
completion. Input vectors and IDs, the resulting HNSW file, and statistics use
private temporary files; their transport and worker startup/teardown are included
in the construction time. The worker links the versioned C ABI bridge and hipVS,
without VSAG or seekdb's allocator interposer. This separates the two libraries'
hnswlib implementations and allocators. No C++ objects or allocated pointers
cross the process boundary. The server does not include HIP/cuVS headers or
link the GPU libraries into its ordinary CPU build.

## Supported scope and failures

- GPU construction: dense FP32, L2, ordinary HNSW, `2 <= m <= 256`, snapshots
  with at least 256 rows and more than twice the effective M, and no inline extra-info payload.
  The effective M is clamped to `[8,64]`, exactly like the pinned VSAG CPU
  implementation; effective construction breadth is at least this M. The
  success log records declared and effective values.
- Small snapshots, other metrics/index kinds and inline extra-info use the
  original CPU builder with an explicit `[HIPVS_BUILD] CPU fallback` log.
- A requested but missing worker, GPU failure, incompatible serialization or
  failed VSAG import rejects the build. It is not silently retried on CPU.
- Existing `lib=vsag` declarations stay valid. The server-side opt-in currently
  applies to all eligible bulk snapshots in that process, rather than to an
  individual index.
- GPU transient allocations are owned by hipVS/HIP; host input and import
  buffers use the existing tenant allocator. Budget GPU memory and temporary
  disk space separately. Intermediate files are removed on success and failure.
- The imported GPU graph remains in the normal tenant cache after DDL, even
  when `vector_index_memory_saving_mode` is enabled. This prevents a shared
  snapshot adapter from retaining a valid prefix after another adapter has
  released its graph. Normal cache eviction and shutdown still release it.
  CPU memory-saving behavior is unchanged; account for this host-memory tradeoff
  when comparing end-to-end results.

Successful builds emit `[HIPVS_BUILD] GPU HNSW build and VSAG import completed`,
including the row count, device, hierarchy levels and stage timings. These
stage timings are diagnostics, **not** the end-to-end score.

## Reproduce the measurements

Install the benchmark dependencies in a dedicated Python 3.11+ environment:

```bash
python3 -m venv amd-venv
amd-venv/bin/pip install numpy pymysql
amd-venv/bin/python amd/benchmark.py prepare --data /path/to/data-100k \
  --rows 100000 --dim 128 --queries 100 --k 10 --seed 20261030
```

For the image's original 10,000-vector dataset, add
`--demo /seekdb_workspace/datasets --rows 10000` instead. The script preserves
all its vectors and calculates exact ground truth independently. Each dataset
has a manifest with input hashes and deliberately non-contiguous 64-bit IDs.

Run CPU and GPU groups on the same machine, CPU allocation, database settings,
data and index parameters. Use independent database directories. Pass a fresh
schema and a fresh output directory for each group:

```bash
amd-venv/bin/python amd/benchmark.py run --data /path/to/data-100k \
  --output /path/to/cpu-results --schema amd_cpu_100k --backend cpu --port 2888
# Restart the database with GPU construction enabled, then:
amd-venv/bin/python amd/benchmark.py run --data /path/to/data-100k \
  --output /path/to/gpu-results --schema amd_gpu_100k --backend hipvs --port 2888
```

`--backend` labels the report; server startup variables select the actual
backend. Preserve server startup records and successful GPU construction logs
to prove which path executed. A label alone does not prove GPU execution.

The query session explicitly sets `ob_hnsw_ef_search` to the configured value
(default 1000), since the session variable overrides the index declaration.

The repaired implementation is measured with both `m=16` and `m=64`,
`ef_construction=200`, and `ob_hnsw_ef_search=1000` for both CPU and GPU. Pass
`--m 16` or `--m 64` to reproduce each profile. A profile with poor recall cannot
be counted as a quality-qualified speedup.

Each group builds three fresh indexes. Timing starts before `CREATE VECTOR
INDEX`, and stops after the index is usable by both a probe query and a self
query. It includes scanning, worker startup/teardown, file transport, host/device transfers, GPU synchronization,
hierarchy construction, format conversion, import, persistence required by
DDL, and any work deferred to these first queries. Import and data preparation
are separately recorded. Every round then runs all 100 queries, validates IDs
and SQL Euclidean distances, and computes Recall@10 against exhaustive FP64
ground truth with a fixed denominator.

Divide the CPU median end-to-end build time by the GPU median. Preserve all
rounds, regressions, logs and hashes; do not substitute GPU kernel time or the
notebook's batch-search speedup. The competition documents supplied here do not
specify the official speedup target or Recall threshold, so these fields remain
unknown until the evaluation configuration is supplied.

## References

- [Competition task](https://oceanbase.yuque.com/org-wiki-obtech-vh7w9r/hc3pio/seebnbaoxot1b0tb)
- [hipVS](https://github.com/AMD-Ecosystem/hipVS)
- hipVS installation: `include/cuvs/neighbors/cagra.h` and
  `include/cuvs/neighbors/hnsw.h` document graph construction and full hierarchy
  conversion; the latter distinguishes GPU/CPU hierarchies from `NONE`.

## Original SQL regression suite

Run the 20 unchanged HNSW-related cases from the repository's original
`vector_index` suite against a dedicated test instance. The runner executes the
original SQL and compares its complete output to the original expected files;
it records input hashes, commands, exit codes and per-case logs. It does not
rewrite expected results or add test cases.

```bash
amd-venv/bin/python amd/run_original_tests.py --tree "$PWD" \
  --mysqltest /path/to/obclient/bin/mysqltest --port 2888 \
  --output /path/to/original-vector-results
```

Use the repository-pinned obclient 2.2.2 test client. The runner creates a
local-only `admin` test account and fixture directories. It must be used on the
isolated test instance, since original cases create/drop their own databases and
modify test configuration.
