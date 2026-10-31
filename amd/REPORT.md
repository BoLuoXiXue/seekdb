# AMD hipVS HNSW construction report (2026-10-31)

This report validates the independent AMD-only branch against official
`oceanbase/seekdb:2026-competition`, base `5bce9e06d1281347ad61344e4b01fab9158c58a2`.
Tested implementation: `dd874eff03ba9ab417f8cec60955d8ca76dabf7b`. A subsequent documentation/evidence
commit changes no compiled, deployed or benchmark inputs. Earlier measurements
from the integrated contest repository are not reused as this branch's results.

## Implementation

An isolated worker runs hipVS CAGRA graph construction and full GPU HNSW hierarchy
conversion, validates every exported FP32 vector and row label, restores original
64-bit VIDs, and imports the complete graph into native VSAG. Native CPU search,
filtering, incremental writes and persistence remain in use. The CPU path is
preserved; unsupported snapshots log fallback, while a required GPU or worker
failure is an error. See [README.md](README.md) for scope and commands.

## Fixed measurement conditions

Same AMD cloud instance, Linux x86_64, gfx1100 (~48 GiB GPU memory), ROCm 7.2.4,
cgroup 16 CPU / 55 GiB RAM. Database cpu_count=8, memory_budget=8G,
OMP_NUM_THREADS=8. Build and benchmarks run sequentially.

The two inputs are the image's complete 10,000-vector demo and seed=20261030
Gaussian 100,000-vector data, both 128D FP32 L2. Each has 100 fixed queries,
independent exhaustive FP64 truth, K=10, and IDs `100000000001 + 3*i`.
Every report records all input hashes. Exact library fingerprints and Python
package versions are in [environment.json](results/upstream-20261031/environment.json).
Both backends use the same M, construction
ef=200 and actual session search ef=1000.

CPU and GPU use the **same freshly built candidate binary**: CPU is the original
VSAG construction path selected with SEEKDB_HIPVS_BUILD=0; GPU runs standard
deployment in default auto mode without BUILD or WORKER overrides. This is a
controlled backend comparison, not a claim that the CPU binary is an independently
compiled unmodified upstream commit. All four groups per backend are measured
anew, with three fresh DROP/CREATE rounds each and independent database stores.

Timing starts before CREATE VECTOR INDEX and ends after the same availability
probe and self-query. It includes scanning, worker startup and file transport,
GPU transfers/synchronization, full hierarchy conversion, native import, and any
work required before the first queries. Generation, raw import and startup are
separate. All 1,000 results per round are checked for count, valid IDs, distances
and ordering before calculating Recall@10 with a fixed denominator.

## Every end-to-end round

| Rows / M | Backend | Round 1 / 2 / 3 (seconds) | Median (seconds) | Recall@10 round 1 / 2 / 3 |
| --- | --- | --- | ---: | --- |
| 10000 / 16 | cpu | 2.854518 / 3.535700 / 2.739685 | 2.854518 | 1.000 / 1.000 / 1.000 |
| 10000 / 16 | gpu | 1.545779 / 1.526878 / 1.526820 | 1.526878 | 1.000 / 1.000 / 1.000 |
| 10000 / 64 | cpu | 3.763047 / 3.752986 / 3.751489 | 3.752986 | 1.000 / 1.000 / 1.000 |
| 10000 / 64 | gpu | 1.527885 / 1.529817 / 1.528186 | 1.528186 | 1.000 / 1.000 / 1.000 |
| 100000 / 16 | cpu | 40.900807 / 39.502785 / 40.203188 | 40.203188 | 0.934 / 0.934 / 0.934 |
| 100000 / 16 | gpu | 7.350914 / 7.064734 / 7.167190 | 7.167190 | 0.975 / 0.969 / 0.968 |
| 100000 / 64 | cpu | 108.682036 / 108.483221 / 108.620583 | 108.620583 | 0.998 / 0.998 / 0.998 |
| 100000 / 64 | gpu | 8.877836 / 8.777554 / 8.972270 | 8.877836 | 1.000 / 1.000 / 1.000 |

| Rows / M | CPU median / GPU median | Minimum CPU / GPU Recall@10 |
| --- | ---: | --- |
| 10000 / 16 | 1.869513× | 1.000 / 1.000 |
| 10000 / 64 | 2.455843× | 1.000 / 1.000 |
| 100000 / 16 | 5.609338× | 0.934 / 0.968 |
| 100000 / 64 | 12.235030× | 0.998 / 1.000 |

These are development-data results. Official evaluation data, speedup target and
Recall threshold were not supplied; this report does not claim an official score
or acceptance. The companion reports preserve all timings and input hashes.

## Product validation

- [Full native product build](results/upstream-20261031/build.json): PASS; valid
  object cache reused, affected translation units rebuilt and final server linked in the
  [initial native build](results/upstream-20261031/initial-build.json). The final
  configuration change reran the full product target and retained identical
  server/bridge/worker hashes.
  All 9,191 tracked source hashes were unchanged during the build.
- Explicit GPU OFF removes discoverable/deployed helpers; re-enabling restores
  them without requiring a relink: PASS.
- Standard server-component helper install and deploy: PASS. Deployed server,
  bridge and worker hashes match the full build. GPU worker discovery uses the
  executable directory, independent of the current working directory.
- [24 new CPU/GPU construction rounds](results/upstream-20261031/acceptance.json):
  PASS, with all fixed-query correctness checks. [12 GPU build records](results/upstream-20261031/gpu-markers.txt)
  identify full row counts, hierarchy levels and cpu_fallback=false.
- [Original HNSW-related mysqltest suite](results/upstream-20261031/original-vector-summary.json):
  20/20 PASS against complete unchanged SQL and expected files. No new test cases
  or modified expectations were introduced.
- [Restart with GPU construction disabled](results/upstream-20261031/persistence-cpu-only.json):
  400 queries across four GPU-built indexes exactly match pre-restart IDs,
  distances and order, without rebuilding.
- Normal shutdown and worker/temporary-file cleanup: PASS. No long-duration
  stress or peak-memory measurement is claimed.

Server SHA256: `850b73adc5536863aaa0d7c66fd1df55768369028c10c211a07883ae20b5a426`.
Bridge SHA256: `f0357e1ac7686c618b6b3a318859b3d58ebe4bb37f1801a67d542efaa2b4e2af`.
Complete original logs, JSON query outputs, vectors/truth, exact source and final
native products are retained in the persistent workspace outside the cloud
container. Published JSONL changes formatting only; query-hashes.json binds it
to the original JSON, and all values were compared for equality.

## Limits

GPU construction supports ordinary FP32 L2 HNSW without inline extra info.
Other metrics/types and small snapshots explicitly use CPU. Effective M follows
the pinned VSAG clamp [8,64]. GPU-built graphs stay in the ordinary tenant cache
through snapshot publication even when memory-saving mode is enabled; normal
cache eviction/shutdown still frees them. This host-memory tradeoff and temporary
GPU/disk requirements are documented. GPU query acceleration, additional distance
types, GPU build concurrency admission, full cross-platform validation and
undisclosed evaluation requirements are outside the validated scope.
