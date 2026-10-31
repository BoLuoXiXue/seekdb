#!/usr/bin/env python3
"""Reproduce end-to-end HNSW construction and SQL Recall@K measurements.

Data preparation/import are outside the construction clock. Every measured
round creates a new index and includes its first SQL queries in the clock.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import numpy as np
import pymysql


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def literal(vector):
    # Nine significant decimal digits round-trip every finite FP32 value.
    return "[" + ",".join(format(float(x), ".9g") for x in vector) + "]"


def prepare(args):
    out = args.data
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    if args.demo:
        base = np.fromfile(args.demo / "base.f32", dtype="<f4").reshape(-1, args.dim)
        queries = np.fromfile(args.demo / "query.f32", dtype="<f4").reshape(-1, args.dim)
        if base.shape[0] != args.rows or queries.shape[0] < args.queries:
            raise ValueError("demo data dimensions do not match the requested configuration")
        queries = queries[:args.queries].copy()
    else:
        rng = np.random.default_rng(args.seed)
        base = rng.standard_normal((args.rows, args.dim), dtype=np.float32)
        queries = rng.standard_normal((args.queries, args.dim), dtype=np.float32)
    if not np.isfinite(base).all() or not np.isfinite(queries).all():
        raise ValueError("non-finite input")
    # Deliberately non-contiguous 64-bit document IDs exercise the GPU row map.
    ids = np.arange(args.rows, dtype=np.int64) * 3 + 100_000_000_001
    base.astype("<f4").tofile(out / "base.f32")
    queries.astype("<f4").tofile(out / "query.f32")
    np.save(out / "ids.npy", ids)
    truth = []
    for query in queries:
        # Exhaustive FP64 distances: no ANN index participates in the truth.
        distance = np.sum((base.astype(np.float64) - query.astype(np.float64)) ** 2, axis=1)
        top = np.argpartition(distance, args.k - 1)[:args.k]
        top = top[np.lexsort((ids[top], distance[top]))]
        truth.append(ids[top].tolist())
    np.save(out / "truth.npy", np.array(truth, dtype=np.int64))
    config = {"rows": args.rows, "dim": args.dim, "queries": args.queries,
              "k": args.k, "seed": args.seed, "demo": str(args.demo) if args.demo else None,
              "preparation_seconds": time.perf_counter() - start,
              "files": {p.name: sha256(p) for p in sorted(out.iterdir())}}
    save(out / "data.json", config)
    print(json.dumps(config), flush=True)


def connect(args):
    connection = pymysql.connect(host=args.host, port=args.port, user="root", database="test",
                                 autocommit=True, connect_timeout=10, read_timeout=1800)
    with connection.cursor() as cursor:
        cursor.execute("SET ob_query_timeout=1800000000")
        cursor.execute("SET ob_trx_timeout=1800000000")
        cursor.execute("SET ob_hnsw_ef_search=%s", (args.ef_search,))
    return connection


def query(cursor, vector, k):
    value = literal(vector)
    cursor.execute(f"SELECT id,l2_distance(embedding,{value}) AS distance FROM vectors "
                   f"ORDER BY l2_distance(embedding,{value}) APPROXIMATE LIMIT {k}")
    return [(int(row[0]), float(row[1])) for row in cursor.fetchall()]


def check_results(cursor, base, ids, queries, truth, k):
    output = []
    hits = 0
    for qid, vector in enumerate(queries):
        rows = query(cursor, vector, k)
        if len(rows) != k or len({row[0] for row in rows}) != k:
            raise AssertionError(f"query {qid}: missing or duplicate results")
        distances = []
        for doc_id, distance in rows:
            offset = doc_id - 100_000_000_001
            if offset < 0 or offset % 3 or offset // 3 >= len(ids):
                raise AssertionError(f"query {qid}: invalid document ID {doc_id}")
            index = offset // 3
            expected = float(np.linalg.norm(base[index].astype(np.float64) - vector.astype(np.float64)))
            if not np.isfinite(distance) or not np.isclose(distance, expected, rtol=2e-5, atol=2e-5):
                raise AssertionError(f"query {qid}: wrong ID/distance mapping: {doc_id}, {distance}, {expected}")
            distances.append(distance)
        if distances != sorted(distances):
            raise AssertionError(f"query {qid}: results not ordered by distance")
        hits += len(set(row[0] for row in rows) & set(map(int, truth[qid])))
        output.append({"query": qid, "rows": rows})
    # Missing queries/results can never improve the denominator.
    return hits / (len(queries) * k), output


def run(args):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.schema):
        raise ValueError("invalid schema")
    if args.rounds < 3:
        raise ValueError("at least three independently constructed indexes are required")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.data / "data.json").read_text())
    for name, expected in config["files"].items():
        if sha256(args.data / name) != expected:
            raise ValueError(f"input hash mismatch: {name}")
    n, dim, k = config["rows"], config["dim"], config["k"]
    base = np.fromfile(args.data / "base.f32", dtype="<f4").reshape(n, dim)
    queries = np.fromfile(args.data / "query.f32", dtype="<f4").reshape(-1, dim)
    ids, truth = np.load(args.data / "ids.npy"), np.load(args.data / "truth.npy")
    report = {"backend": args.backend, "schema": args.schema, "data": config,
              "parameters": {"m": args.m, "ef_construction": args.ef_construction,
                             "ef_search": args.ef_search, "distance": "l2", "dtype": "float32"},
              "official_speedup_target": None, "official_recall_threshold": None,
              "timing": "CREATE VECTOR INDEX through completion of first probe and self query",
              "rounds": [], "status": "running"}
    save(args.output / "report.json", report)
    connection = connect(args)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {args.schema}")
            cursor.execute(f"USE {args.schema}")
            cursor.execute("SHOW TABLES LIKE 'vectors'")
            if cursor.fetchone():
                raise RuntimeError("schema already contains vectors; use a fresh schema")
            cursor.execute(f"CREATE TABLE vectors(id BIGINT PRIMARY KEY, embedding VECTOR({dim}))")
            start = time.perf_counter()
            for begin in range(0, n, 256):
                values = [(int(ids[i]), literal(base[i])) for i in range(begin, min(begin + 256, n))]
                cursor.executemany("INSERT INTO vectors VALUES (%s,%s)", values)
            report["import_seconds"] = time.perf_counter() - start
            cursor.execute("SELECT COUNT(*),MIN(id),MAX(id) FROM vectors")
            report["input_row_check"] = list(cursor.fetchone())
            if tuple(report["input_row_check"]) != (n, int(ids[0]), int(ids[-1])):
                raise AssertionError("data import is incomplete")
            ddl = ("CREATE VECTOR INDEX hnsw_idx ON vectors(embedding) WITH "
                   f"(distance=l2,type=hnsw,lib=vsag,m={args.m},"
                   f"ef_construction={args.ef_construction},ef_search={args.ef_search})")
            report["ddl"] = ddl
            for iteration in range(args.rounds):
                if iteration:
                    cursor.execute("DROP INDEX hnsw_idx ON vectors")
                start = time.perf_counter()
                cursor.execute(ddl)
                ddl_seconds = time.perf_counter() - start
                first = query(cursor, queries[0], k)
                self_result = query(cursor, base[0], 1)
                elapsed = time.perf_counter() - start
                if len(self_result) != 1 or self_result[0][0] != int(ids[0]) or abs(self_result[0][1]) > 1e-5:
                    raise AssertionError("self query failed: invalid ID mapping or incomplete index")
                recall, results = check_results(cursor, base, ids, queries, truth, k)
                save(args.output / f"queries-{iteration + 1}.json", results)
                row = {"round": iteration + 1, "ddl_seconds": ddl_seconds,
                       "build_through_first_queries_seconds": elapsed, "recall_at_k": recall,
                       "returned_results": len(results) * k, "first_probe": first,
                       "self_query": self_result}
                report["rounds"].append(row)
                save(args.output / "report.json", report)
                print(json.dumps(row), flush=True)
            vector = literal(queries[0])
            cursor.execute(f"EXPLAIN SELECT id FROM vectors ORDER BY l2_distance(embedding,{vector}) APPROXIMATE LIMIT {k}")
            report["query_plan"] = [list(row) for row in cursor.fetchall()]
            report["median_build_seconds"] = statistics.median(
                row["build_through_first_queries_seconds"] for row in report["rounds"])
            report["minimum_recall_at_k"] = min(row["recall_at_k"] for row in report["rounds"])
            report["status"] = "measured"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        save(args.output / "report.json", report)
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--data", type=Path, required=True)
    prepare_parser.add_argument("--demo", type=Path)
    prepare_parser.add_argument("--rows", type=int, default=100000)
    prepare_parser.add_argument("--dim", type=int, default=128)
    prepare_parser.add_argument("--queries", type=int, default=100)
    prepare_parser.add_argument("--k", type=int, default=10)
    prepare_parser.add_argument("--seed", type=int, default=20261030)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--data", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--schema", required=True)
    run_parser.add_argument("--backend", choices=("cpu", "hipvs"), required=True)
    run_parser.add_argument("--host", default="127.0.0.1")
    run_parser.add_argument("--port", type=int, default=2888)
    run_parser.add_argument("--rounds", type=int, default=3)
    run_parser.add_argument("--m", type=int, default=16)
    run_parser.add_argument("--ef-construction", type=int, default=200)
    run_parser.add_argument("--ef-search", type=int, default=1000)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
