#!/usr/bin/env python3
"""Run all unchanged HNSW-related cases from the original vector_index suite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

TREE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
DEPLOY = TREE / 'tools/deploy'
SUITE = DEPLOY / 'mysql_test/test_suite/vector_index'
CLIENT = TREE / 'deps/3rd/u01/obclient/bin/mysqltest'
CASES = ('all_virtual_vector_index_info', 'create_table_with_hybrid_vector_index', 'create_table_with_vector_index', 'drop_vector_index', 'fix_rebuild_or_post_create_hung', 'rebuild_vector_index', 'sparse_vector_index_vsag_ddl', 'sparse_vector_index_vsag_query', 'vector_index_basic', 'vector_index_bugfix1', 'vector_index_heap_rr_snapshot', 'vector_index_identifier_quoting', 'vector_index_ivfflat_basic', 'vector_index_ivfflat_post_create', 'vector_index_offline_ddl', 'vector_index_partitioned', 'vector_index_post_create', 'vector_index_rebuild', 'vector_index_rr_snapshot', 'vector_similarity')

PASSWORD = 'local-regression'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tree', type=Path, default=TREE)
    parser.add_argument('--mysqltest', type=Path, required=True)
    parser.add_argument('--port', type=int, default=2888)
    parser.add_argument('--source-record', type=Path)
    args = parser.parse_args()
    global DEPLOY, SUITE, CLIENT, ROOT
    tree = args.tree.resolve(strict=True)
    DEPLOY = tree / 'tools/deploy'
    SUITE = DEPLOY / 'mysql_test/test_suite/vector_index'
    CLIENT = args.mysqltest.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ROOT = output
    port = args.port
    inputs = {}
    for name in CASES:
        test = SUITE / 't' / (name + '.test')
        result = SUITE / 'r/mysql' / (name + '.result')
        inputs[name] = {'test_file': str(test), 'result_file': str(result),
                        'test_sha256': sha256(test), 'result_sha256': sha256(result)}
    actual = {p.stem for p in (SUITE / 't').glob('*.test')}
    related = {p.stem for p in (SUITE / 't').glob('*.test') if 'hnsw' in p.read_text().lower()}
    if related != set(CASES):
        raise RuntimeError('Original HNSW-related suite inventory changed')
    git_command = ['git', '-c', 'safe.directory=*', '-C', str(tree)]
    manifest = {
        'tree': str(tree), 'suite': 'vector_index', 'host': '127.0.0.1', 'port': port,
        'source_record': json.loads(args.source_record.read_text()) if args.source_record else None,
        'mysqltest': str(CLIENT), 'mysqltest_sha256': sha256(CLIENT),
        'timeout_seconds_per_case': 600, 'cases': inputs,
    }
    write_json(output / 'manifest.json', manifest)
    results = {}
    write_json(output / 'status.json', results)
    setup_command = ['mysql', '--no-defaults', '-h127.0.0.1', '-P' + str(port), '-uroot']
    # Test-only account; the original cases also reconnect as admin and root@sys.
    (ROOT / 'fixtures').mkdir(exist_ok=True)
    setup_sql = (
        "CREATE DATABASE IF NOT EXISTS test;\n"
        "CREATE USER IF NOT EXISTS admin@'%' IDENTIFIED BY 'local-regression';\n"
        "ALTER USER admin@'%' IDENTIFIED BY 'local-regression';\n"
        "GRANT ALL PRIVILEGES ON *.* TO admin@'%' WITH GRANT OPTION;\n"
    )
    started = time.monotonic()
    setup_status = 125
    try:
        with (output / 'setup.log').open('w') as stream:
            setup_status = subprocess.run(setup_command, input=setup_sql, text=True,
                                          stdout=stream, stderr=subprocess.STDOUT,
                                          timeout=30).returncode
    except subprocess.TimeoutExpired:
        setup_status = 124
    finally:
        write_json(output / 'setup.json', {'command': setup_command,
                   'exit_code': setup_status, 'duration_seconds': time.monotonic() - started})
    if setup_status:
        return 1
    prologue = output / 'connection.inc'
    prologue.write_text(
        'let $OBMYSQL_MS0 = 127.0.0.1;\nlet $OBMYSQL_MS0_DEV = 127.0.0.1;\n'
        'let $OBMYSQL_PORT = ' + str(port) + ';\nlet $OBMYSQL_USR = admin;\n'
        'let $OBMYSQL_PWD = ' + PASSWORD + ';\n'
        'let $OBSERVER_DIR = ' + str(ROOT / 'fixtures') + ';\n')
    env = dict(os.environ)
    env.update({'PWD': str(DEPLOY), 'OBMYSQL_MS0': '127.0.0.1',
                'OBMYSQL_MS0_DEV': '127.0.0.1', 'OBMYSQL_PORT': str(port),
                'OBMYSQL_USR': 'admin', 'OBMYSQL_PWD': PASSWORD,
                'OBSERVER_DIR': str(ROOT / 'fixtures'), 'TENANT': 'mysql', 'IS_BUSINESS': '0'})
    env['PATH'] = str(CLIENT.parent) + os.pathsep + env.get('PATH', '')
    for name in CASES:
        case_output = output / name
        case_output.mkdir()
        tmp = case_output / 'tmp'
        tmp.mkdir()
        argv = [str(CLIENT), '--no-defaults', '--host=127.0.0.1', '--port=' + str(port),
                '--user=admin', '--password=' + PASSWORD, '--database=test',
                '--basedir=' + str(DEPLOY) + '/',
                '--test-file=' + inputs[name]['test_file'],
                '--result-file=' + inputs[name]['result_file'],
                '--prologue=' + str(prologue), '--logdir=' + str(case_output),
                '--tmpdir=' + str(tmp), '--mark-progress']
        write_json(case_output / 'command.json', argv)
        started = time.monotonic()
        status = 125
        with (case_output / 'output.log').open('w') as stream:
            try:
                status = subprocess.run(argv, cwd=DEPLOY, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=600).returncode
            except subprocess.TimeoutExpired:
                status = 124
                stream.write('\nRunner: mysqltest exceeded 600 seconds.\n')
            except OSError as error:
                stream.write('\nRunner: ' + str(error) + '\n')
        unchanged = (sha256(Path(inputs[name]['test_file'])) == inputs[name]['test_sha256']
                     and sha256(Path(inputs[name]['result_file'])) == inputs[name]['result_sha256'])
        results[name] = dict(inputs[name], exit_code=status,
                             duration_seconds=time.monotonic() - started,
                             inputs_unchanged=unchanged, passed=status == 0 and unchanged,
                             command=argv)
        write_json(case_output / 'status.json', results[name])
        write_json(output / 'status.json', results)
        print(name + ': ' + ('PASS' if results[name]['passed'] else 'FAIL') +
              ' (exit=' + str(status) + ')', flush=True)
    passed = len(results) == len(CASES) and all(v['passed'] for v in results.values())
    write_json(output / 'summary.json', {'passed': passed, 'expected_count': len(CASES),
               'executed_count': len(results),
               'passed_count': sum(v['passed'] for v in results.values())})
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
