"""Unit tests for modules.workflow_common.start_run()/TeeStream (RC-02)."""

import sys
from pathlib import Path

from modules import workflow_common as wc


def test_start_run_creates_cmd_and_log_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['fake_script.py', '--foo', 'bar'])
    ctx = wc.start_run(Path('/some/fake/script.py'), 'my_output', provenance_dir=tmp_path)
    try:
        assert ctx.cmd_file.exists()
        assert ctx.log_file.exists()
        assert ctx.cmd_file.parent == tmp_path
        assert ctx.cmd_file.name == f'run_script_my_output_{ctx.run_ts}.cmd'
        assert ctx.log_file.name == f'run_script_my_output_{ctx.run_ts}.log'
    finally:
        ctx.close()


def test_cmd_file_contains_exact_quoted_argv_with_space(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['fake_script.py', '--path', 'a value with spaces'])
    ctx = wc.start_run(Path('/some/fake/script.py'), 'out', provenance_dir=tmp_path)
    try:
        content = ctx.cmd_file.read_text()
        assert "--path 'a value with spaces'" in content
        assert content.startswith(f'# timestamp={ctx.run_ts}\n')
        assert '# cwd=' in content
    finally:
        ctx.close()


def test_print_after_start_run_goes_to_both_console_and_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['fake_script.py'])
    ctx = wc.start_run(Path('/some/fake/script.py'), 'out', provenance_dir=tmp_path)
    try:
        print('hello from test')
    finally:
        ctx.close()

    captured = capsys.readouterr()
    assert 'hello from test' in captured.out

    log_content = ctx.log_file.read_text()
    assert 'hello from test' in log_content
    # The two provenance announcement lines are also always logged.
    assert '[provenance] cmd:' in log_content
    assert '[provenance] log:' in log_content


def test_close_restores_original_stdout_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['fake_script.py'])
    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    ctx = wc.start_run(Path('/some/fake/script.py'), 'out', provenance_dir=tmp_path)
    assert sys.stdout is not orig_stdout
    assert sys.stderr is not orig_stderr

    ctx.close()
    assert sys.stdout is orig_stdout
    assert sys.stderr is orig_stderr


def test_context_manager_restores_streams_on_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['fake_script.py'])
    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    with wc.start_run(Path('/some/fake/script.py'), 'out', provenance_dir=tmp_path):
        assert sys.stdout is not orig_stdout

    assert sys.stdout is orig_stdout
    assert sys.stderr is orig_stderr


def test_default_provenance_dir_is_cwd_provenance_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['fake_script.py'])
    monkeypatch.chdir(tmp_path)
    ctx = wc.start_run(Path('/some/fake/script.py'), 'out')
    try:
        assert ctx.cmd_file.parent == tmp_path / 'provenance_logs'
    finally:
        ctx.close()


def test_tee_stream_writes_to_all_streams():
    class FakeStream:
        def __init__(self):
            self.buf = []

        def write(self, data):
            self.buf.append(data)
            return len(data)

        def flush(self):
            self.buf.append('<flush>')

    s1, s2 = FakeStream(), FakeStream()
    tee = wc.TeeStream(s1, s2)
    tee.write('hello')
    tee.flush()

    assert s1.buf == ['hello', '<flush>']
    assert s2.buf == ['hello', '<flush>']
