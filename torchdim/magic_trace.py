from contextlib import contextmanager
import os
import subprocess
import signal
import time

@contextmanager
def magic_trace(output='trace.fxt', magic_trace_cache='/tmp/magic-trace'):
    pid = os.getpid()
    if not os.path.exists(magic_trace_cache):
        print(f"Downloading magic_trace to: {magic_trace_cache}")
        subprocess.run(['wget', '-O', magic_trace_cache,  '-q', 'https://github.com/janestreet/magic-trace/releases/download/v1.0.2/magic-trace'])
        subprocess.run(['chmod', '+x', magic_trace_cache])
    args = [magic_trace_cache, 'attach', '-pid', str(pid), '-o', output]
    p = subprocess.Popen(args, stderr=subprocess.PIPE, encoding='utf-8')
    while True:
        x = p.stderr.readline()
        print(x)
        if 'Attached' in x:
            break
    try:
        yield
    finally:
        p.send_signal(signal.SIGINT)
        r = p.wait()
        print(p.stderr.read())
        p.stderr.close()
        if r != 0:
            raise ValueError(f'magic_trace exited abnormally: {r}')