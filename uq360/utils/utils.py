
import os
import tempfile
import shutil
import atexit
import time
import subprocess
import numpy as np
from sklearn.base import TransformerMixin
import tensorflow_hub as hub


class Timer:
    def __init__(self):
        self.time_started = {}
        self.elapsed = {}
        self.elapsed['total'] = 0
        
    def start(self, name):
        self.time_started[name] = time.time()

    def stop(self, name):
        elapsed = time.time() - self.time_started[name]
        del self.time_started[name]

        self.elapsed[name] = elapsed
        self.elapsed['total'] += elapsed

    def report(self):
        print('Timings:')
        for k, v in self.elapsed.items():
            print (' ', k,v)

        for k,v in self.time_started.items():
            print ('Warning: timer started but never stopped', k)


class UseTransformer(TransformerMixin):
    '''
    Wrapper to run the Universal Sentence Embeddings (USE) encoder.
    Organizes the USE into the fit, transform and fit_transform standard methods of TransformerMixin.
    '''
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        return encoder(X.ravel()).numpy()


def new_tmp_file_or_dir(dir=False):
    tmp_path = tempfile.mkstemp()[1]
    if dir:
        os.remove(tmp_path)
        os.mkdir(tmp_path)

    def delete_tmp_file():
        rm_rf(tmp_path)

    atexit.register(delete_tmp_file)
    return tmp_path


def chmod_r(path, mode):
    """ Recursive chmod """
    os.chmod(path, mode)

    def try_chmod(filepath, mode):
        try:
            os.chmod(filepath, mode)
        except Exception:
            # potentially a symlink where we cannot chmod the target
            if not os.path.islink(filepath):
                raise

    for root, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            try_chmod(os.path.join(root, dirname), mode)
        for filename in filenames:
            try_chmod(os.path.join(root, filename), mode)


def rm_rf(path):
    """
    Recursively removes a file or directory
    """
    if not path or not os.path.exists(path):
        return
    # Make sure all files are writeable and dirs executable to remove
    chmod_r(path, 0o777)
    if os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)


def new_tmp_dir():
    return new_tmp_file_or_dir(dir=True)


def short_uuid():
    import uuid
    return str(uuid.uuid4())[0:8]


def execute(cmd, env=None, pipe_output=True, wait=True):
    env = env or {}
    env['PATH'] = env.get('PATH') or os.environ.get('PATH')

    kwargs = {
        'shell': True,
        'env': env
    }

    if not pipe_output:
        return subprocess.check_output(cmd, **kwargs)
    process = subprocess.Popen(cmd, bufsize=-1, **kwargs)
    if wait:
        result = process.wait()
        return result
    return process
