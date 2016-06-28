import os
import tempfile
import subprocess


def call_or(command, complain=False, error=False):
    """Call a shell function and either complain or raise an exception on nonzero exit status."""
    p = subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if error and p != 0:
        raise Exception('Command returned %d: %s' % (p, command))

    if complain and p != 0:
        print('Command returned %d: %s' % (p, command))


def maybe_mkdir(dirpath):
    """Maybe make a directory"""
    if type(dirpath) is not str:
        dirpath = tempfile.mkdtemp()
    elif not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath
