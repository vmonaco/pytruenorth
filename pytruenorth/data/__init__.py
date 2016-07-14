
import os
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')

from .mnist import load as load_mnist
from .cmu import load as load_cmu
from .biosig import load as load_biosig