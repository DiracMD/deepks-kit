import os
import sys
try:
    import deepqc
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from deepqc.main import train_cli

if __name__ == "__main__":
    train_cli()