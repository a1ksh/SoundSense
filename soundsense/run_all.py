"""Verify the existing YAMNet dataset, artifacts and local application."""
import argparse
from .common import config
from .download_data import prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-download',action='store_true')
    args=parser.parse_args()
    prepare(config(),skip_download=args.skip_download)
    from experiments.accuracy_v2.check import main as check
    check()


if __name__=='__main__':
    main()
