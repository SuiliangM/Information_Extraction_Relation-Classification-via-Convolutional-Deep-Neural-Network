#!/bin/sh
# set the dependence 
TMPDIR=./ /opt/python-3.6/bin/python3 -m pip install --cache-dir=./ --user torch
TMPDIR=./ /opt/python-3.6/bin/python3 -m pip install --cache-dir=./ --user tb-nightly
TMPDIR=./ /opt/python-3.6/bin/python3 -m pip install --cache-dir=./ --user future

/opt/python-3.6/bin/python3 main.py --train_filename data/2010/merge_bin_2010/train/train-1 --test_filename data/2010/merge_bin_2010/test/test-1
/opt/python-3.6/bin/python3 main.py --train_filename data/2010/merge_bin_2010/train/train-2 --test_filename data/2010/merge_bin_2010/test/test-2
/opt/python-3.6/bin/python3 main.py --train_filename data/2010/merge_bin_2010/train/train-3 --test_filename data/2010/merge_bin_2010/test/test-3
/opt/python-3.6/bin/python3 main.py --train_filename data/2010/merge_bin_2010/train/train-4 --test_filename data/2010/merge_bin_2010/test/test-4
/opt/python-3.6/bin/python3 main.py --train_filename data/2010/merge_bin_2010/train/train-5 --test_filename data/2010/merge_bin_2010/test/test-5
/opt/python-3.6/bin/python3 main.py --train_filename data/2010/merge_bin_2010/train/train-6 --test_filename data/2010/merge_bin_2010/test/test-6
/opt/python-3.6/bin/python3 main.py --train_filename data/2010/merge_bin_2010/train/train-7 --test_filename data/2010/merge_bin_2010/test/test-7
