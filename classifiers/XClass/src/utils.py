#!/usr/bin/env python3

# XClass Classifier - Utilities
# Rewritten by Christophe Leung based on https://github.com/ZihanWangKi/XClass
# November 13, 2023

# Tested on Python 3.10.7; should be compatible between v3.6 through v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
import logging, sys, argparse, time

# =================================================================================================

# -----------------------------------------------------------------------------
# handles the main program's bootstrapping
class Bootstrap:
    @staticmethod
    def logging_init():
        logging.basicConfig(
            filename='run.log', 
            filemode='w', 
            format='%(asctime)s - [%(levelname)s] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S', 
            level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info('Logger initialized...')
        return logging

    @staticmethod
    def argparse_init():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=False)
        args = parser.parse_args()
        return args


# -----------------------------------------------------------------------------
# handles the main program's timers
class Timer:
    start, finish, total = -1, -1, -1

    @staticmethod
    def start():
        Timer.start = time.time()
        return Timer.start

    @staticmethod
    def finish():
        Timer.finish = time.time()
        Timer.total = Timer.finish - Timer.start
        return Timer.finish

