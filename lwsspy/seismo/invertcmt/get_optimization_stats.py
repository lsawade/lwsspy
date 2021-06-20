import os
from glob import glob
import numpy as np


def get_optimization_stats(database):

    # Get all cmts in database
    cmts = os.listdir(database)

    summaryfiles = []
    for cmt in cmts:

        # Define summary filename
        summaryfile = os.path.join(database, cmt, 'summary.npz')

        # Check if exists and append if does
        if os.path.exists(summaryfile):
            summaryfiles.append(summaryfile)

    print("Number if inversions availables:", len(summaryfiles))

    # Summary
    iterations = []
    
    for summaryfile in summaryfiles:
        summary = np.load(summaryfile)
        iterations.append(len(summary['fcost_hist']))

    print("Average number of iterations:", np.mean(iterations))


def bin():

    import argparse

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest='data_database', type=str,
        help='Database that contains the downloaded data.')
    args = parser.parse_args()

    get_optimization_stats(args.data_database)
