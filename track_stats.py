import csv
import os
import data_parser as lp
import numpy as np
from dataset_description import *


def create_track_count_histogram(tracks):
    histogram = {}
    for data in tracks.values():
        count = int(data[0])
        if count in histogram:
            histogram[count] += 1
        else:
            histogram[count] = 1
    return histogram


def create_track_skip_histogram(tracks):
    histogram = {}
    for data in tracks.values():
        # skip 2 is the target value in the challenge
        # just whole percents
        percents = np.floor(100.0 * float(data[2]) / float(data[0]))
        if percents in histogram:
            histogram[percents] += 1
        else:
            histogram[percents] = 1
    return histogram


tracks = {}
session_lengths = {i: 0 for i in range(21)}
cur_sess_id = None

print('starting track stats')

for filename in os.listdir("."):
    print('processing file ' + filename)
    if filename.endswith('.csv'):
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                lp.DataParser.append_track_data(row, tracks)
                id = row[SessionFeaturesFields.SESSION_ID]
                if cur_sess_id != id:
                    session_lengths[int(row[SessionFeaturesFields.SESSION_LENGTH])] += 1
                    cur_sess_id = id

count_histogram = create_track_count_histogram(tracks)
skip_histogram = create_track_skip_histogram(tracks)

lngths = open('session_lengths.txt', 'w+')
for length, count in session_lengths.items():
    lngths.write(str(length) + ' ' + str(count))
    lngths.write(os.linesep)
lngths.close()

counts = open('track_counts.txt', 'w+')
for sessions, count in count_histogram.items():
    counts.write(str(sessions) + ' ' + str(count))
    counts.write(os.linesep)
counts.close()

skips = open('track_skips.txt', 'w+')
for percents, count in skip_histogram.items():
    skips.write(str(percents) + ' ' + str(count))
    skips.write(os.linesep)
skips.close()

print('done')
