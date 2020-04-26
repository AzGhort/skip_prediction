import csv
import os
import data_parser as dp
import numpy as np
from dataset_description import *


def create_track_count_histogram(tracks):
    histogram = {}
    for data in tracks.values():
        count = int(data[0])
        try_increase(histogram, count)
    return histogram


def create_track_skip_histogram(tracks):
    histogram = {}
    for data in tracks.values():
        # skip 2 is the target value in the challenge
        # just whole percents
        percents = np.floor(100.0 * float(data[2]) / float(data[0]))
        try_increase(histogram, percents)
    return histogram


def append_to_enums_histograms(histograms, row):
    try_increase(histograms[SessionFeaturesFields.CONTEXT_TYPE], row[SessionFeaturesFields.CONTEXT_TYPE])
    try_increase(histograms[SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_END], row[SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_END])
    try_increase(histograms[SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_START], row[SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_START])
    try_increase(histograms[SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKFWD], row[SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKFWD])
    try_increase(histograms[SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKBACK], row[SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKBACK])
    try_increase(histograms[SessionFeaturesFields.HOUR_OF_DAY], row[SessionFeaturesFields.HOUR_OF_DAY])


def try_increase(histogram, key):
    if key in histogram:
        histogram[key] += 1
    else:
        histogram[key] = 1


def write_histogram(histogram, filename):
    file = open(filename + '.txt', 'w+')
    for key, value in histogram.items():
        file.write(str(key) + ' ' + str(value))
        file.write(os.linesep)
    file.close()


enum_histograms = { SessionFeaturesFields.CONTEXT_TYPE: {},
                    SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_END: {},
                    SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_START: {},
                    SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKFWD: {},
                    SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKBACK: {},
                    SessionFeaturesFields.HOUR_OF_DAY: {}}
tracks = {}
session_lengths = {i: 0 for i in range(21)}
cur_sess_id = None

print('computing session logs feature stats')

for filename in os.listdir("."):
    print('processing file ' + filename)
    if filename.endswith('.csv'):
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dp.DataParser.append_track_data(row, tracks)
                id = row[SessionFeaturesFields.SESSION_ID]
                append_to_enums_histograms(enum_histograms, row)
                if cur_sess_id != id:
                    session_lengths[int(row[SessionFeaturesFields.SESSION_LENGTH])] += 1
                    cur_sess_id = id

count_histogram = create_track_count_histogram(tracks)
skip_histogram = create_track_skip_histogram(tracks)

enum: str
for enum in enum_histograms.keys():
    write_histogram(enum_histograms[enum], enum)

write_histogram(session_lengths, 'session_lengths')
write_histogram(count_histogram, 'track_counts')
write_histogram(skip_histogram, 'track_skips')

print('done')
