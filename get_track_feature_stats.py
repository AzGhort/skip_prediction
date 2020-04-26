import track_feature_parser as tfp
import numpy as np
import os


def create_histograms(map):
    duration = {}
    release_year = {}
    us_pop_est = {}
    acousticness = {}
    beat = {}
    bounciness = {}
    danceability = {}
    dyn_range_mean = {}
    energy = {}
    flatness = {}
    instrumentalness = {}
    key = {}
    liveness = {}
    loudness = {}
    mechanism = {}
    mode = {}
    organism = {}
    speechiness = {}
    tempo = {}
    time_signature = {}
    valence = {}

    histograms = {'duration': duration, 'release_year': release_year, 'us_pop_estimate': us_pop_est,
              'acousticness': acousticness, 'beat': beat, 'bounciness': bounciness, 'danceability': danceability,
              'dyn_range_mean': dyn_range_mean, 'energy': energy, 'flatness': flatness,
              'instrumentalness': instrumentalness,
              'key': key, 'liveness': liveness, 'loudness': loudness, 'mechanism': mechanism, 'mode': mode,
              'organism': organism, 'speechiness': speechiness, 'tempo': tempo, 'time_signature': time_signature,
              'valence': valence}

    for track_id, features in map.items():
        append_value_to_histogram(duration, 0, features[0])
        append_value_to_histogram(release_year, 0, features[1])
        append_value_to_histogram(us_pop_est, 0, features[2])
        append_value_to_histogram(acousticness, 2, features[3])
        append_value_to_histogram(beat, 2, features[4])
        append_value_to_histogram(bounciness, 2, features[5])
        append_value_to_histogram(danceability, 2, features[6])
        append_value_to_histogram(dyn_range_mean, 2, features[7])
        append_value_to_histogram(energy, 2, features[8])
        append_value_to_histogram(flatness, 2, features[9])
        append_value_to_histogram(instrumentalness, 2, features[10])
        append_value_to_histogram(key, 0, features[11])
        append_value_to_histogram(liveness, 2, features[12])
        append_value_to_histogram(loudness, 0, features[13])
        append_value_to_histogram(mechanism, 2, features[14])
        append_value_to_histogram(mode, 0, features[15])
        append_value_to_histogram(organism, 2, features[16])
        append_value_to_histogram(speechiness, 2, features[17])
        append_value_to_histogram(tempo, 0, features[18])
        append_value_to_histogram(time_signature, 0, features[19])
        append_value_to_histogram(valence, 2, features[20])
    return histograms


def append_value_to_histogram(histogram, decimals, value):
    rounded = np.around(value, decimals)
    if rounded in histogram:
        histogram[rounded] += 1
    else:
        histogram[rounded] = 1


def write_histograms(histograms):
    for name, histogram in histograms.items():
        file = open(name + '.txt', 'w+')
        for key, count in histogram.items():
            file.write(str(key) + ' ' + str(count))
            file.write(os.linesep)
        file.close()


def append_features_stats(map, feature_values):
    map['std'] = np.std(feature_values)
    map['mean'] = np.mean(feature_values)
    map['first_quartile'] = np.percentile(feature_values, 25)
    map['median'] = np.percentile(feature_values, 50)
    map['third_quartile'] = np.percentile(feature_values, 75)
    map['minimum'] = np.min(feature_values)
    map['maximum'] = np.max(feature_values)


def create_stats(map):
    stats = [[] for _ in range(21)]

    for track_id, features in map.items():
        for i in range(21):
            stats[i].append(features[i])

    duration = {}
    release_year = {}
    us_pop_est = {}
    acousticness = {}
    beat = {}
    bounciness = {}
    danceability = {}
    dyn_range_mean = {}
    energy = {}
    flatness = {}
    instrumentalness = {}
    key = {}
    liveness = {}
    loudness = {}
    mechanism = {}
    mode = {}
    organism = {}
    speechiness = {}
    tempo = {}
    time_signature = {}
    valence = {}

    append_features_stats(duration, stats[0])
    append_features_stats(release_year, stats[1])
    append_features_stats(us_pop_est, stats[2])
    append_features_stats(acousticness, stats[3])
    append_features_stats(beat, stats[4])
    append_features_stats(bounciness, stats[5])
    append_features_stats(danceability, stats[6])
    append_features_stats(dyn_range_mean, stats[7])
    append_features_stats(energy, stats[8])
    append_features_stats(flatness, stats[9])
    append_features_stats(instrumentalness, stats[10])
    append_features_stats(key, stats[11])
    append_features_stats(liveness, stats[12])
    append_features_stats(loudness, stats[13])
    append_features_stats(mechanism, stats[14])
    append_features_stats(mode, stats[15])
    append_features_stats(organism, stats[16])
    append_features_stats(speechiness, stats[17])
    append_features_stats(tempo, stats[18])
    append_features_stats(time_signature, stats[19])
    append_features_stats(valence, stats[20])

    return {'duration': duration, 'release_year': release_year, 'us_pop_estimate': us_pop_est,
                  'acousticness': acousticness, 'beat': beat, 'bounciness': bounciness, 'danceability': danceability,
                  'dyn_range_mean': dyn_range_mean, 'energy': energy, 'flatness': flatness,
                  'instrumentalness': instrumentalness,
                  'key': key, 'liveness': liveness, 'loudness': loudness, 'mechanism': mechanism, 'mode': mode,
                  'organism': organism, 'speechiness': speechiness, 'tempo': tempo, 'time_signature': time_signature,
                  'valence': valence}


def write_stats(stats):
    file = open('track_feature_stats.txt', 'w+')
    for feature_name, feature_stats in stats.items():
        file.write(str(feature_name) + ' ')
        for statistic in feature_stats.values():
            file.write(str(statistic) + ' ')
        file.write(os.linesep)
    file.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="tf", type=str)
    args = parser.parse_args()
    print("Getting track features.")
    track_features_map = tfp.TrackFeatureParser.get_track_features(args.folder)
    print("Creating histograms.")
    histograms = create_histograms(track_features_map)
    write_histograms(histograms)
    print("Computing stats.")
    stats = create_stats(track_features_map)
    write_stats(stats)
    print("Done")
