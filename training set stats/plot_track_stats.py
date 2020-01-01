import chart_studio.plotly as py
import plotly.graph_objects as go
import numpy as np


def read_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        temp = filter(None, (line.rstrip() for line in file))
        for line in temp:
            line = line.replace(",", ".")
            parts = line.split(" ")
            data.append((float(parts[0]), float(parts[1])))
    data.sort(key=lambda t: t[0])
    res = list(zip(*data))
    X = res[0]
    Y = res[1]
    return X, Y


def create_plotly_graph(x_name, y_name, file_name, graph_name, data):
    layout = dict(title=graph_name, autosize=True, titlefont=dict(size=32),
                  legend=dict(font=dict(size=18)),
                  xaxis=dict(title=x_name, tickfont=dict(size=16), zerolinewidth=1,
                             showgrid=True, type='linear', ticklen=6, showline=True, zeroline=True,
                             linewidth=1.5, ticks="inside", gridwidth=1, autorange=True, titlefont=dict(size=24)),
                  yaxis=dict(title=y_name, tickfont=dict(size=16), zerolinewidth=1,
                             showgrid=True, type='linear', ticklen=6, showline=True, zeroline=True,
                             linewidth=1.5, ticks="inside", gridwidth=1, autorange=True, titlefont=dict(size=24)))
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename=file_name)


def preprocess_track_frequency(track_ses_count, track_count, total_ses):
    histogram = {}
    for ses, trak in zip(track_ses_count, track_count):
        perc = np.round(100.0 * ses / total_ses, decimals=2)
        if perc in histogram:
            histogram[perc] += 1
        else:
            histogram[perc] = 1
    return list(histogram.keys()), list(histogram.values())


[ses_lng, ses_cnt] = read_data("session_lengths.txt")
[track_ses_cnt, track_cnt] = read_data("track_counts.txt")
[percents, tracks_skipped] = read_data("track_skips.txt")

ses_perc, track_cnt = preprocess_track_frequency(track_ses_cnt, track_cnt, sum(ses_cnt))

trace1 = go.Scatter(x=ses_lng, y=ses_cnt, name="Number of sessions of different lengths", line=dict(width=2))
data1 = [trace1]
create_plotly_graph("Session length", "Count", "Session lengths", "Session lengths", data1)

trace2 = go.Scatter(x=ses_perc, y=track_cnt, name="Frequency of tracks in sessions", line=dict(width=2))
data2 = [trace2]
create_plotly_graph("Percents of sessions", "Number of tracks", "Tracks frequency", "Tracks frequency", data2)

trace3 = go.Scatter(x=percents, y=tracks_skipped, name="Skip ratio of tracks", line=dict(width=2))
data3 = [trace3]
create_plotly_graph("Skip percents", "Number of tracks", "Skip ratio", "Skip ratio", data3)


