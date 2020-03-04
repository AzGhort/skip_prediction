import chart_studio.plotly as py
import plotly.graph_objects as go


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
    fig = go.Figure(data=data, layout=layout)

    py.plot(fig, filename=file_name)


def create_graph_from_file(file_name, graph_name, x_name, feature_stats):
    [x, y] = read_data(file_name)
    trace = go.Scatter(x=x, y=y, name=graph_name, line=dict(width=2), mode='lines')

    trace_median = go.Scatter(x=[feature_stats['median'], feature_stats['median']],
                              y=[min(y), max(y)], name="Median", line=dict(width=2, color='midnightblue'), mode='lines')
    trace_first_quartile = go.Scatter(x=[feature_stats['first_quartile'], feature_stats['first_quartile']],
                                      y=[min(y), max(y)], line=dict(width=2, color="mediumblue"),
                                      name="First quartile", mode='lines')
    trace_third_quartile = go.Scatter(x=[feature_stats['third_quartile'], feature_stats['third_quartile']],
                                      y=[min(y), max(y)], mode='lines', line=dict(width=2, color="deepskyblue"),
                                      name="Third quartile")
    trace_mean = go.Scatter(x=[feature_stats['mean'], feature_stats['mean']], y=[min(y), max(y)],
                            mode='lines', line=dict(width=2, color="lightskyblue"), name="Mean")

    traces = [trace, trace_first_quartile, trace_median, trace_third_quartile, trace_mean]
    create_plotly_graph(x_name, "Count", graph_name, graph_name, traces)


def get_stats():
    stats = {}
    with open('track_feature_stats.txt', 'r', encoding='utf-8') as file:
        temp = filter(None, (line.rstrip() for line in file))
        for line in temp:
            line = line.replace(",", ".")
            parts = line.split(" ")
            stats[parts[0]] = {'mean': float(parts[2]), 'first_quartile': float(parts[3]), 'median': float(parts[4]),
                               'third_quartile': float(parts[5]), 'std': float(parts[1])}
    return stats


stats = get_stats()
create_graph_from_file("acousticness.txt", "Acousticness", "Value", stats['acousticness'])
create_graph_from_file("beat.txt", "Beat strength", "Value", stats['beat'])
create_graph_from_file("bounciness.txt", "Bounciness", "Value", stats['bounciness'])
create_graph_from_file("danceability.txt", "Danceability", "Value", stats['danceability'])
create_graph_from_file("duration.txt", "Duration", "Seconds", stats['duration'])
create_graph_from_file("dyn_range_mean.txt", "Dynamic range mean", "Value", stats['dyn_range_mean'])
create_graph_from_file("energy.txt", "Energy", "Value", stats['energy'])
create_graph_from_file("flatness.txt", "Flatness", "Value", stats['flatness'])
create_graph_from_file("instrumentalness.txt", "Instrumentalness", "Value", stats['instrumentalness'])
create_graph_from_file("key.txt", "Key", "Key in pitch notation", stats['key'])
create_graph_from_file("liveness.txt", "Liveness", "Value", stats['liveness'])
create_graph_from_file("loudness.txt", "Loudness", "Average decibels", stats['loudness'])
create_graph_from_file("release_year.txt", "Release year", "Year", stats['release_year'])
create_graph_from_file("mode.txt", "Mode", "Major/Minor", stats['mode'])
create_graph_from_file("speechiness.txt", "Speechiness", "Value", stats['speechiness'])
create_graph_from_file("time_signature.txt", "Time signature", "Value", stats['time_signature'])
create_graph_from_file("tempo.txt", "Tempo", "BPM", stats['tempo'])
create_graph_from_file("us_pop_estimate.txt", "US popularity estimate", "Percentile as of October 2018", stats['us_pop_estimate'])
create_graph_from_file("valence.txt", "Valence", "Value", stats['valence'])
