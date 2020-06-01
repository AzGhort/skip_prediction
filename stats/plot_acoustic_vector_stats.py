import chart_studio.plotly as py
import plotly.graph_objects as go


def read_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        temp = filter(None, (line.rstrip() for line in file))
        for line in temp:
            line = line.replace(",", ".")
            parts = line.split(" ")
            data.append((float(parts[0]) / 100.0, float(parts[1])))
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


def create_graph_from_file(file_name, graph_name, x_name):
    [x, y] = read_data(file_name)
    trace = go.Scatter(x=x, y=y, name=graph_name, line=dict(width=2), mode='lines')
    traces = [trace]
    create_plotly_graph(x_name, "Count", graph_name, graph_name, traces)


for i in range(8):
    vector_name = "acoustic_vector_" + str(i)
    graph_name = "Acoustic Vector " + str(i)
    create_graph_from_file(vector_name + ".txt", graph_name, "Value")
