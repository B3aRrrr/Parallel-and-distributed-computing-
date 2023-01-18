def saveToHTML(name):
    import os
    results = r'/content/results'
    if not os.path.exists(results):
        os.makedirs(results)

    plotsDir = f'{results}/plots'
    htmlDir = f'{results}/html'
    if not os.path.exists(plotsDir):
        os.makedirs(plotsDir)
    if not os.path.exists(htmlDir):
        os.makedirs(htmlDir)

    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.offline as offline

    with open(f'/content/results/{name}/train_losses.txt', 'r') as file:
        train_losses = list(map(float, file.read().replace('\n', '').replace('[', '').replace(']', '').split(sep=', ')))

    with open(f'/content/results/{name}/train_counter.txt', 'r') as file:
        train_counter = list(
            map(float, file.read().replace('\n', '').replace('[', '').replace(']', '').split(sep=', ')))

    with open(f'/content/results/{name}/test_losses.txt', 'r') as file:
        test_losses = list(map(float, file.read().replace('\n', '').replace('[', '').replace(']', '').split(sep=', ')))

    with open(f'/content/results/{name}/test_counter.txt', 'r') as file:
        test_counter = list(map(float, file.read().replace('\n', '').replace('[', '').replace(']', '').split(sep=', ')))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train_counter, y=train_losses))

    fig.add_scatter(
        x=test_counter,
        y=test_losses,
        mode="markers",
        marker=dict(size=20, color="LightSeaGreen"),
        # name="a",
        # row=i+1, col=1
    )

    offline.plot(fig, filename=f'{htmlDir}/{name}.html', auto_open=False)


def showPlotly(name):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.offline as offline

    import numpy as np

    with open(f'/content/results/{name}/train_losses.txt', 'r') as file:
        train_losses = list(map(float, file.read().replace('\n', '').replace('[', '').replace(']', '').split(sep=', ')))

    with open(f'/content/results/{name}/train_counter.txt', 'r') as file:
        train_counter = list(
            map(float, file.read().replace('\n', '').replace('[', '').replace(']', '').split(sep=', ')))

    with open(f'/content/results/{name}/test_losses.txt', 'r') as file:
        test_losses = list(map(float, file.read().replace('\n', '').replace('[', '').replace(']', '').split(sep=', ')))

    with open(f'/content/results/{name}/test_counter.txt', 'r') as file:
        test_counter = list(map(float, file.read().replace('\n', '').replace('[', '').replace(']', '').split(sep=', ')))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train_counter, y=train_losses))

    fig.add_scatter(
        x=test_counter,
        y=test_losses,
        mode="markers",
        marker=dict(size=20, color="LightSeaGreen"),
        # name="a",
        # row=i+1, col=1
    )
    fig.show()


def txtForPlot(
        train_losses,  # = dict_of_models[name]['train_losses']
        train_counter,  # = dict_of_models[name]['train_counter']
        test_losses,  # = dict_of_models[name]['test_losses']
        test_counter,  # = dict_of_models[name]['test_counter']
        pathToSave='/content/results',
        name='SGD'):
    with open(f'{pathToSave}/{name}/train_losses.txt', "w") as output:
        output.write(str(train_losses))
    with open(f'{pathToSave}/{name}/train_counter.txt', "w") as output:
        output.write(str(train_counter))
    with open(f'{pathToSave}/{name}/test_losses.txt', "w") as output:
        output.write(str(test_losses))
    with open(f'{pathToSave}/{name}/test_counter.txt', "w") as output:
        output.write(str(test_counter))

        
def make_model_optimizer_dict(model,Optimizer,train_loader,n_epochs,learning_rate=0.001):
  return {
        "model":model,
        "optimizer":Optimizer(model.parameters(), lr=learning_rate),
        "train_losses" : [],
        "train_counter" : [],
        "test_losses" : [],
        "test_counter" : [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    }


def s_to_bitlist(s):
      ords = (ord(c) for c in s)
      shifts = (7, 6, 5, 4, 3, 2, 1, 0)
      return [(o >> shift) & 1 for o in ords for shift in shifts]
def bitlist_to_chars(bl):
  bi = iter(bl)
  bytes = zip(*(bi,) * 8)
  shifts = (7, 6, 5, 4, 3, 2, 1, 0)
  for byte in bytes:
    yield chr(sum(bit << s for bit, s in zip(byte, shifts)))

def bitlist_to_s(bl):
  return ''.join(bitlist_to_chars(bl))

