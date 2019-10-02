import visdom
import time
import csv
import os
import pandas as pd
import argparse
import ntpath
import glob
import colorlover as cl
import random

# TODO: Also extend to check models. Probably by path?
parser = argparse.ArgumentParser(description='Visdom server Arguments')
parser.add_argument('--port',
                    metavar='p',
                    type=int,
                    required=True,
                    help='Port the visdom server is running on')
parser.add_argument('--root',
                    metavar='r',
                    type=str,
                    default='../results/',
                    help='Where to look for CSV files')
parser.add_argument('--interval',
                    metavar='s',
                    type=int,
                    default=10,
                    help='Number of seconds to wait before refreshing data')

FLAGS = parser.parse_args()

# Default running on localhost
vis = visdom.Visdom(port=FLAGS.port)


def color_picker(index):
    colors = cl.scales['9']['qual']['Set1']
    if index in color_picker.color_map:
        return colors[color_picker.color_map[index]]
    color_picker.color_map[index] = color_picker.counter
    color_picker.counter += 1
    color_picker.counter %= len(colors)
    return colors[color_picker.color_map[index]]


color_picker.counter = 0
color_picker.color_map = {}


def fetch_all_traces():
    """
        Find all the csv files in a folder and generate a dict of list of traces.
        dict keys: 'grad_iter_trace', 'residual_iter_trace', 'grad_time_trace', 'residual_time_trace'
    """
    all_traces = {'residual_iter_trace': [], 'residual_time_trace': []}
    # for all files
    for filename in glob.glob(os.path.join(FLAGS.root, '*.csv')):
        try:
            traces = generate_trace_from_csv(filename)
            for k, v in all_traces.items():
                v.append(traces[k])
        except BaseException as err:
            print('Ignoring error: ', err)

    return all_traces


def generate_trace_from_csv(filename):
    """
        Return a list of traces
    """
    # filename = '../results/out.csv'

    df = pd.read_csv(filename, sep=",")
    basename = ntpath.basename(filename)
    model_prefix = (os.path.splitext(basename)[0])

    color = color_picker(abs(hash(model_prefix)))
    residual_iter_trace = dict(x=df['iterations'].tolist(),
                               y=df['residual'].tolist(),
                               mode="markers+lines",
                               type='custom',
                               marker={
                                   'color': color,
                                   'symbol': 104,
                                   'size': "1"
                               },
                               name=model_prefix)
    residual_time_trace = dict(x=df['time'].tolist(),
                               y=df['residual'].tolist(),
                               mode="markers+lines",
                               type='custom',
                               marker={
                                   'color': color,
                                   'symbol': 104,
                                   'size': "1"
                               },
                               name=model_prefix)
    return {
        'residual_iter_trace': residual_iter_trace,
        'residual_time_trace': residual_time_trace
    }


while True:

    all_traces = fetch_all_traces()
    residual_iter_layout = dict(title="residual vs iterations",
                                xaxis={'title': 'iterations'},
                                yaxis={
                                    'type': 'log',
                                    'title': 'residual'
                                })
    residual_time_layout = dict(title="residual vs time",
                                xaxis={'title': 'time'},
                                yaxis={
                                    'type': 'log',
                                    'title': 'residual'
                                })

    vis._send({
        'data': all_traces['residual_iter_trace'],
        'layout': residual_iter_layout,
        'win': 'win2'
    })
    vis._send({
        'data': all_traces['residual_time_trace'],
        'layout': residual_time_layout,
        'win': 'win4'
    })
    time.sleep(FLAGS.interval)
