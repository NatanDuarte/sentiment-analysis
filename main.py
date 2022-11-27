import dearpygui.dearpygui as dpg

import numpy as np
import pandas as pd

import nltk
from nltk import tokenize

from pickle import load
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer


labels = ['comentário negativo', 'comentário positivo']

with open("models/bayesian_model.pkl", "rb") as file:
    ml_model = load(file)
with open("models/bayesian_vectorizer.pkl", "rb") as file:
    vec = load(file)


def analise_text_callback(sender, data):
    input_text_tag = '__input_text'
    input_string = dpg.get_value(input_text_tag)
    if input_string != '':
        x = ml_model.predict(vec.transform([input_string]))[0]

        dpg.set_value(item='__output_0_text',
                      value=f'Sentença  : {input_string}')
        dpg.set_value(item='__output_1_text', value=f'Inferência: {labels[x]}')

        dpg.set_value(item=input_text_tag, value='')
        dpg.focus_item(item=input_text_tag)


width = 640
height = 400

dpg.create_context()
dpg.create_viewport(
    x_pos=0, y_pos=0,
    title="Sentiment Analysis",
    width=width, height=height,
    resizable=False)
dpg.setup_dearpygui()


def input_form(analise_text_callback):
    with dpg.group(horizontal=True):
        dpg.add_input_text(
            tag="__input_text",
            on_enter=True,
            hint='Digitar texto',
            default_value='',
            callback=analise_text_callback)
        dpg.add_button(label="analisar", callback=analise_text_callback)
    dpg.add_spacer(height=16)


with dpg.window(
    width=width,
    height=height,
    no_close=True,
    no_move=True,
    no_collapse=True
):
    dpg.add_spacer(height=16)
    input_form(analise_text_callback)
    dpg.add_separator()
    dpg.add_text(tag="__output_0_text")
    dpg.add_text(tag="__output_1_text")
    dpg.add_spacer(height=4)

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
