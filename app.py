import json
import pickle
import numpy as np
import pandas as pd
import gradio as gr

## Load the model
regmodel = pickle.load(open('air_quality.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

def predict_api(data):
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    return output[0]

def predict(data):
    data = [float(x) for x in data]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    output = regmodel.predict(final_input)[0]
    return f"Air Quality {output}"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_data = gr.Textbox(label="Input Data (JSON format)")
            predict_button = gr.Button("Predict")
            output_text = gr.Textbox(label="Prediction Output")

        predict_button.click(fn=predict_api, inputs=input_data, outputs=output_text)

    with gr.Row():
        input_data_form = gr.Textbox(label="Input Data (Form format)")
        predict_button_form = gr.Button("Predict")
        output_text_form = gr.Textbox(label="Prediction Output")

        predict_button_form.click(fn=predict, inputs=input_data_form, outputs=output_text_form)

demo.launch(debug=True)
