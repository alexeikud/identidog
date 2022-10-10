"""
Main script for gradio app identidog.
We define the app's execution in the function "run_app",
then define app atrributes before building app with the gradio
library (see https://www.gradio.app/docs/).

Notes:
- Script is dependent on pre-defined functions and models from "identifiers.py"
- Example images are loaded from the folder "img_examples"
"""

# ## Import necessary libraries
import gradio as gr
import identifiers as ids

# ## Defining app logic


def run_app(img_path: str) -> tuple[str, dict]:
    """
    App predicts most resembling dog breeds from image of human or dog.
    Logic handles cases for a human, dog, both or neither.

    Inputs:
        img_path = path to image
    Outputs:
        detection:  Output display message stored as string
        preds:      Dictionary of prediction confidences with items
                      (breed labels: probabilities).
    """
    # detect humans/ dogs
    face_detected = ids.face_detector(img_path)
    dog_detected = ids.dog_detector(img_path)

    # app logic:
    # if both human and dog detected
    # output error message
    if face_detected and dog_detected:
        detection = (
            "Both human and dog found! Please crop the image "
            "until only a person or dog are showing to obtain a prediction"
        )
        return detection, None
    # if neither human or dog detected:
    # output error message
    elif not face_detected and not dog_detected:
        detection = (
            "No human or dog found! Please zoom in by cropping "
            "or upload a new image."
        )
        return detection, None
    # if only a human OR a dog detected:
    # run breed identifier and output relevant message.
    else:
        preds = ids.breed_identifier(img_path)
        detection = (
            "Human detected! Your identification results are:"
            if face_detected
            else "Dog detected! The most resembling breeds are:"
        )
        return detection, preds


# ## Creating GUI with Gradio

# ### Defining Inputs
# app title
title = "Identidog"

# app main description
description = """
Upload an image of a dog or human, or try an example from below. \
The app returns the top 3 dog breeds they resemble and the corresponding \
confidences!

**Note**: Cropped or zoomed-in images showing only a dog or human \
yield better predictions.
"""

# app extra info
article = """

___

<p style="text-align: center;">
</li>
<a href="https://github.com/alexeikud/identidog"
target="_blank">
Github Repo<a>
</p>
"""
# app inputs
inputs = gr.components.Image(type="filepath", shape=(512, 512), label="image")

# app outputs
outputs = [
    gr.components.Textbox(label="Output:"),
    gr.components.Label(label="Identification results", num_top_classes=3),
]

# input image examples
examples = [
    "img_examples/cosmo_smart.png",
    "img_examples/Nessy.jpg",
    "img_examples/sleepy_dog.jpg",
    "img_examples/romanian.jpg",
    "img_examples/lokii.jpg",
    "img_examples/monty.jpg",
]


# ### Launching app
app = gr.Interface(
    fn=run_app,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging="never",
)
app.launch()
