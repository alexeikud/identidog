"""
This script defines all the identification functions the app relies on.
We use a full frontal Haarcascade model from opencv for face detection
and a pretrained VGG16 CNN from pytorch for dog detection.

We also import our pretrained CNN model and use it
to define an inference function for dog breed identification
given any path to an input image.
"""
# imports
import pathlib
from contextlib import contextmanager

import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from fastai.vision.all import PILImage, load_learner
from PIL import Image, ImageFile

# Set PIL to be tolerant of image files that are truncated.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Face detector
def face_detector(img_path: str) -> bool:
    """
    Using opencv's haar cascade classifier to detect human faces in images

    Inputs:
        img_path: path to an image of type string or path object

    Outputs:
        True or False depending on whether at least one face detected or not.
    """
    img = cv2.imread(img_path, 0)  # 0 flag for greyscale
    fd_model = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")
    faces = fd_model.detectMultiScale(img)
    return len(faces) > 0


# Inference model for dog detector
# Check for GPU and move model to GPU if available
use_cuda = torch.cuda.is_available()
dev = torch.device("cuda" if use_cuda else "cpu")

# load VGG16 model
dd_model = models.vgg16(pretrained=True).to(dev)


def dd_predict(img_path: str) -> int:
    """
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Inputs:
        img_path: path to an image

    Outputs:
        Integer index corresponding to VGG-16 model's prediction
    """
    # Load and pre-process an image from the given img_path
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img_t = preprocess(img)
    # create a mini-batch and load to gpu as expected by the model
    batch_t = torch.unsqueeze(img_t, 0).to(dev)
    # initialise model
    dd_model.eval()
    # Use the model and print the predicted category
    probs = dd_model(batch_t).squeeze(0).softmax(0)
    class_id = probs.argmax().item()
    return class_id  # predicted class index


# Set dog category indices from Imagenet1000 classes
DOG_INDICES = range(151, 269)


# dog detector
def dog_detector(img_path: str) -> bool:
    """
    Function returns "True" if a dog is detected in the image.

    Inputs:
        img_path: path to image as str/path.
    Outputs:
        Boolean True/False if dog detected or not.
    """
    image_index = dd_predict(img_path)
    return image_index in DOG_INDICES


# Loading CNN Model for dog breed classification

# Custom label function must be preloaded when using load_lerner
def get_breed_name(filepath: pathlib.Path) -> str:
    """
    Function to grab dog breed name from full pathname.
    The name can be obtained by dropping the last 10
    characters from filename and converting underscores
    to spaces.

    Input:
        filepath as Path object.
    Output:
        breed name as string.
    """
    return filepath.name[:-10].replace("_", " ")


# Defining function to deal with swapping posix to windows
# paths when on windows machine. Credit goes to to Jean Francois T.'s
# answer on Stackoverflow (https://stackoverflow.com/a/68796747)
@contextmanager
def set_path_windows():
    """
    Function switches from posixpath to windows path if necessary
    and returns back to posixpath.
    """
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


EXPORT_PATH = pathlib.Path("models/breed_model.pkl")


with set_path_windows():
    breeds_inf = load_learner(EXPORT_PATH, cpu=True)


# convert category names to more readable format for output in identification
breeds = tuple(breed.replace("_", " ") for breed in breeds_inf.dls.vocab)


# Breed Identifier
def breed_identifier(img_path: str) -> dict:
    """
    Function to identify dog breed from a possible of 133 categories in the
    udacity dog dataset. It takes an image path, and returns breed and
    probabilities for all breeds. We use a preloaded fastai vision learner
    as a model.

    Input:
        img_path (file path) to image
    Output:
        dictionary whose keys are the identifications categories
        and values are the corresponding probabilities.
    """
    img = PILImage.create(img_path)
    _, _, probs = breeds_inf.predict(img)
    return {breeds[i]: float(probs[i]) for i in range(len(breeds))}
