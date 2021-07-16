#pickling
import pickle
from camera import take_picture
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage.io as io
from facenet_models import FacenetModel
import numpy as np

def file_image(path_to_image):
    # shape-(Height, Width, Color)
    image = io.imread(str(path_to_image))
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB


def get_image():
    """
    returns image data
    ----------
    Parameters:
    None
    
    Returns
    -------
    3D numpy array

    Notes
    -----
    
    """
    image_type = int(input("Enter 0 to load an image file, Enter 1 to take a picture with your webcam: "))
    # Image File
    if image_type == 0:
        filename = input("What's the name of the desired image file? (Include file path and extension): ")
        image = file_image(filename) # file_image() will have to take in a filepath as a string and convert with pathlib to actually use the image file
    # Webcamera Sample
    elif image_type == 1:
        image = take_picture()
    # Invalid Option
    else:
        print("Error: Invalid Option")  
    return image
    
def load_dictionary(file_path):
    """
    loads a dictionary from a Pickle file
    Parameters
    ----------
    file_path: string
        path and name of file
    
    Returns
    -------
    dictionary 
        unpickled dictionary

    Notes
    -----
    
    """
    with open(file_path, mode = "rb") as opened_file:
        return pickle.load(opened_file)
    

def save_dictionary(dict, file_path):
    """
    saves a dictionary to a Pickle file
    Parameters
    ----------
    dict: dictionary
        dictionary to pickle
    file_path: string
        path and name of file to store dictionary to 
    Returns
    -------
    None
    
    Notes
    -----
    
    """
    with open(file_path, mode = "wb") as opened_file:
        pickle.dump(dict, opened_file)

def add_profile(name, dvectors, database):
    """
    adds a profile to the database
    
    Parameters
    ----------
    name : string
        person's name
    dvectors : 
        descriptor vectors
    
    database: Dict
        Stores dvectors under names key
    
    Returns
    ---------
    None
"""
    # creates new key, value in dictionary/database
    database[name] = dvectors

def remove_profile(name, database):
    """
    removes a profile from the database
    
    Parameters
    ----------
    name : string
        the name of face profile to be removed
        
    database : dict
        database to be searched through for profile removal

    Returns
    -------
    None
    """
    # create list of keys from database
    key_list = list(database.keys())
    # iterate through indices of the key_list
    for i in range(len(key_list)):
        # check if the key at this index is equal to the name
        if str(key_list[i]) == name:
            database.pop(name)
            break

def add_image(name, image, database):
    """
    gets image and adds it to an existing profile or makes a new one

    Parameters
    ----------
    name : string

    image : np.array

    database : dict

    Returns
    -------
    None
    """
    # calculate dvectors from image data
    boxes, probabilities, landmarks = bound_image(image)
    image_dvectors = vectorize_image(image, boxes)
    # if profile exists in database, add image
    if database.__contains__(name):
        database[name].apppend(image_dvectors)
    # if profile doesnt exist yet, create it then add image
    else:
        add_profile(name, image_dvectors, database)

def find_match(face_d_vec, database, cutoff=1.0):
    """
    takes in the descriptor vector of the identified face from the inputted image 
    and matches it to the closest match in the profile database
    
    Parameters
    ---------
    face_d_vec: np.array shape - (512,)

    database: dictionary of Profiles

    cutoff: float
        the maximum cosine distance between two d vectors needed for a match
        
    Returns
    ------
    string
        name associated with match found

    Notes
    -----
    returns None if no match below cutoff is found
    """
    

def bound_image(image_data):
    """

    Parameters
    ---------

    Returns
    ------
    """
    boxes, probabilities, landmarks = model.detect(image_data)
def vectorize_image(img, boxes):
    

def graph(image_data, boxes,probabilities, landmarks):
    #display_output(image_input, labels, boxes, landmarks)
    fig, ax = plt.subplots()
    ax.imshow(image_data)

    for box, prob, landmark in zip(boxes, probabilities, landmarks):
        # draw the box on the screen
        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))

        # Get the landmarks/parts for the face in box d.
        # Draw the face landmarks on the screen.
        for i in range(len(landmark)):
            ax.plot(landmark[i, 0], landmark[i, 1], "+", color="blue")