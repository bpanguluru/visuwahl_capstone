#pickling
import pickle
import mygrad as mg
import numpy as np
from facenet_models import FacenetModel # assume facenet_models is already installed in conda environment
from camera import take_picture
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage.io as io
import pathlib
import numpy as np

def initialize_database():
    """
    Initalizes a dictionary database 

    Parameters
    ----------
    None
    
    Returns
    ------
    database : dict
        initialized database
    """
    database = {}
    dictionary_type = int(input("Enter 0 to input a pickled dictionary, Enter 1 to have it initialized: "))
    # Pickled Dictionary
    if dictionary_type == 0:
        file_path = input("Enter the file path and file name to the dictionary: ")
        database = load_dictionary(file_path)
    # We initialized 
    elif dictionary_type == 1:
        pass
    # Invalid Option
    else:
        print("Error: Invalid Option") 
    return database

def file_image(path_to_image):
    # shape-(Height, Width, Color)
    image = io.imread(str(path_to_image))
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        return image[..., :-1]  # png -> RGB
    else:
        return image

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
        file_path = input("Enter the file path and file name to the dictionary: ")
        p = pathlib.Path(file_path)
        file_path = str(p.absolute())
        
        image = file_image(file_path) 
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

def find_match(img, database, cutoff=1.0):
    """
    takes in an image, detects faces within the image,
    and matches each face to the closest match in the profile database
    
    Parameters
    ---------
    img: np.array
        image to match to a label

    database: dictionary of Profiles

    cutoff: float
        the maximum cosine distance between two d vectors needed for a match
        
    Returns
    ------
    np.array[list[string], list[], list[], list[]]
        names associated with match found

    Notes
    -----
    tuple(labels, boxes, probabilites, landmarks - shape(4,))
        tuple of (labels, boxes, probabilities, landmarks)
    """
    labels = []
    model, boxes, probabilities, landmarks = bound_image(img) # box all faces
    image_dvectors = vectorize_image(model, img, boxes) # extract descriptor vector of each face
    for vec in image_dvectors:   # loop through each face identified in the image
        dists = {} # dists = [dist : <label>] make a new dictionary matching cosine distances to labels
        # database = {str <name>:Profile prof}
        # Profile -> str name, List[np.array] d_vectors
        for prof in database.values(): # loop through all the descriptor vectors in our dtaabase and compare
            dist = cosine_distance(vec, prof.get_avg_d_vector()) # find cosine distance between each profile and the face descriptor vector
            if dist < cutoff:
                dists[dist] = prof.get_name() # add distance and label to dists dictionary

        label = 'Unknown' # assume label will be 'Unknown'
        if len(dists) > 0: # if the dictionary is not empty - indicates that there will be an actual label
            min_dist = min(list(dists.keys())) # find the minimum distance from the list of keys from dists
            label = dists[min_dist] # find the new label with the minimum distance
        
        labels.append(label) # append the label to the labels list

    return labels, boxes, probabilities, landmarks # return tuple of information including a label for each of the N faces in the iamge

def bound_image(img):
    """
    boxes all of the identified faces in the image using the face_net model MTCNN (multi-task cascaded neural network)
    Parameters
    ---------
    img: np.array - shape(?)
        input image for model to detect faces
    
    Returns
    ------
    Tuple(model, Tuple(boxes, probabilities, landmarks)) 
        given the N faces identified in the image
    
    """
    # this will download the pretrained weights for MTCNN and resnet
    # (if they haven't already been fetched)
    # which should take just a few seconds
    model = FacenetModel()
    
    # detect all faces in an image
    # returns a tuple of (boxes, probabilities, landmarks)
    # assumes ``pic`` is a numpy array of shape (R, C, 3) (RGB is the last dimension)
    # If N faces are detected then arrays of N boxes, N probabilities, and N landmark-sets are returned.
    boxes, probs, landmarks = model.detect(img)
    
    return model, boxes, probs, landmarks

    
def vectorize_image(model, img, boxes):
    """
    calculates the description vector for each of the uniquely identified faces (boxes)
    in the inputted image using the provided function in the RESNET face_net model

    Parameters
    ---------
    model: RESNET model
        RESNET model from face_net initialized in bound_image
        
    img: np.array(?)
        image inputted by user

    boxes: list[]
        box locations for each face in the inputted image

    Returns
    ------
    list[np.array - shape (N, 512)]
        list of descriptor vectors for each of the N boxed faces in the image
    """

    # Crops the image once for each of the N bounding boxes
    # and produces a shape-(512,) descriptor for that face.
    #
    # If N bounding boxes were supplied, then a shape-(N, 512)
    # array is returned, corresponding to N descriptor vectors
    return model.compute_descriptors(img, boxes)

def cosine_distance(dvector, database_vector):
    """
    computes the cosine distance between two descriptor vectors

    Parameters
    ---------
    dvector: np.array() - shape: (512,)
        descriptor vector
    database_vector: np.array() - shape: (512,)
        descriptor vector

    Returns
    ------
    float
        cosine distance between vectors
    """
    return (dvector@database_vector)/(np.linalg.norm(dvector, axis=1, keepdims=True)*np.linalg.norm(database_vector, axis=1, keepdims=True))
    
    
def graph(image_data, boxes, probabilities, landmarks):
    """
    graphs boxed images with landmarks and probabilities

    Parameters
    ---------
    image_data : 3D numpy array

    boxes, probabilities, landmarks : from bound_image() 

    Returns
    ------
    None
    """
    # display_output(image_input, labels, boxes, landmarks)
    fig, ax = plt.subplots()
    ax.imshow(image_data)

    for box, prob, landmark in zip(boxes, probabilities, landmarks):
        # draw the box on the screen
        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))

        # Get the landmarks/parts for the face in box d.
        # Draw the face landmarks on the screen.
        for i in range(len(landmark)):
            ax.plot(landmark[i, 0], landmark[i, 1], "+", color="blue")