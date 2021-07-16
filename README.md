# visuwahl_capstone
## Installation
# Simple Python Project Template

The basics of creating an installable Python package.

To install this package, in the same directory as `setup.py` run the command:

```shell
pip install -e .
```

This will install `visuwahl` in your Python environment. You can now use it as:

```python
from visuwahl import *
from visuwahl import get_image
import visuwahl as vw
```

# Planning

(X) Create a Profile class with functionality to store face descriptors associated with a named individual.
    (X) init(name, dvectors) -> initializes profile with name and vector list
    (X) set_name(string) -> sets profile name given a string
    (X) get_name() -> returns the profile's name as a string
    (X) set_descriptor_vectors(list) -> sets profile's vector list given a list 
    (X) get_descriptor_vectors() -> returns profile's vector list as a list
    (X) get_avg_d_vector(descriptor_vector_list) -> returns avg of profile's vector list as a list
    (X) add_d_vector(descriptor_vector) -> adds a descriptor vector to the vector list

(X) Functionality to create, load, and save a database of profiles - dictionary(key=name, value=profile)
    (X) Pickle code from last time
        (X) load_dictionary() -> load dict from pickle file     
        (X) save_dictionary() -> save dict to pickle file

    (X) Functionality to add and remove profiles
        (X) add_profile (profile, database) -> adds profile to database
        (X) remove_profile (profile, database) -> removes profile from database
     
    (X) Functionality to add an image to the database, given a name (create a new profile if the name isn’t in the database, otherwise add the image’s face descriptor vector to the proper profile)
        (X) add_image(name, image, database) -> gets dvectors for image and adds it to an existing profile or makes a new one
    
    (X) Function to generate dvectors using resnet - [MINDY]
        (X) bound_image(img): pass in image data, use the MTCNN model to get: boxes, probabilities, landmarks
            (X) each of these are arrays length N, where N is the number of faces detected
        (X) vectorize_image(model, img, boxes): pass in the model, image data, and boxes, use the RESNET model to get descriptor vectors for each face in an image

    (X) Functionality to find a match for a new face descriptor in the database, using a cutoff value - mindy + celine
        (X) def find_match(img, database, cutoff=1.0): takes in an image, detects faces within the image, and matches each face to the closest match in the profile database
            (X) calls bound_image and vectorize_image
            (X) calls cosine_distance()
            (X) check which cosine distance is least in the database
            (X) if distance is less than cutoff, return name
            (X) returns tuple of labels, boxes, probabilities, landmarks
        
        (X) Function to measure cosine distance between face descriptors - work together with above (to know how database functions work) [CELINE]
            (X) cosine_distance(dvector, database_vector) -> returns cosine distance as float
            
    (X) Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise 
        (x) graph(image_input, labels, boxes, landmarks) -display image with probs, labels, boxes, landmarks

    (X) [TESTED] Functionality to get image data regardless of whether it is being stored in database or compared to the database - mindy + celine
        (X) get_image() -> main function that implements file_image and camera function take_picture
        (X) file_image() -> prompts user for file image and returns image np.array

----------

Figuring out hyperparameters (Later Part 1)

    Estimating a good detection probability threshold for rejecting false detections (e.g. a basketball detected as a face)
        - Display some test cosine distances 

    Estimating a good cosine-distance cutoff for eliminating false-matches
        - Display image probabilities with variety of images to determine 

---------

Implement the whispers algorithm - (Later Part 2)
    (X) Create a Node class to define each image as a Node object - already provided
    
    (X) Function to plot the nodes and edges on a graph given a tuple of nodes and the adjacent matrix - already provided

    Function to run the whispers algorithm
        - takes in a folder of images and iterates through each of them to turn it into a list of nodes
        - creates an adjacent matrix with the weighted nodes
        - returns a tuple of nodes and the adjacent matrix to run in plot_graph
Test whispers algorithm