# visuwahl_capstone
## Installation
# Simple Python Project Template

The basics of creating an installable Python package.

To install this package, in the same directory as `setup.py` run the command:

```shell
pip install -e .
```

This will install `example_project` in your Python environment. You can now use it as:

```python
from example_project import returns_one
from example_project.functions_a import hello_world
from example_project.functions_b import multiply_and_sum
```

To change then name of the project, do the following:
   - change the name of the directory `example_project/` to your project's name (it must be a valid python variable name, e.g. no spaces allowed)
   - change the `PROJECT_NAME` in `setup.py` to the same name
   - install this new package (`pip install -e .`)

If you changed the name to, say, `my_proj`, then the usage will be:

```python
from my_proj import returns_one
from my_proj.functions_a import hello_world
from my_proj.functions_b import multiply_and_sum
```

You can read more about the basics of creating a Python package [here](https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Modules_and_Packages.html).

# Planning

Create a Profile class with functionality to store face descriptors associated with a named individual.
    init(name, dvectors) -> initializes profile with name and vector list
    set_name(string) -> sets profile name given a string
    get_name() -> returns the profile's name as a string
    set_descriptor_vectors(list) -> sets profile's vector list given a list 
    get_descriptor_vectors() -> returns profile's vector list as a list
    get_avg_d_vector(descriptor_vector_list) -> returns avg of profile's vector list as a list
    add_d_vector(descriptor_vector) -> adds a descriptor vector to the vector list

Functionality to create, load, and save a database of profiles - dictionary(key=name, value=profile)
    Pickle code from last time
        (X) load_dictionary() -> load dict from pickle file     
        (X) save_dictionary() -> save dict to pickle file

    Functionality to add and remove profiles
        (X) add_profile (profile, database) -> adds profile to database
        (X) remove_profile (profile, database) -> removes profile from database
     
    Functionality to add an image to the database, given a name (create a new profile if the name isn’t in the database, otherwise add the image’s face descriptor vector to the proper profile)
        (X) add_image(name, image, database) -> gets dvectors for image and adds it to an existing profile or makes a new one
    
    Function to generate dvectors using resnet - [MINDY]
        -bound_image(image_data): pass in image data, use the MTCNN model to get: boxes, probabilities, landmarks
            -each of these are arrays length N, where N is the number of faces detected
        -vectorize_image(img, boxes): pass in image data and boxes, use the RESNET model to get descriptor vectors for each face in an image

    Functionality to find a match for a new face descriptor in the database, using a cutoff value - mindy + celine
        find_match(face_descriptor, database, cutoff) -> returns name of guessed face as string (for one face)
            -calls cosine_distance()
            -check which cosine distance is least in the database
            -if distance is less than cutoff, return name
            - calls bound_image and vectorize_image
        
        Function to measure cosine distance between face descriptors - work together with above (to know how database functions work) [CELINE]
            cosine_distance(avgdvector, database_vector) -> returns cosine distance as float

    Functionality to get image data regardless of whether it is being stored in database or compared to the database - mindy + celine
        (X) get_image() -> main function that implements camera/file image functions
        camera_image() -> prompts user for webcam image and returns image np.array
        file_image() -> prompts user for file image and returns image np.array
Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise 
    display_output(image_input, labels, boxes, landmarks)
        -display image with probs, labels, boxes, landmarks

----------

Figuring out hyperparameters (Later Part 1)

    Estimating a good detection probability threshold for rejecting false detections (e.g. a basketball detected as a face)
        - Display some test cosine distances 

    Estimating a good cosine-distance cutoff for eliminating false-matches
        - Display image probabilities with variety of images to determine 

---------

Implement the whispers algorithm - (Later Part 2)