# remember to import package in jupyter notebook

from visuwahl.databases import *
# initialize dictionaries <-- handled by user


# get dvectors, image_data, and name, return all 3
def get_data_name_dvectors():
    image_data = get_image()
    model, boxes, probs, landmarks = bound_image(image_data)
    dvectors = vectorize_image(model, image_data, boxes)
    return dvectors, image_data, name

#add profiles to dictionary


#take input