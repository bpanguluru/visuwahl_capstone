#pickling

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

       