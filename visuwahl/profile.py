import numpy as np

class Profile:
    def __init__(self, name, dvectors):
        """
        initializes profile with name and dvector list

        Parameters:
        ----------
        name : String
            profile name
            
        dvectors : list
            list of profile's dvectors
            
        Returns:
        -------
        None
        """
        self.name = name
        self.dvectors = dvectors

    def set_name(self, name):
        """
        Sets profile name given a string

        Parameters:
        ----------
        name: String 
        
        Returns
        -------
        None
        """
        self.name = name

    def get_name(self):
        """
        Returns profile's name

        Parameters:
        ----------
        None
        
        Returns
        -------
        name: String
        """
        return self.name

    def set_dvectors(self, dvectors):
        """
        Sets profile dvectors given a list

        Parameters:
        ----------
        dvectors : list
            list of dvectors to be assigned
        
        Returns
        -------
        None
        """
        self.dvectors = dvectors

    def get_dvectors(self):
        """
        Return profile's list of dvectors

        Parameters:
        ----------
        None
        
        Returns
        -------
        dvectors : list
        """
        return self.dvectors
        
    def get_avg_dvector(self):
        """
        Return profile's average dvector

        Parameters:
        ----------
        None
        
        Returns
        -------
        average dvector : list
        """
        return np.mean(self.dvectors)
        
    def add_dvector(self, new_dvector):
        """
        Adds a dvector to the profile's list of dvectors

        Parameters:
        ----------
        new_dvector : list
            to be added to profile's list
        
        Returns
        -------
        None
        """
        self.dvectors.append(new_dvector)
    
    