# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 20:22:07 2021

@author: tiago
"""
# external
import json
from bunch import Bunch

class configs:
    """Configuration loader class"""

    @staticmethod
    def load_cfg(path):
        
        """ Loads configuration file

        Args
        ----------
            path (str): The path of the configuration file

        Returns
        -------
            config (json): The configuration file
        """
        
        
        with open(path, 'r') as config_file:
            config_dict = json.load(config_file)
            config = Bunch(config_dict)
        return config