# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:10:35 2021

@author: Sam
"""

""" RML Insert in progress """
elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratngs file
            itemfile = 'u.item'  # movie titles file            
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users [0:10]:' 
                  % len(prefs), list(prefs.keys())[0:10] ) 