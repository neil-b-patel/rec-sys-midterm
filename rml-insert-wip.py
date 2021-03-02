# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:10:35 2021

@author: Sam
"""

""" REPLACE YOUR OLD FUNCTION WITH THIS BUGFIXED VERSION"""
def calculateSimilarItems(prefs,n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print ("%d / %d" % (c,len(itemPrefs)))
            
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,similarity,n=n)
        result[item]=scores
    return result

""" Add this to the usual place """
file_io = input('RML(ead ml100K data)?, ') 

""" RML Insert """
elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratngs file
            itemfile = 'u.item'  # movie titles file            
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users [0:10]:' 
                  % len(prefs), list(prefs.keys())[0:10] ) 
            
