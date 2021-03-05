# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:25:24 2021

@author: Sam
"""

def getRecommendationSim(prefs, sim=sim_pearson):
    '''
        Returns user-based recommendations
        
        Parameters: 
            prefs: dictionary containing user-item matrix
            sim: funciton to calc similarity (sim_pearson is default)
        
        Returns:
            -- A dictionary joining users (key/string) to a list of sorted 
            tuples (string movie, float prediction)
        
        #TODO
        -check accuracy

    '''
    recommendations = {}
    toRec = set()  #set of items to recommend
    uu = calculateSimilarUsers(prefs, similarity=sim)
    c=0
    for user in prefs:
        c += 1 #Status
        if c % 100 == 0:
            percent_complete = (100*c)/len(prefs)
            print(str(percent_complete)+"% complete | getRecSim")
        recs = {}
        for other in uu[user]: #other is a tuple (sim, user)
            for item in prefs[other[1]]:
                if item not in prefs[user] or prefs[user][item] ==0: #unrated
                    toRec.add(item)
        for item in toRec:
            numerator, denominator = 0, 0
            for other in uu[user]:
                if item in prefs[other[1]]: #Other has rated item
                    numerator += prefs[other[1]][item] * other[0]
                    denominator += other[0]
            if denominator != 0:
                recValue = numerator / denominator
                recs[item] = recValue
                
            recommendations[user] = recs
            toRec = set()
                
            
    #Sort the dictionaries (really make a list of tuples)
    for user in prefs:
        recommendations[user] = sorted(recommendations[user].items(), key=lambda x:x[1], reverse=True)

    return recommendations
          