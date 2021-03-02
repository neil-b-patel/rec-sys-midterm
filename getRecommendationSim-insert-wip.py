# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:25:24 2021

@author: Sam
"""

def getRecommendationSim(prefs, sim=sim_pearson):
    '''
        Returns user-based recommendations
        Returns:
            -- Dict of format <user:<movie:prediction>>
        #TODO
        -flesh out header
        -sort inner dict by value
        -runtime optimazation as necessary
        -check accuracy
        -remove comments
    '''
    recommendations = {}
    toRec = set()  #set of ites to recommend
    uu = calculateSimilarUsers(prefs, similarity=sim)
    #print(uu["1"])
    for user in prefs:
        recs = {}
        for other in uu[user]: #other is a tuple (sim, user)
            for item in prefs[other[1]]:
                if item not in prefs[user] or prefs[user][item] ==0:
                    toRec.add(item)
        for item in toRec:
            numerator, denominator = 0, 0
            for other in uu[user]:
                if item in prefs[other[1]]:
                    numerator += prefs[other[1]][item] * other[0]
                    denominator += other[0]
            if denominator != 0:
                recValue = numerator / denominator
                recs[item] = recValue
                
            recommendations[user] = recs
            toRec = set()
                
            
    #print(recommendations['1'])
    #for user in prefs:
    #    sorted(recommendations[user], key=recommendations[user].get)
    #print()
    #print(recommendations['1'])
    return recommendations