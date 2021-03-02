'''
CSC381: Building a simple Recommender System

The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
This code is for academic use/purposes only.

CSC381 Programmer/Researcher: Neil Patel

'''
from matplotlib import pyplot as plt
from math import sqrt
import numpy as np
import os
import copy
import math
import pickle


def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file

        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name

        Returns:
        -- prefs: a nested dictionary containing item ratings for each user

    '''

    # Get movie titles, place into movies dictionary indexed by itemID
    movies = {}
    try:
        with open(path + '/' + itemfile, encoding='iso8859') as myfile:
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id, title) = line.split('|')[0:2]
                movies[id] = title.strip()

    # Error processing
    except UnicodeDecodeError as ex:
        print(ex)
        print(len(movies), line, id, title)
        return {}
    except Exception as ex:
        print(ex)
        print(len(movies))
        return {}

    # Load data into a nested dictionary
    prefs = {}
    for line in open(path+'/' + datafile):
        # print(line, line.split('\t')) #debug
        (user, movieid, rating, ts) = line.split('\t')
        user = user.strip()  # remove spaces
        movieid = movieid.strip()  # remove spaces
        prefs.setdefault(user, {})  # make it a nested dicitonary
        prefs[user][movies[movieid]] = float(rating)

    # return a dictionary of preferences
    return prefs


def data_stats(prefs, filename):
    ''' Computes/prints descriptive analytics:
        -- Total number of users, items, ratings
        -- Overall average rating, standard dev (all users, all items)
        -- Average item rating, standard dev (all users)
        -- Average user rating, standard dev (all items)
        -- Matrix ratings sparsity
        -- Ratings distribution histogram (all users, all items)

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed

        Returns:
        -- None

    '''
    items = []
    num_users = len(prefs)
    ratings = []
    item_ratings = {}
    user_ratings = {}

    for user in prefs:
        for item in prefs[user]:
            # create list of all unique items
            if item not in items:
                items.append(item)

            # create list of all ratings
            ratings.append(prefs[user][item])

            # create dictionary of all ratings for an item
            if item not in item_ratings:
                item_ratings[item] = []
            item_ratings[item].append(prefs[user][item])

            # create dictionary of all ratings for a user
            if user not in user_ratings:
                user_ratings[user] = []
            user_ratings[user].append(prefs[user][item])

    # calc mean overall rating and std
    mean_rating = round(np.average(ratings), 2)
    rating_std = round(np.std(ratings), 2)

    # calc mean item rating and std
    mean_item_ratings = []
    for item in item_ratings:
        mean_item_ratings.append(np.average(item_ratings[item]))
    mean_item_rating = round(np.average(mean_item_ratings), 2)
    item_std = round(np.std(mean_item_ratings), 2)

    # calc mean user rating and std
    mean_user_ratings = []
    for user in user_ratings:
        mean_user_ratings.append(np.average(user_ratings[user]))
    mean_user_rating = round(np.average(mean_user_ratings), 2)
    user_std = round(np.std(mean_user_ratings), 2)

    # calc u-i matrix sparsity
    matrix_sparsity = round(
        ((1 - (len(ratings)/(num_users * len(items)))) * 100), 2)

    print('Number of users: ', num_users)
    print('Number of items: ', len(items))
    print('Number of ratings: ', len(ratings))
    print('Overall average rating: {} out of 5, and std dev of {}'.format(
        mean_rating, rating_std))
    print('Average item rating: {} out of 5, and std dev of {}'.format(
        mean_item_rating, item_std))
    print('Average user rating: {} out of 5, and std dev of {}'.format(
        mean_user_rating, user_std))
    print('User-Item Matrix Sparsity: {}%'.format(matrix_sparsity))
    print()

    # create ratings histogram
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(ratings, bins=[1, 2, 3, 4, 5])
    plt.xlabel('Rating')
    plt.ylabel('Number of user ratings')
    plt.title('Ratings Histogram')
    plt.show()
    print()

    return


def popular_items(prefs, filename):
    ''' Computes/prints popular items analytics
        -- popular items: most rated (sorted by # ratings)
        -- popular items: highest rated (sorted by avg rating)
        -- popular items: highest rated items that have at least a
                          "threshold" number of ratings

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed

        Returns:
        -- None

    '''
    ratings = {}
    most_rated = {}
    highest_rated = {}
    highest_rated_min5 = {}
    num_ratings = {}
    mean_item_ratings = {}

    # create a dictionary of ratings (all users, all items)
    for user in prefs:
        for item in prefs[user]:
            if item not in ratings:
                ratings[item] = []
            ratings[item].append(prefs[user][item])

    # calc number of ratings per item and average item rating
    # use these values to rewrite ratings dict
    for item in ratings:
        num_ratings[item] = len(ratings[item])
        mean_item_ratings[item] = round(np.average(ratings[item]), 2)
        ratings[item] = [len(ratings[item]), round(
            np.average(ratings[item]), 2)]

    # sort titles by decreasing values, using the secondary column as a tiebreaker
    most_rated = sorted(ratings, key=lambda x: (
        ratings[x][0], ratings[x][1]), reverse=True)
    highest_rated = sorted(ratings, key=lambda x: (
        ratings[x][1], ratings[x][0]), reverse=True)

   # counter is used in following print statements to limit to 5 items

    # prints items in order of descending number of ratings
    print('Popular items -- most rated: ')
    print('Title'.ljust(30) + '#Ratings'.ljust(15) + 'Avg Rating'.ljust(15))

    counter = 0
    for item in most_rated:
        counter += 1
        print(item.ljust(30) +
              str(ratings[item][0]).ljust(15) + str(ratings[item][1]).ljust(15))
        if counter == 5:
            break
    print()

    # prints items in order of descending average rating
    print('Popular items -- highest rated: ')
    print('Title'.ljust(30) + 'Avg Rating'.ljust(15) + '#Ratings'.ljust(15))

    counter = 0
    for item in highest_rated:
        counter += 1
        print(item.ljust(30) +
              str(ratings[item][1]).ljust(15) + str(ratings[item][0]).ljust(15))
        if counter == 5:
            break
    print()

    # prints items with at least 5 ratings in order of descending average rating
    print('Overall best rated items (number of ratings >= 5): ')
    print('Title'.ljust(30) + 'Avg Rating'.ljust(15) + '#Ratings'.ljust(15))

    counter = 0
    for item in highest_rated:
        counter += 1
        if ratings[item][0] >= 5:
            print(item.ljust(
                30) + str(ratings[item][1]).ljust(15) + str(ratings[item][0]).ljust(15))
        if counter == 5:
            break
    print()

    return


def sim_distance(prefs, person1, person2):
    '''
        Calculate Euclidean distance similarity

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2

        Returns:
        -- Euclidean distance similarity as a float

    '''

    # Get the list of shared_items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    # if they have no ratings in common, return 0
    if len(si) == 0:
        return 0

    # Add up the squares of all the differences
    sum_of_squares = sum([pow(prefs[person1][item]-prefs[person2][item], 2)
                          for item in prefs[person1] if item in prefs[person2]])

    return 1/(1+sqrt(sum_of_squares))


def sim_pearson(prefs, p1, p2):
    '''
        Calculate Pearson Correlation similarity

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2

        Returns:
        -- Pearson Correlation similarity as a float

    '''

    # Get the list of shared_items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # if they have no ratings in common, return 0
    if len(si) == 0:
        return 0

    # sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2)
        # for item in prefs[person1] if item in prefs[person2]])

    # calc avg rating for p1 and p2, using only shared ratings
    x_avg = 0
    y_avg = 0

    for item in si:
        x_avg += prefs[p1][item]
        y_avg += prefs[p2][item]

    x_avg /= len(si)
    y_avg /= len(si)

    # calc numerator of Pearson correlation formula
    numerator = sum([(prefs[p1][item] - x_avg) * (prefs[p2][item] - y_avg)
                     for item in si])

    # calc denominator of Pearson correlation formula
    denominator = math.sqrt(sum([(prefs[p1][item] - x_avg)**2 for item in si])) * \
        math.sqrt(sum([(prefs[p2][item] - y_avg)**2 for item in si]))

    # catch divide-by-0 errors
    if denominator != 0:
        return numerator/denominator
    else:
        return 0


def getRecommendations(prefs, person, similarity=sim_pearson):
    '''
        Calculates recommendations for a given user

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)

        Returns:
        -- A list of recommended items with 0 or more tuples,
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.

    '''

    totals = {}
    simSums = {}
    for other in prefs:
      # don't compare me to myself
        if other == person:
            continue
        sim = similarity(prefs, person, other)

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in prefs[other]:

            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item] == 0:
                # Similarity * Score
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

    # Create the normalized list
    rankings = [(total/simSums[item], item) for item, total in totals.items()]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings


def calc_all_users(prefs, method=sim_distance):
    '''
        Prints selected calculation for all users

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- method: function to calc for all users
                    (sim_distance is default, but sim_pearson
                     and getRecommendations are also valid)

        Returns:
        -- None

    '''
    if method == getRecommendations:
        for user in prefs:
            print('Recommendations for {}:'.format(user))
            print('User-based CF recs for {}, sim_pearson: '.format(user),
                  getRecommendations(prefs, user))
            print('User-based CF recs for {}, sim_distance: '.format(user),
                  getRecommendations(prefs, user, similarity=sim_distance))
            print()
    else:
        for u1 in prefs:
            for u2 in prefs:
                if u1 != u2:
                    if method == sim_distance:
                        print('Distance sim {} & {}: '.format(
                            u1, u2), sim_distance(prefs, u1, u2))
                    else:
                        print('Pearson sim {} & {}: '.format(
                            u1, u2), sim_pearson(prefs, u1, u2))

    return


def loo_cv(prefs, metric, sim, algo):
    """
        Leave_One_Out Evaluation: evaluates recommender system ACCURACY

        Parameters:
        -- prefs: dataset of critics, ml-100K, etc.
        -- metric: MSE, MAE, RMSE, etc.
        -- sim: distance, pearson, etc.
        -- algo: user-based recommender, item-based recommender, etc.

        Returns:
        -- error: MSE, MAE, RMSE totals for this set of conditions
        -- error_list: list of actual-predicted differences

    """
    error = 0
    error_list = []
    pred_found = False
    recs = []

    # Create a temp copy of prefs
    prefs_cp = copy.deepcopy(prefs)

    for user in prefs:
        for item in prefs[user]:
            removed_rating = prefs_cp[user].pop(item)
            recs = algo(prefs_cp, user, similarity=sim)
            print(recs)
            for rec in recs:
                if item in rec:
                    pred_found = True
                    predicted_rating = rec[0]
                    error_i = 0
                    if metric == "MAE" or metric == "mae":
                        error_i = abs(predicted_rating - removed_rating)
                    else:
                        error_i = (predicted_rating - removed_rating)**2
                    error_list.append(error_i)
                    print('User : {}, Item: {}, Prediction {}, Actual: {}, Error: {}'.format(
                        user, item, predicted_rating, removed_rating, error_i))
            if pred_found == False:
                print('From loo_cv(), No prediction calculated for item {}, user {} in pred_list: {}'.format(
                    item, user, recs))
            pred_found = False
            prefs_cp[user][item] = removed_rating

    if metric == "RMSE" or metric == "rmse":
        error = sqrt(np.average(error_list))
    else:
        error = np.average(error_list)

    return error, error_list


def topMatches(prefs, person, similarity=sim_pearson, n=5):
    '''
        Returns the best matches for person from the prefs dictionary

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)

        Returns:
        -- A list of similar matches with 0 or more tuples,
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.

    '''
    scores = [(similarity(prefs, person, other), other)
              for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary)

        Parameters:
        -- prefs: dictionary containing user-item matrix

        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix,
           this function returns an I-U matrix

    '''
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            # Flip item and person
            result[item][person] = prefs[person][item]
    return result


def calculateSimilarItems(prefs, n=100, similarity=sim_pearson):
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
    result = {}
    # Invert the preference matrix to be item-centric
    itemPrefs = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
      # Status updates for larger datasets
        c += 1
        if c % 100 == 0:
            print("%d / %d" % (c, len(itemPrefs)))

        # Find the most similar items to this one
        scores = topMatches(itemPrefs, item, similarity, n=n)
        result[item] = scores
    return result


def calculateSimilarUsers(prefs, n=100, similarity=sim_pearson):
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
    result = {}
    c = 0
    for user in prefs:
      # Status updates for larger datasets
        c += 1
        if c % 100 == 0:
            percent_complete = (100*c)/len(prefs)
            print(str(percent_complete)+"% complete")

        # Find the most similar items to this one
        scores = topMatches(prefs, user, similarity, n=n)
        result[user] = scores
    return result


def getRecommendedItems(prefs, itemMatch, user):
    '''
        Calculates recommendations for a given user

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- itemMatch: dictionary containing similarity matrix
        -- user: string containing name of user

        Returns:
        -- A list of recommended items with 0 or more tuples,
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.

    '''
    userRatings = prefs[user]
    scores = {}
    totalSim = {}
    # Loop over items rated by this user
    for (item, rating) in userRatings.items():

      # Loop over items similar to this one
        for (similarity, item2) in itemMatch[item]:

            # Ignore if this user has already rated this item
            if item2 in userRatings:
                continue
            # ignore scores of zero or lower
            if similarity <= 0:
                continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2, 0)
            scores[item2] += similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity

    # Divide each total score by total weighting to get an average

    rankings = [(score/totalSim[item], item) for item, score in scores.items()]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings


def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):
    '''
        Print item-based CF recommendations for all users in dataset

        Parameters
        -- prefs: U-I matrix (nested dictionary)
        -- itemsim: item-item similarity matrix (nested dictionary)
        -- sim_method: name of similarity method used to calc sim matrix (string)
        -- num_users: max number of users to print (integer, default = 10)
        -- top_N: max number of recommendations to print per user (integer, default = 5)

        Returns:
        -- None

    '''
    counter = 0

    if num_users > 0:
        for user in prefs:
            if counter < num_users:
                recs = getRecommendedItems(prefs, itemsim, user)
                if len(recs) > top_N:
                    recs = recs[0:top_N]
                print('Item-based CF recs for %s, %s: ' %
                      (user, sim_method), recs)
            else:
                break
            counter += 1

    return


def loo_cv_sim(prefs, metric, sim, algo, sim_matrix):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY

     Parameters:
         prefs dataset: critics, etc.
         metric: MSE, or MAE, or RMSE
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
         sim_matrix: pre-computed similarity matrix

    Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
         error_list: list of actual-predicted differences

    """
    error = 0
    error_list = []
    pred_found = False
    recs = []

    # Create a temp copy of prefs
    prefs_cp = copy.deepcopy(prefs)

    for user in prefs:
        for item in prefs[user]:
            removed_rating = prefs_cp[user].pop(item)
            recs = algo(prefs_cp, sim_matrix, user)
            for rec in recs:
                if item in rec:
                    pred_found = True
                    predicted_rating = rec[0]
                    error_i = 0
                    if metric == "MAE" or metric == "mae":
                        error_i = abs(predicted_rating - removed_rating)
                    else:
                        error_i = (predicted_rating - removed_rating)**2
                    error_list.append(error_i)
                    print('User : {}, Item: {}, Prediction {}, Actual: {}, Error: {}'.format(
                        user, item, predicted_rating, removed_rating, error_i))
            if pred_found == False:
                print('From loo_cv(), No prediction calculated for item {}, user {} in pred_list: {}'.format(
                    item, user, recs))
            pred_found = False
            prefs_cp[user][item] = removed_rating

    if metric == "RMSE" or metric == "rmse":
        error = sqrt(np.average(error_list))
    else:
        error = np.average(error_list)

    return error, error_list


def main():
    ''' User interface for Python console '''

    # Load critics dict from file
    path = os.getcwd()  # this gets the current working directory
    # you can customize path for your own computer here
    print('\npath: %s' % path)  # debug
    done = False
    prefs = {}
    itemsim = {}

    while not done:
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, '
                        'RML(ead ml100K data)?, '
                        'P(rint) the U-I matrix?, '
                        'V(alidate) the dictionary?, '
                        'S(tats) print?, '
                        'D(istance) critics data?, '
                        'PC(earson Correlation) critics data? '
                        'U(ser-based CF Recommendations)? '
                        'LCV(eave one out cross-validation)? '
                        'LCVSIM(eave one out cross-validation)?'
                        'Sim(ilarity matrix) calc? '
                        'I(tem-based CF Recommendations)? ')

        if file_io == 'R' or file_io == 'r':
            # read in u-i matrix data
            print()
            file_dir='data/'
            datafile='critics_ratings.data'
            itemfile='critics_movies.item'
            print('Reading "%s" dictionary from file' % datafile)
            prefs=from_file_to_dict(
                path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs),
                  list(prefs.keys()))

        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratings file
            itemfile = 'u.item'  # movie titles file            
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users [0:10]:' 
                  % len(prefs), list(prefs.keys())[0:10] )

        elif file_io == 'P' or file_io == 'p':
            # print the u-i matrix
            print()
            if len(prefs) > 0:
                print('Printing "%s" dictionary from file' % datafile)
                print('User-item matrix contents: user, item, rating')
                for user in prefs:
                    for item in prefs[user]:
                        print(user, item, prefs[user][item])
            else:
                print('Empty dictionary, R(ead) in some data!')

        elif file_io == 'V' or file_io == 'v':
            print()
            if len(prefs) > 0:
                # Validate the dictionary contents ..
                print('Validating "%s" dictionary from file' % datafile)
                print("critics['Lisa']['Lady in the Water'] =",
                      prefs['Lisa']['Lady in the Water'])  # ==> 2.5
                print("critics['Toby']:", prefs['Toby'])
                # ==> {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0,
                #      'Superman Returns': 4.0}
            else:
                print('Empty dictionary, R(ead) in some data!')

        elif file_io == 'S' or file_io == 's':
            # print useful statistics
            print()
            filename = 'critics_ratings.data'
            if len(prefs) > 0:
                data_stats(prefs, filename)
                popular_items(prefs, filename)
            else:  # Make sure there is data  to process ..
                print('Empty dictionary, R(ead) in some data!')

        elif file_io == 'D' or file_io == 'd':
            # calculate u-u Euclidean distance similarities
            print()
            if len(prefs) > 0:
                print('User-User distance similarities:')

                calc_all_users(prefs)

                print()
            else:
                print('Empty dictionary, R(ead) in some data!')

        elif file_io == 'PC' or file_io == 'pc':
            # calculate u-u Pearson correlation similarities
            print()
            if len(prefs) > 0:
                print('Pearson for all users:')

                calc_all_users(prefs, method=sim_pearson)

                print()

            else:
                print('Empty dictionary, R(ead) in some data!')

        elif file_io == 'U' or file_io == 'u':
            # calculate User-based CF Recommendations for all users
            print()
            if len(prefs) > 0:
                print('User-based CF recommendations for all users:')

                calc_all_users(prefs, method=getRecommendations)

                print()

            else:
                print('Empty dictionary, R(ead) in some data!')

        elif file_io == 'LCV' or file_io == 'lcv':
            # calculate MSEs of User-based CF Recommendations for all users and items
            print()
            if len(prefs) > 0:
                # ---- START User-User LOOCV ----

                # first with Pearson correlations similarity
                error, error_list = loo_cv(
                    prefs, "MSE", sim_pearson, getRecommendations)
                print('MSE for critics: {}, len(SE list): {}, using function sim_pearson'.format(
                    error, len(error_list)))

                print()

                # then with Euclidean distance similarity
                error, error_list = loo_cv(
                    prefs, "MSE", sim_distance, getRecommendations)
                print('MSE for critics: {}, len(SE list): {}, using function sim_distance'.format(
                    error, len(error_list)))

                # ---- END User-User LOOCV ----

                print()

            else:
                print('Empty dictionary, R(ead) in some data!')

        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
            print()
            if len(prefs) > 0 and itemsim != {}:
                print('LOO_CV_SIM Evaluation')
                
                # check for small or large dataset (for print statements)
                if len(prefs) <= 7:
                    prefs_name = 'critics'
                else:
                    prefs_name = 'MLK-100k'

                algo = input('Enter algorithm: U(ser-based) or I(tem-based)')
                if algo == 'I' or algo =='i':
                    algo = getRecommendedItems
                else:
                    algo = getRecommendationsSim


                if sim_method == 'sim_pearson':
                    sim = sim_pearson
                    error_total, error_list = loo_cv_sim(
                        prefs, sim, algo, itemsim)
                    print('%s for %s: %.5f, len(SE list): %d, using %s'
                          % (prefs_name, error_total, len(error_list), sim))
                    print()
                elif sim_method == 'sim_distance':
                    sim = sim_distance
                    error_total, error_list = loo_cv_sim(
                        prefs, sim, algo, itemsim)
                    print('%s for %s: %.5f, len(SE list): %d, using %s'
                          % (prefs_name, error_total, len(error_list), sim))
                    print()
                else:
                    print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
                
                if prefs_name == 'critics':
                    print(error_list)

            else:
                print('Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!')

        elif file_io == 'Sim' or file_io == 'sim':
            print()
            if len(prefs) > 0:
                ready = False  # subc command in progress
                sub_cmd = input(
                    'RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open("save_itemsim_distance.p", "rb"))
                        sim_method = 'sim_distance'

                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open("save_itemsim_pearson.p", "rb"))
                        sim_method = 'sim_pearson'

                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs, similarity=sim_distance)
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open(
                            "save_itemsim_distance.p", "wb"))
                        sim_method = 'sim_distance'

                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs, similarity=sim_pearson)
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open(
                            "save_itemsim_pearson.p", "wb"))
                        sim_method = 'sim_pearson'

                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue

                    ready = True  # sub command completed successfully

                except Exception as ex:
                    print('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                          ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()

                if len(itemsim) > 0 and ready == True and len(itemsim) <= 10:
                    # Only want to print if sub command completed successfully
                    print('Similarity matrix based on %s, len = %d'
                          % (sim_method, len(itemsim)))
                    print()
``
                    for item in itemsim:
                        print('{} : {}'.format(item, itemsim[item]))

                    # EUCLIDEAN RESULTS
                    # Lady in the Water is most similar to You, Me and Dupree (0.45)
                    # Snakes on a Plane is least similar to You, Me and Dupree (0.19)

                    # PEARSON RESULTS
                    # Lady in the Water is most similar to Snake on a Plane (0.76)
                    # Superman Returns is least similar to Snakes on a Plane (0.11)
                    # Lady in the Water is the most inversely similar to Just My Luck (-0.94)

            else:
                print('Empty dictionary, R(ead) in some data!')

        elif file_io == 'I' or file_io == 'i':
            print()
            if len(prefs) > 0 and len(itemsim) > 0:
                print('Item-based CF recommendations for all users:')
                # Calc Item-based CF recommendations for all users

                get_all_II_recs(prefs, itemsim, sim_method)

                print()

            else:
                if len(prefs) == 0:
                    print('Empty dictionary, R(ead) in some data!')
                else:
                    print(
                        'Empty similarity matrix, use Sim(ilarity) to create a sim matrix!')
        else:
            done = True

    print('\nGoodbye!')


if __name__ == '__main__':
    main()
