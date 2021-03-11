# Midterm Project for CSC 381: Recommender Systems Research

## Authors
- Sam Davis '22
- Luna Jerjees '23
- Neil Patel '22
- Michael Zemanek '23

## Introduction
Hello! We are a group of students/researchers at Davidson College.

This was the result of a midterm project for the Recommender Systems Research course.

Recommender Systems provide users with product/service recommendations in order to save time and effort maneuvering through the plethora of choices available.
This work reviews various Similarity and Prediction Techniques used in the generation of Collaborative Filtering User-based and Item-based Recommendations,
providing evaluation data to determine the best recommender system basedon defined metrics.

Therefore, our team would like to answer what is the best collaborative filtering recommender system configuration, in terms of the following parameters: type of collaborative filtering algorithm, method of calculating similarity between neighbors, significance weighting, similarity threshold? Due to the fact that there are a greater number of items than users, our team believes that a user-based collaborative filtering algorithm will be more accurate [1]. Furthermore, we expect that a recommender system that contains the Pearson similarity method, no significance weighting due to the large dataset, and a similarity threshold of >0.3 will be the best.


## Experimental Design
The dataset used for our trials was the Movie Lens 100K dataset [2]. 

The variables tested in our trials were the i.) recommender system algorithm, ii.) similarity method, iii.) similarity significance weighting, and iv.) similarity threshold.

- The recommender system algorithm was either 
  - a.) User-Based or b.) Item-Based.
- The method used to calculate similarity was either 
  - a.) Euclidean Distance or b.) Pearson Correlation.
- The similarity significance weighting was either 
  - a.) ignored, b.) set at n/25, or c.) set at n/50.
- The similarity threshold for determining "neighbors" was either set at 
  - a.) >0, b.) >0.3, c.) or >0.5.
 
Rating prediction normalization, if any, would be weighted.

Evaluation of accuracy was completed using LOOCV(Leave-One-Out Cross Validation).

The computed accuracy metrics are MSE (Mean Square Error), MAE (Mean AbsoluteError), and RMSE (Root Mean Square Error).


## How to Use
1. Clone the GitHub Repo @ https://github.com/nepatel/rec-sys-midterm.git

2. Run 'recommendations.py' (NOTE: Directory path variable may need to be adjusted)

3. Read in a dataset with: R or RML 

  (R = critics, RML = MLK-100)

4. Generate a similarity matrix with: Sim or Simu 

  (Sim = Item-Based, Simu = User-Based)

5. Select a signicance similarity weighting: 0, 25, 50
  
  (0 = NONE, 25 = n/25, 50 = n/50); where n is the number of shared items between two users

6. Select a minimum similarity threshold: >0, >0.3, >0.5
   
   (0.3 or 3 = >0.3, 0.5 or 5 = >0.5, otherwise default to >0)

7. Select a subcommand: RD, RP, WD, WP 

  (R = Read, W = Write, D = Euclidean Distance, P = Pearson Correlation)

8. Test metrics of accuracy with: LCVSIM (NOTE: LCV is deprecated)

## References
[1] Christian Desrosiers and George Karypis. 2011. A comprehensive survey of neighborhood-based recommendation methods.Recommender systemshandbook(2011), 107–144.

[2] https://grouplens.org/datasets/movielens/100k/
