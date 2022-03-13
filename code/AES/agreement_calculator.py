
from math import nan
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import sys
from scipy.stats import pearsonr
import warnings

# 
import helpers

def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
    # https://github.com/benhamner/Metrics
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    # https://github.com/benhamner/Metrics
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def format_preds(y_pred):
    return(y_pred)

def format_actuals(p_config):
    return((p_config.y_test).values)
    
def qk(y_pred, p_config):
    # Edit: Quadratic Kappa Metric is the same as cohen 
    # kappa metric in Sci-kit learn @ sklearn.metrics.cohen_kappa_score 
    # when weights are set to 'Quadratic'. Thanks to Johannes 
    # for figuring that out.

    #    
    test = (p_config.y_test).values

    # 
    quad_kappa = cohen_kappa_score(test, y_pred, weights='quadratic')

    # 
    return quad_kappa

def qwk(y_pred, p_config):
    # https://github.com/benhamner/Metrics
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """

    #    
    y = (p_config.y_test).values

    # 
    # y_pred = np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 1])
    # y   = np.array([0, 2, 1, 0, 0, 0, 1, 1, 2, 1])

    # 
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    # 
    # rater_a = np.array(rater_a, dtype=int)
    # rater_b = np.array(rater_b, dtype=int)
    # 
    assert(len(rater_a) == len(rater_b))
    # 
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    # 
    # print(rater_a)
    # print(rater_b)
    # 
    conf_mat = Cmatrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

def export_measures(
    dataset,
    type, 
    version,
    rows,
    scores_to_predict,
    feature_categorys,
    agreement_measures
    ):

    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(
        helpers.agreement_measures_output_file_name( 
            dataset,
            type, 
            version,
            rows
        ),'w+') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print("Score, Category, QWK, P, A, E, Model")
        for score_to_predict in scores_to_predict:
            for feature_category in feature_categorys:
                print(
                    str((score_to_predict.replace("_score", "")).title()) + "," + 
                    str((feature_category).title()) + "," + 
                    str(round(agreement_measures[score_to_predict][feature_category+"_score_qwk"],3)) + "," + 
                    str(round(agreement_measures[score_to_predict][feature_category+"_score_p"],3)) + "," + 
                    str(round(agreement_measures[score_to_predict][feature_category+"_score_a"],3)) + "," + 
                    str(round(agreement_measures[score_to_predict][feature_category+"_score_e"],3)) + "," + 
                    str(agreement_measures[score_to_predict][feature_category+"_model"])
                    )
        sys.stdout = original_stdout # Reset the standard output to its original value

def pearsons_correlation(y_pred, p_config):

    y_test = (p_config.y_test).values

    corr = pearsonr(y_pred, y_test)[0]
    if(str(corr) == "nan"):
        corr = 2  

    return corr

def adjacent_agreement_percentage(y_pred, p_config):

    y_test = (p_config.y_test).values

    adjacent_counts = 0

    for i in range(0, len(y_pred)):

        difference = abs(y_pred[i]-y_test[i])

        if(difference <= 1):
            adjacent_counts += 1
            
    percentage = adjacent_counts/len(y_pred)
    
    return percentage

def exact_agreement_percentage(y_pred, p_config):

    y_test = (p_config.y_test).values

    exact_counts = 0

    for i in range(0, len(y_pred)):
        if(y_pred[i] == y_test[i]):
            exact_counts +=1

    percentage = exact_counts/len(y_pred)
    
    return percentage