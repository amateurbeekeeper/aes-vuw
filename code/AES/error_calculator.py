from sklearn import metrics
import math

def calculate_error_measures(y_test, y_pred):

    mean_squared_error =            metrics.mean_squared_error(y_test, y_pred)
    root_mean_squared_error =       math.sqrt(mean_squared_error)                       
    mean_absolute_error =           metrics.mean_absolute_error(y_test, y_pred)             
    explained_variance_score =      metrics.explained_variance_score(y_test, y_pred)   
    r2_score =                      metrics.r2_score(y_test, y_pred)

    return [
        mean_squared_error, 
        root_mean_squared_error,
        mean_absolute_error,
        explained_variance_score,
        r2_score
        ]



        