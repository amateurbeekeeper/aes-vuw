def print_predictions(y_pred):
    print("================================")
    print("printer, predictions : ")
    print("===")
    print(" list(y_pred) : " + str(list(y_pred)[0:3]))
    print("================================")

def print_calculation_values(X_train, X_test, y_train, y_test):
    print("================================")
    print("printer, calculation_values : ")
    print("===")
    print(" list(X_train) : " + str(list(X_train)[0:3]))
    print(" list(X_test) : " + str(list(X_test)[0:3]))
    print(" list(y_train) : " + str(list(y_train)[0:3]))
    print(" list(y_test) : " + str(list(y_test)[0:3]))
    print("================================")

def print_x_y(X,y):
    print("================================")
    print("printer, x & y : ")
    print("===")
    print(" list(X) : " + str(list(X)[0:8]))
    print(X)
    print(" list(y) : " + str(list(y)[0:8]))
    print(y)
    print("================================")

def print_run_info(
    dataset, 
    type, 
    version, 
    input_row_count
    ):
    print("================================")
    print("printer, main : ")
    print("===")
    print(" dataset :       " + dataset)
    print(" type:           " + type)
    print(" version :       " + version)
    print(" rows:           " + input_row_count)
    print("================================")

def print_error_measures(
    feature_score_col,
    feature, 
    mean_squared_error, 
    root_mean_squared_error, 
    mean_absolute_error, 
    explained_variance_score,
    r2_score,
    regressor_name
    ):

    print("================================")
    print("printer, error_measures : ")
    print("===")
    print(" " + feature + " x " + feature_score_col + " x " + regressor_name + " :") 
    print("   " + "mean_squared_error:          ", str(mean_squared_error)[0:5])
    print("   " + "root_mean_squared_error:     ", str(root_mean_squared_error)[0:5])
    print("   " + "mean_absolute_error:         ", str(mean_absolute_error)[0:5])
    print("   " + "explained_variance_score:    ", str(explained_variance_score)[0:5])
    print("   " + "r2_score:                    ", str(r2_score)[0:5])
    print("================================")

def print_predictions_and_actuals(y_pred, p_config, x):
    print("================================")
    print("printer, predictions_and_actuals : ")
    print("===")
    print("   " + "predictions:             ", str(y_pred[0:x]))
    print("   " + "actuals:                 ", str((p_config.y_test).values[0:x]))
    print("================================")
