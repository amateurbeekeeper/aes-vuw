from math import e
import os
from re import A
import pandas as pd
import sys
import csv
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
import traceback
from statistics import mean
import warnings

# try:
#     1/0
# except Exception:
#     traceback.print_exc()

# 
import error_calculator
import predictor
import helpers
import printers
import agreement_calculator

error_measure_headers = [
    'model',
    'feature_category', 
    'score_to_predict',
    'MSE', 
    'RMSE',
    'MAE', 
    "EVS",
    'R2'
    ]

models = [
    SGDRegressor(),
    ElasticNet(),
    GradientBoostingRegressor(),
    SVR(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    LinearRegression(),
    KernelRidge(),
    BayesianRidge()
]

def generate_single():
    pass

def update_error_measures_table(
    count, 
    results_table,
    metrics, 
    model,
    feature,
    feature_score_col
    ):
    results_table[count][0] = helpers.model_name(model)
    results_table[count][1] = feature
    results_table[count][2] = feature_score_col
    results_table[count][3] =  metrics[0]
    results_table[count][4] =  metrics[1]
    results_table[count][5] =  metrics[2]
    results_table[count][6] =  metrics[3]
    results_table[count][7] =  metrics[4]

    return results_table

def set_y(essays_df, score_to_predict):
    y = essays_df[score_to_predict]
    return y

def set_X(essays_df, scores_to_predict):
    X_cols_to_exclude = ['id']
    X_cols_to_exclude.extend(scores_to_predict)

    X = essays_df.drop(
        X_cols_to_exclude,
        axis=1
    )
    
    return X

def get_scores_to_predict(dataset, type):
    scores_to_predict = ['']

    if(dataset == "hewlett"):
        if(version == "2"):
            scores_to_predict = [
                "Writing_Applications",
                "Language_Conventions"
                ]
            
        elif(version == "8"):
            scores_to_predict = [
                "overall_score", 
                "Ideas_and_Content", 
                "Organization", 
                "Voice", 
                "Word_Choice", 
                "Sentence_Fluency", 
                "Conventions"
                ]

        elif(version == "7"):
            scores_to_predict = [
                "overall_score", 
                "Ideas", 
                "Organization", 	
                "Style", 
                "Conventions", 
                ]
        else:
            scores_to_predict = [
                "overall_score"
                ]
    elif(dataset == "vuw"):
        if(type == "overall"):
            scores_to_predict = [
                "overall_score"
                ]
        else:
            scores_to_predict = [
                "grammar_score", 
                "vocab_score", 
                "flow_score", 
                "ideas_score", 
                "coherence_score",
                "overall_score"
                ]

    return scores_to_predict

def export_error_measures(
    dataset,
    type, 
    version,
    input_row_count,
    data,
    ):

    output_file_name = helpers.error_measures_output_file_name(
        dataset,
        type, 
        version,
        input_row_count
        )

    print("================================")
    print("prediction_config_generator, export_error_measures_output_file_name : ")
    print("===")
    print(" output_file_name(s): " + str(output_file_name))
    print("================================") 

    with open(output_file_name, 'w+', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(error_measure_headers)
        for i in range(0, len(data)):
            writer.writerow(data[i])

def generate_permutations(
    dataset,
    type, 
    version,
    input_row_count,
    ):

    # 
    prediction_configs = []

    # 
    scores_to_predict = get_scores_to_predict(dataset, type)
    feature_categorys = helpers.get_feature_categorys(type)

    for model in models:
        for feature_category in feature_categorys:   
            for score_to_predict in scores_to_predict:  

                # 
                input_file_name = helpers.feature_category_file_name(
                    dataset, 
                    type, 
                    version, 
                    input_row_count,
                    feature_category
                    )

                # 
                essays_df = pd.read_csv(
                    input_file_name, 
                    encoding='utf-8', 
                    delimiter='\t'
                    )

                # 
                X = set_X(essays_df, scores_to_predict)
                y = set_y(essays_df, score_to_predict)
                # printers.print_x_y(X, y)

                # 
                pc = predictor.prediction_config(
                    model,
                    feature_category,
                    score_to_predict,
                    X, 
                    y
                    )

                # 
                prediction_configs.append(pc)
    
    # 
    return prediction_configs

def outlier(y_pred, p_config):
    # might be unsafe/ wrong way to remove identifiers
    # set values in table to identifiers that 
    # make it obvious it was skipped
    # or could just print them and then copy that last at end of execution
    if(mean(y_pred) > ((p_config.y).max()*3) or mean(y_pred) < (0)):
        return True

    return False

@ignore_warnings(category=ConvergenceWarning)
def run(
    dataset,
    type, 
    version,
    input_row_count,
    ):

    # 
    printers.print_run_info(
        dataset, type, version, input_row_count
        )
    
    # 
    if(helpers.params_ok(type)): pass
    else: return

    # 
    prediction_configs = generate_permutations(
        dataset, type, version, input_row_count
        )

    # 
    scores_to_predict = get_scores_to_predict(dataset, type)
    feature_categorys = helpers.get_feature_categorys(type)

    # 
    excluded = 0
    included = 0

    # 
    agreement_measures = {}
    for score_to_predict in scores_to_predict:
        agreement_measures[score_to_predict] = {}
        for feature_category in feature_categorys:
            agreement_measures[score_to_predict][feature_category+"_score_qwk"] = 0
            agreement_measures[score_to_predict][feature_category+"_score_p"] = 0
            agreement_measures[score_to_predict][feature_category+"_score_a"] = 0
            agreement_measures[score_to_predict][feature_category+"_score_e"] = 0
            agreement_measures[score_to_predict][feature_category+"_model"] = ""

    # 
    for p_config in prediction_configs:

        # 
        y_pred = predictor.predict_y(p_config)
        y_pred = helpers.predictions_to_integers(y_pred)

        # 
        if(outlier(y_pred, p_config)):
            excluded+=1
            continue # next iteration
        else:
            included+=1
       
        # 
        qwk = agreement_calculator.qwk(y_pred, p_config)
        p = agreement_calculator.pearsons_correlation(y_pred, p_config)
        a = agreement_calculator.adjacent_agreement_percentage(y_pred, p_config)
        e = agreement_calculator.exact_agreement_percentage(y_pred, p_config)

        # 
        current_score = agreement_measures[p_config.score_to_predict][p_config.feature_category+"_score_qwk"]
        if(qwk > current_score):
            agreement_measures[p_config.score_to_predict][p_config.feature_category+"_score_qwk"] = qwk
            agreement_measures[p_config.score_to_predict][p_config.feature_category+"_score_p"] = p
            agreement_measures[p_config.score_to_predict][p_config.feature_category+"_score_a"] = a
            agreement_measures[p_config.score_to_predict][p_config.feature_category+"_score_e"] = e
            agreement_measures[p_config.score_to_predict][p_config.feature_category+"_model"] = helpers.model_name(p_config.model)

    # 
    agreement_calculator.export_measures(
         dataset,
            type, 
            version,
            rows,
            scores_to_predict,
            feature_categorys,
            agreement_measures
    )

# ================================
#   run: command line 
# ================================ 

if __name__ == "__main__":
    dataset = str(sys.argv[1])
    type = str(sys.argv[2])
    version = str(sys.argv[3])
    rows = str(sys.argv[4])

    run(dataset,type, version, rows)

# ================================
#   run: manual
# ================================ 

# python3 -u "prediction_config_generator.py"

# 
# dataset = "hewlett"
# type = "overall"    # categorized, overall
# version = "3"           
# rows = "1726"             # total id's in file

# 
# run(dataset,type, version, rows)






# ================================

# 
# results_table = [
#     ["" for i in range(len(error_measure_headers))]
#     for j in range(len((feature_categorys*len(scores_to_predict))*len(models)) )
#     ]
# count = -1
# 
# error_measures = error_calculator.calculate_error_measures(
#     p_config.y_test, 
#     y_pred
#     )
# count += 1
# results_table = add_results_to_table( 
#     count, 
#     results_table,
#     error_measures, 
#     p_config.model,
#     p_config.feature_category,
#     p_config.score_to_predict
# ) 

# export(
#     dataset,
#     type, 
#     version,
#     input_row_count,
#     results_table,
# )