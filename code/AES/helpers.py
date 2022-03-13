import os
import numpy as np
import sys

def directory_name():
    dirname = os.path.dirname(__file__)
    return str(dirname)

def model_name(model):
    name  = str(model.__class__.__name__)
  
    if(name== "SGDRegressor"):          return "SGD"
    elif(name==  "ElasticNet"):         return "EN"
    elif(name==  "GradientBoostingRegressor"): return "GBR"
    elif(name==  "SVR"):                   return "SVR"
    elif(name==  "DecisionTreeRegressor"): return "DTR"
    elif(name==  "RandomForestRegressor"): return "RFR"
    elif(name==  "LinearRegression"):      return "LR"
    elif(name==  "KernelRidge"):           return "KR"
    elif(name==  "BayesianRidge"):         return "BR"
    
    return name

def params_ok(type):
    if(type == "overall"): return True
    elif(type == "categorized"): return True
    else:
        print("Error: params_ok: false \n")
        return False

def feature_category_file_name(
    dataset, 
    type, 
    version, 
    rows,
    feature
    ):

    n = str(
        directory_name()+"/"+
        "feature_categories"+"/"+
        dataset+"/"+
        type+"/"+
        version+"/"+
        str(rows)+"/"+
        feature+"_features"+".tsv"
    )

    return n

def error_measures_output_file_name(
    dataset,
    type, 
    version,
    rows
    ):

    n = str(
        directory_name()+"/"+
        "error_tests"+"/"+
        dataset+"/"+
        type+"/"+
        version+"/"+
        str(rows)+
        ".csv"
        )

    return n

def feature_importance_tests_output_file_name(
    dataset,
    type, 
    version,
    title
    ):

    n = str(
        directory_name()+"/"+
        "feature_importance_tests"+"/"+
        dataset+"/"+
        type+"/"+
        version+"/"+
        title+".csv"
        )

    return n

def structured_essays_file_name( 
    dataset,
    type, 
    version,
    rows
    ):

    n = str(
        directory_name()+"/"+
        "structured_essays/"+
        dataset+"/"+
        type+"/"+
        version+"/"+
        rows+".tsv"
    )

    return n

def feature_category_output_file_name(  
    dataset,
    type, 
    version,
    rows,
    feature
    ):

    n = str(
        directory_name()+"/"+
        "feature_categories/"+
        dataset+"/"+
        type+"/"+
        version+"/"+
        rows+"/"+
        feature+"_features"+".tsv"
    )

    return n

def get_feature_categorys(type):
    if(type == "overall"):
        feature_categorys = ["overall"]
    else:
        feature_categorys = [
            "grammar", 
            "vocab", 
            "flow", 
            "ideas", 
            "coherence",
            "overall"
        ]

    return feature_categorys


def predictions_to_integers(y_pred):
    a = np.array(y_pred)
    a = a.astype(int)
    return a

def simple_export(content, filename):
    print("================================")
    print("helpers, simple_export : ")
    print("===")
    print(" output_file_name(s): " + str(filename))
    print("================================") 

    original_stdout = sys.stdout # Save a reference to the original standard output

    with open(filename, 'w+') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(content)
        sys.stdout = original_stdout # Reset the standard output to its original value

def agreement_measures_output_file_name(
    dataset,
    type, 
    version,
    rows
    ):

    n = str(
        directory_name()+"/"+
        "agreement_tests"+"/"+
        dataset+"/"+
        type+"/"+
        version+"/"+
        str(rows)+
        ".csv"
        )

    return n