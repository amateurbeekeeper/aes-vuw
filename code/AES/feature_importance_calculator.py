import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
import numpy as np

# 
import prediction_config_generator 
import helpers

@ignore_warnings(category=ConvergenceWarning)
def run():

    dataset="vuw"
    type="categorized"
    version="1"
    rows="113"

    input_file_name = helpers.feature_category_file_name(
        dataset, 
        type, 
        version, 
        rows,
        ("overall")
    )

    essays_df = pd.read_csv(
        input_file_name, 
        encoding='utf-8', 
        delimiter='\t'
    )

    fs = SelectKBest(
        score_func=f_regression, k=10
    )

    scores_to_predict = prediction_config_generator.get_scores_to_predict("vuw", "categorical")

    for score_to_predict in scores_to_predict:

        X = prediction_config_generator.set_X(essays_df, scores_to_predict)
        y = prediction_config_generator.set_y(essays_df, score_to_predict)

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=1
        )

        title = str("overall_x_"+str(score_to_predict))
    
        fs.fit(
            X_train, 
            y_train
        )

        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)

        # export
        original_stdout = sys.stdout # 
        with open(
            helpers.feature_importance_tests_output_file_name(
                dataset,
                type, 
                version,
                title
        ),'w+') as f:
            sys.stdout = f #
            print("Feature,Score")
            removed_nans = np.nan_to_num(list(fs.scores_))
            b = np.array(removed_nans)
            a = np.array(list(X))
            b = b.astype(np.float)
            c = np.column_stack((a,b))

            for row in c:
                print(row[0] +','+ row[1])
        sys.stdout = original_stdout # 

# ================================
#   run: command line 
#    ...
# ================================ 

if __name__ == "__main__":
    run()

# ================================
#   run: manual
#    ...
# ================================ 

# python3 -u "feature_importance_tester.py"

# run()