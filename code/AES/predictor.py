from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

class prediction_config:
    def __init__(
        self,  
        model,
        feature_category,
        score_to_predict,
        X, 
        y
        ):

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=1
            )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X = X
        self.y = y
        self.model = model
        self.feature_category = feature_category
        self.score_to_predict = score_to_predict

    def trained_model(self):
        trained_model = self.model.fit(self.X_train, self.y_train)
        return trained_model

@ignore_warnings(category=ConvergenceWarning)
def predict_y(config: prediction_config):
    model = config.trained_model()
    y_pred = model.predict(config.X_test)
    return y_pred


# Save the trained model to a file so we can use it in other programs
# joblib.dump(model, 'trained_essay_scoring_model.pkl')