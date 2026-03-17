
import mlflow #type: ignore
import pandas as pd
from sklearn.utils import estimator_html_repr
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt # type: ignore

def log_dataset(data_train, data_train_path, data_test, data_test_path):
    train_dataset = mlflow.data.from_pandas(
        data_train,
        source=data_train_path,
        name="dataset"
    )
    test_dataset = mlflow.data.from_pandas(
        data_test,
        source=data_test_path,
        name="dataset"
    )

    mlflow.log_input(train_dataset, context="Train")
    mlflow.log_input(test_dataset, context="Eval")



def log_metric(model, X_test, y_test, y_pred, y_pred_proba):
    test_metrics = {
        "training_accuracy_score": accuracy_score(y_test, y_pred),

        "training_f1_score": f1_score(
            y_test,
            y_pred,
            average="macro"
        ),

        "training_log_loss": log_loss(
            y_test,
            y_pred_proba
        ),

        "training_precision_score": precision_score(
            y_test,
            y_pred,
            average="weighted"
        ),

        "training_recall_score": recall_score(
            y_test,
            y_pred,
            average="weighted"
        ),

        "training_roc_auc": roc_auc_score(
            y_test,
            y_pred_proba,
            multi_class="ovr",
            average="macro"
        ),

        "training_score": model.score(X_test, y_test)
    }

    mlflow.log_metrics(test_metrics)



def log_model(model,input_example):

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )



def log_artifact(model, X_train, y_test, y_pred):

    # Estimator
    html = estimator_html_repr(model)

    with open("estimator.html", "w", encoding="utf-8") as f:
        f.write(html)

    mlflow.log_artifact("estimator.html")



    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.savefig("training_confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("training_confusion_matrix.png")


    # Classification Report
    report = classification_report(y_test, y_pred)

    with open("classification_report.txt", "w") as f:
        f.write(str(report))

    mlflow.log_artifact("classification_report.txt")

    # Feature importance
    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importance
    }).sort_values("importance", ascending=False)

    feature_importance.head(7).plot.bar(x="feature", y="importance")

    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    mlflow.log_artifact("feature_importance.png")

