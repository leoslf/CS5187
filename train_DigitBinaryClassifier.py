from streetnumber.detection import *
import pandas as pd

from sklearn.model_selection import train_test_split

metrics = ["loss", "accuracy"]

def handle_history(model, history, test_metric, test_metric_value):
    for metric, val_metric in map(tee_val, metrics):
        fig, ax = plt.subplots(1)

        ax.plot(history.history[metric], label = "Train")
        ax.plot(history.history[val_metric], label = "Validation")
        ax.set_title("Model: %s - %s" % (model.name, metric))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)

        fig.legend(loc = "upper right")
        fig.savefig("outputs/%(model_name)s_%(metric)s_%(test_metric)s_%(value).4f.png" % dict(model_name = model.name, metric = metric, test_metric = test_metric, value = test_metric_value))
        
        plt.close(fig)

if __name__ == "__main__":
    model = DigitBinaryClassifier()

    df = pd.read_csv("train_patches.csv")
    df.label = df.label.apply(str)
    test_df = pd.read_csv("test_patches.csv")
    test_df.label = test_df.label.apply(str)

    train_df, validation_df = train_test_split(df, test_size = 0.3, random_state = 5187)
    print (train_df)
    history = model.fit_df(train_df, validation_df, directory = "train/patches")

    test_metrics = model.evaluate_df(test_df, directory = "test/patches")
    test_loss, test_accuracy = test_metrics

    handle_history(model, history, "accuracy", test_accuracy)

    


    
