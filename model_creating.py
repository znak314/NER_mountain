import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from simpletransformers.ner import NERModel, NERArgs


def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path, encoding="utf-8")


def prepare_data(data):
    """Prepare training and test datasets."""
    X = data[["sentence_id", "word", "POS"]]
    Y = data["label"]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

    train_data = pd.DataFrame({
        "sentence_id": x_train["sentence_id"],
        "words": x_train["word"],
        "POS": x_train["POS"],
        "labels": y_train
    })

    test_data = pd.DataFrame({
        "sentence_id": x_test["sentence_id"],
        "words": x_test["word"],
        "POS": x_test["POS"],
        "labels": y_test
    })

    return train_data, test_data


def create_model(labels):
    """Create and return a NER model."""
    args = NERArgs(
        num_train_epochs=10,
        learning_rate=2e-4,
        overwrite_output_dir=True,
        train_batch_size=16,
        eval_batch_size=16,
        output_dir='trained_ner_model'
    )

    return NERModel('bert', 'bert-base-cased', labels=labels, args=args, use_cuda=False)


def train_and_evaluate_model(model, train_data, test_data):
    """Train and evaluate the NER model."""
    model.train_model(train_data, eval_data=test_data, acc=accuracy_score)
    result, model_outputs, preds_list = model.eval_model(test_data)

    return result


def main():
    # Load data
    data = load_data("test_task_NER_dataset.csv")

    # Prepare training and test data
    train_data, test_data = prepare_data(data)

    # Create model
    dataset_labels = data["label"].unique().tolist()
    model = create_model(dataset_labels)

    # Train and evaluate the model
    evaluation_result = train_and_evaluate_model(model, train_data, test_data)

    print("Model evaluation:")
    print(evaluation_result)


if __name__ == "__main__":
    main()
