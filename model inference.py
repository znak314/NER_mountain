from simpletransformers.ner import NERModel
import re

# Initialize the NER model
model_directory = "model/"
model = NERModel(model_type='bert', model_name=model_directory, use_cuda=False)

def highlight_mountains(input_sentence, model_prediction):
    """
    Highlights mountain names in the input sentence using the model's predictions.

    Args:
        input_sentence (str): The sentence containing potential mountain names.
        model_prediction (list): The model's output predictions for the sentence.

    Returns:
        str: The input sentence with highlighted mountain names.
    """
    mountains = []  # List to store identified mountain names
    current_mountain = []  # Temporary list to build the current mountain name

    for word in model_prediction[0]:
        # Check for the beginning of a mountain name
        if 'B-MOUNT' in word.values():
            if current_mountain:
                # Store the previous mountain name if exists
                mountains.append(' '.join(current_mountain))
                current_mountain = []  # Reset for a new mountain
            current_mountain.append(list(word.keys())[0])  # Add the new mountain name
        elif 'I-MOUNT' in word.values() and current_mountain:
            # Continue adding to the current mountain name
            current_mountain.append(list(word.keys())[0])

    # Add the last mountain name if it exists
    if current_mountain:
        mountains.append(' '.join(current_mountain))

    # Replace mountain names in the input sentence with highlighted versions
    for mountain in mountains:
        input_sentence = re.sub(r'(?<!\w)' + re.escape(mountain) + r'(?!\w)',
                                f'**{mountain}** (Mountain)', input_sentence)

    return input_sentence


def process_prediction(input_sentence):
    """
    Processes the input sentence, predicting mountain names and highlighting them.

    Args:
        input_sentence (str): The input sentence to be processed.

    Returns:
        str: The highlighted sentence with mountain names.
    """
    prediction, model_output = model.predict([input_sentence])

    # Highlight mountain names in the input sentence
    highlighted_sentence = highlight_mountains(input_sentence, prediction)
    return highlighted_sentence


def main():
    # Example input sentence
    input_sentence = ("While climbing the majestic peaks of the Himalayas, "
                      "including Mount Everest and Kanchenjunga, the mountaineers "
                      "marveled at the breathtaking views and the unique challenges "
                      "presented by each summit.")

    # Process the input sentence
    highlighted_sentence = process_prediction(input_sentence)
    print(highlighted_sentence)


if __name__ == "__main__":
    main()
