import pandas as pd
import spacy
from spacy.util import minibatch
from spacy.training.example import Example
import random

if __name__ == '__main__':
    spam = pd.read_csv('spam.csv')
    print(spam.head(10))

    # Create an empty model
    nlp = spacy.blank("en")

    # Add the Text Categorize to the empty model
    text_cat = nlp.add_pipe("textcat")
    # Add labels to text classifier
    text_cat.add_label("ham")
    text_cat.add_label("spam")

    train_texts = spam['text'].values
    train_labels = [{'cats': {'ham': label == 'ham',
                              'spam': label == 'spam'}}
                    for label in spam['label']]

    train_data = list(zip(train_texts, train_labels))
    # show first 3 row from train data
    print(train_data[:3])

    spacy.util.fix_random_seed(1)
    optimizer = nlp.begin_training()

    # Create the batch generator with batch size = 8
    batches = minibatch(train_data, size=8)
    # Iterate through mini batches
    for batch in batches:
        # Each batch is a list of (text, label)
        for text, labels in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, labels)
            nlp.update([example], sgd=optimizer)

    random.seed(1)
    spacy.util.fix_random_seed(1)
    optimizer = nlp.begin_training()

    losses = {}
    print("losses : ")
    for epoch in range(10):
        random.shuffle(train_data)
        # Create the batch generator with batch size = 8
        batches = minibatch(train_data, size=8)
        # Iterate through minibatches
        for batch in batches:
            for text, labels in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, labels)
                nlp.update([example], sgd=optimizer, losses=losses)
        print(losses)

    texts = ["Are you ready for the tea party????? It's gonna be wild",
             "URGENT Reply to this message for GUARANTEED FREE TEA"]
    docs = [nlp.tokenizer(text) for text in texts]

    # Use text cat to get the scores for each doc
    text_cat = nlp.get_pipe('textcat')
    scores = text_cat.predict(docs)

    print("scores : ".format(scores))

    # From the scores, find the label with the highest score/probability
    predicted_labels = scores.argmax(axis=1)
    print("labels : ".format([text_cat.labels[label] for label in predicted_labels]))
