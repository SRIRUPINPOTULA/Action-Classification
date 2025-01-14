# Import necessary libraries
import json
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel

import nltk
from nltk.tokenize import word_tokenize

import logging


def LDA(data, k, feature_space):

    # LDA works on documents, not on vectors
    string_data = []
    for sublist in data:
        string_data.append(' '.join(map(str, sublist)))

    # To remove Package download messages during runtime
    logging.getLogger('nltk').setLevel(logging.WARNING)
    
    # Download NLTK Sentence Tokenizer
    nltk.download('punkt', quiet=True)

    # Tokenize the document
    tokens = [word_tokenize(string) for string in string_data]

    # Create a dictionary representation of the document
    dictionary = corpora.Dictionary(tokens)

    # Create a corpus: List of bag-of-words
    corpus = [dictionary.doc2bow(token) for token in tokens]

    # Build the LDA model
    lda_model = LdaModel(corpus, num_topics=k, id2word=dictionary, passes=15)

    # Get the topic distribution for each document
    topic_distributions = [lda_model.get_document_topics(bow) for bow in corpus]

    # Convert to a feature matrix (dimensionality-reduced representation)
    def topic_vector(lda_output, num_topics):
        vector = np.zeros(num_topics)
        for topic_num, prob in lda_output:
            vector[topic_num] = prob
        return vector
    
    feature_matrix = np.array([topic_vector(doc, k) for doc in topic_distributions])
    lda_data_json = {}
    for index, i in enumerate(feature_matrix):
        lda_data_json[index*2] = i.tolist()

    # Print the topics found by the model
    print(f"Top-{k} latent Semantics for LDA")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")
    
    with open("../Outputs/Task_2/LDA_latent.json", "w") as json_file:
        json.dump(lda_model.print_topics(-1), json_file, indent=4)

    with open(f'../Outputs/Task_2/videoID-weight_LDA_{feature_space}.json', 'w') as f:
        json.dump(lda_data_json, f, indent=4)

    # Save the model and dictionary to reuse for next Task
    lda_model.save("../Outputs/Task_2/lda_model")
    dictionary.save("../Outputs/Task_2/dictionary")
