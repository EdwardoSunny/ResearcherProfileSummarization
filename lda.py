# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import nltk
nltk.download('stopwords')
from gensim.test.utils import common_texts
from gensim.models.ldamulticore import LdaModel
import io
import os.path
import re
import os
import sys
import json
import tqdm
from nltk.tokenize import RegexpTokenizer
import spacy
from gensim.models import Phrases
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from gensim.models import LdaModel

DATA_PATH = "cuimc_data/output/Abaci, Hasan.json"
researcher_dict = open(DATA_PATH, "r")
researcher_dict = json.load(researcher_dict)

# read in docs 
docs = []
docs_content = []
for pub in researcher_dict:
    docs.append(pub["Title"] + "\n" + pub["Abstract"])
    docs_content.append(pub["Title"] + "\n" + pub["Abstract"])

# tokenize ==============================
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.
# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
# Remove words that are only one character.
docs = [[token for token in doc if len(token) > 1] for doc in docs]


# lemmatize =============================
nlp = spacy.load('en_core_web_sm')
def lemmatize_tokens(token_list):
    doc = nlp(' '.join(token_list))
    return [token.lemma_ for token in doc]

docs = [lemmatize_tokens(tokens) for tokens in docs]

# bigrams/trigrams to docs ==============
bigram = Phrases(docs, min_count=10)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)


# remove stop words =====================
stop_words = set(stopwords.words('english'))

def remove_stop_words(word_list):
    return [word for word in word_list if word.lower() not in stop_words]
docs = [remove_stop_words(word_list) for word_list in docs]

# construct corpus ======================
dictionary = Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

print('>Finished preprocessing documents...')
print('\tNumber of unique tokens: %d' % len(dictionary))
print('\tNumber of documents: %d' % len(corpus))


print(">Begining Training of LDA Model...")
# Set training parameters.
num_topics = 7
chunksize = 5
passes = 20
iterations = 2000
eval_every = None # Don't evaluate model perplexity, takes too much time. (or 1)

# Make an index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)
print(">Done training! Outputting Statistics")
top_topics = model.top_topics(corpus)
# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

print(">Splitting Documents into Groups by Topic...")
doc_topics = [model.get_document_topics(bow, minimum_probability=0) for bow in corpus]
def assign_topic_distribution(doc_topics):
    topic_assignments = []
    for topics in doc_topics:
        # Sort topics by probability in descending order
        sorted_topics = sorted(topics, key=lambda x: -x[1])
        # Get the most dominant topic
        dominant_topic = sorted_topics[0][0] if sorted_topics else None
        topic_assignments.append(dominant_topic)
    return topic_assignments

# Get the topic assignments
topic_assignments = assign_topic_distribution(doc_topics)

# Extract topic names
num_words = 10  # Number of top words to display per topic
topics_info = model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
# Create a dictionary mapping topic numbers to topic names
topic_names = {}
for topic_num, topic_words in topics_info:
    words = [word for word, _ in topic_words]
    topic_names[topic_num] = " ".join(words)

# Output the topic assignments with topic names
grouped_docs = [[] for _ in range(num_topics)]

# Group documents by their assigned topic
for i, topic in enumerate(topic_assignments):
    topic_name = topic_names.get(topic, "Unknown Topic")
    print(f"Document {i} is assigned to topic {topic} ({topic_name})")
    # Ensure topic index is valid
    if topic < num_topics:
        grouped_docs[topic].append(docs_content[i])
    else:
        print(f"Warning: Document {i} has an invalid topic assignment {topic}")

count = 0
for topic_id, docs in enumerate(grouped_docs):
    print(f"\nDocuments in topic {topic_id} ({topic_names.get(topic_id, 'Unknown Topic')}):")
    for doc in docs:
        print(doc[0:200])  # Adjust the printing format as needed
        count += 1
print(count)
# grouped_docs has grouped by topics [[stuf...], [aigjqigo...], ...] --> 7 docs means 7 inner list
