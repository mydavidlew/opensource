# flair
# A very simple framework for state-of-the-art Natural Language Processing (NLP)
# https://flairnlp.github.io/docs/intro
from flair.data import Sentence
from flair.nn import Classifier

# Text to analyze
text = """
The United States of America (USA), also known as the United States (U.S.) or America, is a country primarily located in North America. 
It is a federal republic of 50 states and a federal capital district, Washington, D.C. The 48 contiguous states border Canada to the 
north and Mexico to the south, with the semi-exclave of Alaska in the northwest and the archipelago of Hawaii in the Pacific Ocean. 
The United States also asserts sovereignty over five major island territories and various uninhabited islands in Oceania and the Caribbean. 
It is a megadiverse country, with the world's third-largest land area and third-largest population, exceeding 340 million.Paleo-Indians 
migrated from North Asia to North America over 12,000 years ago, and formed various civilizations and societies. Spanish exploration and 
colonization led to the establishment in 1513 of Spanish Florida, the first European colony in what is now the continental United States. 
Subsequent British colonization began with the first settlement of the Thirteen Colonies in Virginia in 1607. Forced migration of enslaved 
Africans provided the labor force necessary to make the plantation economy of the Southern Colonies economically viable. Clashes with the 
British Crown over taxation and the denial of parliamentary representation sparked the American Revolution, with the Second Continental 
Congress formally declaring independence on July 4, 1776. Victory in the 1775 – 1783 Revolutionary War brought international recognition 
of U.S. sovereignty, and the country continued to expand westward across North America, resulting in the dispossession of native inhabitants. 
As more states were admitted, a North–South division over slavery led the Confederate States of America to attempt secession, battling the 
states loyal to the Union in the 1861 – 1865 American Civil War. With the victory and preservation of the United States, the newly passed 
Thirteenth Amendment abolished slavery nationally. By 1900, the country had established itself as a great power, a status solidified after 
its involvement in World War I. Following Japan's attack on Pearl Harbor in December 1941, the U.S. entered World War II. Its aftermath 
left the U.S. and the Soviet Union as the world's superpowers and led to the Cold War, during which both countries struggled for ideological 
dominance and international influence. The Soviet Union's collapse and the end of the Cold War in 1991 left the U.S. as the world's sole superpower.  
"""

# [flair #1]
# make a sentence
sentence = Sentence(text)
# load the NER tagger
tagger = Classifier.load('ner')
# run NER over sentence
tagger.predict(sentence)
# print the sentence with all annotations
print(sentence)
# iterate and print
for entity in sentence.get_spans('ner'):
    print(entity)

# [flair #2]
# load the model
tagger = Classifier.load('linker')
# make a sentence
sentence = Sentence(text)
# predict entity links
tagger.predict(sentence)
# iterate over predicted entities and print
for label in sentence.get_labels():
    print(label)

# [flair #3]
# load the model
tagger = Classifier.load('pos')
# make a sentence
sentence = Sentence(text)
# predict NER tags
tagger.predict(sentence)
# print sentence with predicted tags
for label in sentence.get_labels():
    print(label)

# [flair #4]
# 1. make example sentence
sentence = Sentence(text)
# 2. load entity tagger and predict entities
tagger = Classifier.load('ner-fast')
tagger.predict(sentence)
# check which named entities have been found in the sentence
entities = sentence.get_labels('ner')
for entity in entities:
    print(entity)
# 3. load relation extractor
extractor = Classifier.load('relations')
# predict relations
extractor.predict(sentence)
# check which relations have been found
relations = sentence.get_labels('relation')
for relation in relations:
    print(relation)
# Use the `get_labels()` method with parameter 'relation' to iterate over all relation predictions.
for label in sentence.get_labels('relation'):
    print(label)

# [flair #5]
# Text to analyze
text = "I love Beijing, Berlin and New York but do not like London."
# make a sentence
sentence = Sentence(text)
# load the sentiment tagger
tagger = Classifier.load('sentiment')
# run sentiment analysis over sentence
tagger.predict(sentence)
# print the sentence with all annotations
print(sentence)
