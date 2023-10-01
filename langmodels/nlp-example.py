from transformers import pipeline

classifier = pipeline(task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
# No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english
sentences = ["I am thrilled to introduce you to the wonderful world of AI.", "Hopefully, it won't disappoint you.", "DavidLew is a genius"]
results = classifier(sentences)
print(results)
for i, result in enumerate(results):
    print(f"Result {i + 1}:")
    print(f" Label: {result['label']}")
    print(f" Score: {round(result['score'], 3)}\n")


ner_tagger = pipeline(task="ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
# No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english
text = "Elon Musk is the CEO of SpaceX Inc."
results = ner_tagger(text)
print(results)
for i, result in enumerate(results):
    print(f"Result {i + 1} : {result}")


reader = pipeline(task="question-answering", model="distilbert-base-cased-distilled-squad")
# No model was supplied, defaulted to distilbert-base-cased-distilled-squad
text = "Hugging Face is a company creating tools for NLP. It is based in New York and was founded in 2016. DavidLew exam score is 100%. He live in Puchong"
question1 = "Where is Hugging Face based?"
question2 = "Where DavidLew live?"
results = reader(question=question2, context=text)
print(results)
