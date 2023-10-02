from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from transformers import pipeline

model1a = "dslim/bert-base-NER"; model1b = "dslim/bert-large-NER"
model2a = "dbmdz/bert-base-cased-finetuned-conll03-english"; model2b = "dbmdz/bert-large-cased-finetuned-conll03-english"
model3a = "allenai/tk-instruct-base-def-pos"; model3b = "allenai/tk-instruct-large-def-pos"
model4a = "flair/pos-english-fast"; model4b = "flair/pos-english"

model = model1b
sentence = """
A US jury found Roger Ng guilty on all charges in the trial, which concerned the looting of billions of dollars from Malaysia's 1MDB sovereign wealth fund.
He is the only Goldman Sachs banker to face a jury over the scandal, which rocked Malaysian politics and forced the bank to pay billions in fines.
Mr Ng's lawyer said he was a "fall guy "and attacked the credibility of the government's star witness, Tim Leissner, who was Mr Ng's boss at Goldman Sachs and pleaded guilty to his role in the scandal in 2018.
But after a nearly two-month trial and several days of deliberation, the jury in New York convicted Mr Ng of conspiring to launder money and violating an anti-corruption law.
US prosecutors said the decision was "a victory for not only the rule of law, but also for the people of Malaysia".
"""

token_classifier = pipeline(task="ner", model=model, aggregation_strategy="simple")
tokens = token_classifier(sentence)
print(tokens)
# Start and end provide an easy way to highlight words in the original text.
for token in tokens:
    print(token["entity_group"], ":", sentence[token["start"] : token["end"]])


# Some models use the same idea to do part of speech.
model_name = "vblagoje/bert-english-uncased-finetuned-pos"
syntaxer = pipeline(model=model_name, aggregation_strategy="simple")
text="My name is Sarah and I live in London"
results = syntaxer(sentence)
print(results)
for result in results:
    print(result["entity_group"], ":", result["word"])


model_name = "QCRI/bert-base-multilingual-cased-pos-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
outputs = pipeline("David is an intelligent individual, he works at Mimos Inc situated at Bukit Jalil, Kuala Lumpur")
print(outputs)
for i, output in enumerate(outputs):
        print(i+1, ":", output)
