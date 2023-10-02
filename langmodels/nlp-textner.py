from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

model1a = "dslim/bert-base-NER"; model1b = "dslim/bert-large-NER"
model2a = "dbmdz/bert-base-cased-finetuned-conll03-english"; model2b = "dbmdz/bert-large-cased-finetuned-conll03-english"
model3a = "allenai/tk-instruct-base-def-pos"; model3b = "allenai/tk-instruct-large-def-pos"
model4a = "flair/pos-english-fast"; model4b = "flair/pos-english"

model5 = "dbmdz/bert-large-cased-finetuned-conll03-english"
token5 = "bert-base-cased"

# Load the NER model
modelx = model5; tokenx = token5
model = AutoModelForTokenClassification.from_pretrained(modelx)
tokenizer = AutoTokenizer.from_pretrained(tokenx)

# Load the NER pipeline
ner_model = pipeline(task="ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Text to analyze
text1 = "Hugging Face is a technology company based in New York and Paris. David Lew drives a Proton Iriz."
text = """
A US jury found Roger Ng guilty on all charges in the trial, which concerned the looting of billions of dollars from Malaysia's 1MDB sovereign wealth fund.
He is the only Goldman Sachs banker to face a jury over the scandal, which rocked Malaysian politics and forced the bank to pay billions in fines.
Mr Ng's lawyer said he was a "fall guy "and attacked the credibility of the government's star witness, Tim Leissner, who was Mr Ng's boss at Goldman Sachs and pleaded guilty to his role in the scandal in 2018.
But after a nearly two-month trial and several days of deliberation, the jury in New York convicted Mr Ng of conspiring to launder money and violating an anti-corruption law.
US prosecutors said the decision was "a victory for not only the rule of law, but also for the people of Malaysia".
"""

# Perform NER on the text
ner_results = ner_model(text)
token_results = tokenizer(text, return_tensors="pt")
print("ner_results:", ner_results)
print("token_results:", token_results['input_ids'])
#print("token_results.input_ids:", token_results['input_ids'])
#print("token_results.token_type_ids:", token_results['token_type_ids'])
#print("token_results.attention_mask:", token_results['attention_mask'])

# Display the named entities
for entity in ner_results:
    print(f'{entity["entity_group"]:<5}', ":", f'{entity["score"]:<20}', ":", entity["word"])
