import spacy
import spacy.displacy as displacy

# Load the spaCy model with NER capabilities
nlp = spacy.load("en_core_web_sm")

# Input utterance
text = 'Set an alarm for 5:00pm on Saturday. I\'m going to San Francisco that day'

# Process the text and apply NER
doc = nlp(text)

# Extract entities and their types
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Print the extracted entities
print(f'Extracted entities: {entities}')

displacy.serve(doc, style='ent', auto_select_port=True)
