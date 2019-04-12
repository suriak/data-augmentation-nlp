import spacy
import pandas as pd
from nltk.corpus import wordnet
import nltk
import re


file_name = "input_data.csv"

def get_wordvector_similarity(nlp,replacements):
    """
    From the list of synonyms obtained from Wordnet, apply the 
    similarity score to filter out non-relevant synonyms. The word pair who has similarity score less than 
    THRESHOLD is neglected.
    """
    replacements_refined = {}
    THRESHOLD = 0.44
    for key, values in replacements.items():
        key_vec = nlp(key.lower())
        synset_refined = []
        for each_value in values:
            value_vec = nlp(each_value.lower())
            if key_vec.similarity(value_vec) > THRESHOLD:
                synset_refined.append(each_value)
        if len(synset_refined) > 0:
            replacements_refined[key] = synset_refined
    return replacements_refined

nlp = spacy.load('/en_core_web_lg/en_core_web_lg-2.0.0/')

# Load data set
print("Reading input file...")
dataset_df = pd.read_csv(file_name,encoding='utf8')
phrases = dataset_df['Phrases']
print("Number of phrases in input file:", len(phrases))

# Generate paraphrases
print("Generating paraphrases...")
augmented_data = {}
for current_sentence in phrases:
    # print("\tCurrent input sentence:",current_sentence)
    doc = nlp(current_sentence)
    replacements = {}
    for token in doc:
        if 'NN' in token.tag_ and token.ent_type == 0: # if its a noun and not a NER
            """Augment the noun with possible synonyms from Wordnet"""            
            syns = wordnet.synsets(token.text,'n')
            synonyms = set()
            for eachSynSet in syns:
                for eachLemma in eachSynSet.lemmas():
                    current_word = eachLemma.name()
                    if current_word.lower() != token.text.lower() and current_word != token.lemma_:
                        synonyms.add(current_word.replace("_"," "))
            synonyms = list(synonyms)
            # print("\tCurrent noun word:", token.text, "(",len(synonyms),")")
            if len(synonyms) > 0:
                replacements[token.text] = synonyms
        if 'JJ' in token.tag_: # if its an adjective
            """Augment the adjective with possible synonyms from Wordnet"""
            syns = wordnet.synsets(token.text,'a')
            synonyms = set()
            for eachSynSet in syns:
                for eachLemma in eachSynSet.lemmas():
                    current_word = eachLemma.name()
                    if current_word.lower() != token.text.lower() and current_word != token.lemma_:
                        synonyms.add(current_word.replace("_"," "))
            synonyms = list(synonyms)
            # print("\tCurrent adjective word:", token.text, "(",len(synonyms),")")
            if len(synonyms) > 0:
                replacements[token.text] = synonyms
        if 'VB' in token.tag_: # if its a verb
            """Augment the verb with possible synonyms from Wordnet"""
            syns = wordnet.synsets(token.text,'v')
            synonyms = set()
            for eachSynSet in syns:
                for eachLemma in eachSynSet.lemmas():
                    current_word = eachLemma.name()
                    if current_word.lower() != token.text.lower() and current_word != token.lemma_:
                        synonyms.add(current_word.replace("_"," "))
            synonyms = list(synonyms)
            # print("\tCurrent verb word:", token.text, "(",len(synonyms),")")
            if len(synonyms) > 0:
                replacements[token.text] = synonyms
    # print("Input(before filtering):\n",sum(map(len, replacements.values())))
    replacements_refined = get_wordvector_similarity(nlp,replacements)
    # print("Output(after filtering based on similarity score):\n",sum(map(len, replacements_refined.values())))
    # print ("\tReplacements:", replacements_refined)
    generated_sentences = []
    generated_sentences.append(current_sentence)
    for key, value in replacements_refined.items():
        replaced_sentences = []
        for each_value in value:
            for each_sentence in generated_sentences:
                new_sentence = re.sub(r"\b%s\b" % key,each_value,each_sentence)
                replaced_sentences.append(new_sentence)
        generated_sentences.extend(replaced_sentences)
    augmented_data[current_sentence] = generated_sentences    

print("#####################--Paraphrase generation completed--#####################")
print("Total variations created:", sum(map(len, augmented_data.values())))
# print("Each set is shown below:")
# for key in augmented_data.keys():
#     print("Seed sentence:-", key)
#     print("Augmented sentence:-", augmented_data[key],"\n")

print("Saving to disk as CSV...")
# Save to disk as csv
augmented_dataset = {'Phrases':[],'Paraphrases':[]}
phrases = []
paraphrases = []
for key,value in augmented_data.items():
    for each_value in value:
        phrases.append(key)
        paraphrases.append(each_value)
augmented_dataset['Phrases'] = phrases
augmented_dataset['Paraphrases'] = paraphrases
augmented_dataset_df = pd.DataFrame.from_dict(augmented_dataset)
augmented_dataset_df.to_csv("augmented_dataset.csv", encoding='utf8', index=False)