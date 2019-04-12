import spacy
import pandas as pd
from nltk.corpus import wordnet
import nltk
import re

INPUT_FILE = "TREC_10"
OUTPUT_FILE = INPUT_FILE+"_augmented"

def clean_str(string):
    """
    Cleans each string and convert to lower case. Copied from author's implementation.
    :param string: Each sentence from data set.
    :return: Processed sentence in lower case.
    """
    string = re.sub(r"[^A-Za-z0-9:(),!\'\`]", " ", string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

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
print(INPUT_FILE)
data_set = list(open('./trec/'+INPUT_FILE+'.label', encoding='utf-8', errors='replace').readlines())
data_set_cleaned = [clean_str(sent) for sent in data_set]
label = [s.split(" ")[0] for s in data_set_cleaned]
phrases = [s.split(" ")[1:] for s in data_set_cleaned]
print("Number of phrases in input file:", len(phrases))

# Generate paraphrases
print("Generating paraphrases...")
"""
Due to memory constrains, verbs are not expanded while generating synonyms.
"""
augmented_data = {}
for current_sentence in phrases:
    current_sentence = ' '.join(current_sentence)
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
        # if 'VB' in token.tag_: # if its a verb
        #     """Augment the verb with possible synonyms from Wordnet"""
        #     syns = wordnet.synsets(token.text,'v')
        #     synonyms = set()
        #     for eachSynSet in syns:
        #         for eachLemma in eachSynSet.lemmas():
        #             current_word = eachLemma.name()
        #             if current_word.lower() != token.text.lower() and current_word != token.lemma_:
        #                 synonyms.add(current_word.replace("_"," "))
        #     synonyms = list(synonyms)
        #     # print("\tCurrent verb word:", token.text, "(",len(synonyms),")")
        #     if len(synonyms) > 0:
        #         replacements[token.text] = synonyms
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

print("Saving to disk...")
# Save to disk as label
augmented_dataset = {'Phrases':[],'Paraphrases':[]}
phrases = []
paraphrases = []
i = -1
for key,value in augmented_data.items():
    i = i + 1
    for each_value in value:
        phrases.append(label[i])
        paraphrases.append(each_value)
augmented_dataset['Phrases'] = phrases
augmented_dataset['Paraphrases'] = paraphrases
augmented_dataset_df = pd.DataFrame.from_dict(augmented_dataset)
augmented_dataset_df.to_csv('./trec/'+OUTPUT_FILE+".label", encoding='utf8', index=False, header=False, sep=" ", doublequote=False)