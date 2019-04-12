## Data Augmentation for Text - Paraphrase Generation
Here an attempt is made to generate sensible text data from a given seed set using NLP (Natural Language Processing) techniques.

### The overall work flow is below:
1. A set of seed sentences are fed to the module as a CSV file.
2. Each line contains a single sentence in English.
3. A sentence is processed using **Spacy** nlp module to get the POS of each token.
4. For each token from the below category, synonyms are generated using **Wordnet**.
    - Noun which is not a named entity.
    - Adjectives
    - Verbs
5. Once the synonyms are created for each of the tokens, the list is filtered to retain the most sensible synonyms for the given context.
6. Spacy's token-to-token **similarity score** is used to weight each of the synonyms. Those token pair whose similarity score is less than a previously set threshold, is removed from the synonym list.
7. After the filtering process, the resulted synonyms are used to generate new sentences.
8. Finally, the augmented data set is stored to disk as a CSV file.

### Evaluation:
A sample data set is provided as seed set to check and evaluate the efficiency and usability of the approach. On manual inspection, the generated sentences look natural and grammatically valid almost always.

### Sample Use Case:
One can try the approach by running the script `generate_paraphrases.py`. This takes as input the supplied sample data (*input_data.csv*) and generates paraphrases which is stored to disk as *augmented_dataset.csv*.

### Dependencies:
1. Python 3.6 or higher
2. [Spacy](https://spacy.io/usage)
3. Spacy language model - Large. Eg: [`en_core_web_lg-2.0.0`](https://github.com/explosion/spacy-models)
4. NLTK Wordnet
5. Pandas (optional for reading/writing files)
