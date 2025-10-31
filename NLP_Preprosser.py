import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(raw_text):
    """
    Applies Tokenization, Part-of-Speech (POS) Tagging, and Lemmatization to the input text.
    """
    # Handle empty input
    if not raw_text or raw_text.strip() == "":
        print("--- Input is empty. Please enter text to process. ---")
        return [], [], []

    print(f"--- Processing: '{raw_text}' ---")

    # Tokenization: Splitting text into individual words and punctuation
    tokens = word_tokenize(raw_text)
    print(f"\n1. Tokens:\n{tokens}")

    # POS Tagging: Identifying the grammatical role of each token
    # (e.g., Noun, Verb, Adjective). The tagger uses the 'averaged_perceptron_tagger' resource.
    pos_tags = nltk.pos_tag(tokens)
    print(f"\n2. POS Tags (Token, Tag):\n{pos_tags}")

    #  Lemmatization: Reducing words to their base or dictionary form (lemma)
    lemmatizer = WordNetLemmatizer()
    lemmas = []

    # Lemmatization often requires knowing the POS (like 'v' for verb, 'n' for noun)
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN  # Default to Noun if unsure

    for token, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(token, wordnet_pos)
        lemmas.append(f"('{token}' -> '{lemma}', POS: {tag})")

    print(f"\n3. Lemmatization (Original -> Lemma, POS):\n{lemmas}")

    return tokens, pos_tags, lemmas


if __name__ == '__main__':
    print("--- NLTK Preprocessing Tool ---")
    user_input = input("Enter the text you want to analyze: ")
    preprocess_text(user_input)
