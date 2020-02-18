# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:45:10 2016

@author: DIP
"""

from .contractions import CONTRACTION_MAP
import re
import nltk
import string
import unidecode
from nltk.stem import WordNetLemmatizer
from html.parser import HTMLParser
import unicodedata
from tqdm.auto import tqdm
from nltk.corpus import wordnet as wn

import fr_core_news_sm
nlp_fr = fr_core_news_sm.load()

import en_core_web_sm
nlp_en = en_core_web_sm.load()

import de_core_news_sm
nlp_de = de_core_news_sm.load()

import es_core_news_sm
nlp_es = es_core_news_sm.load()

import it_core_news_sm
nlp_it = it_core_news_sm.load()

import pt_core_news_sm
nlp_pt = pt_core_news_sm.load()

import nl_core_news_sm
nlp_nl = nl_core_news_sm.load()

# global variables
wnl = WordNetLemmatizer()
html_parser = HTMLParser()
stopword_list = []
language = ""    

def init_lib(lang):    
    global stopword_list, language
    
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    language = lang
    print("Load "+lang+" language....")
    stopword_list = nltk.corpus.stopwords.words(lang)
        
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def expand_contractions(text, contraction_mapping):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Annotate text tokens with POS tags
def pos_tag_text(text):
    global language

    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('ADJ'):
            return wn.ADJ
        elif pos_tag.startswith('VERB'):
            return wn.VERB
        elif pos_tag.startswith('NOUN'):
            return wn.NOUN
        elif pos_tag.startswith('ADV'):
            return wn.ADV
        else:
            return None

    # ["english", "german", "french", "spanish", "portuguese", "italian", "dutch"]
    if (language=="french"):
        nlp = nlp_fr
    elif (language=="italian"):
        nlp = nlp_it
    elif (language=="spanish"):
        nlp = nlp_sp
    elif (language=="dutch"):
        nlp = nlp_nl
    elif (language=="german"):
        nlp = nlp_de             
    elif (language=="portuguese"):
        nlp = nlp_de        
    else:
        nlp = nlp_en
        
    tagged_text = nlp(text)
    
    tagged_lower_text = [(str(word).lower(), penn_to_wn_tags(word.pos_))
                         for word in
                         tagged_text]
    return tagged_lower_text

# lemmatize text based on POS tags
def lemmatize_text(text):
    global wnl
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text


def remove_special_characters(text):
    print("remove_special_characters")
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    # remove accent
    filtered_text = unidecode.unidecode(filtered_text)
    return filtered_text


def remove_stopwords(text):
    global stopword_list
    
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def sort_terms(text):
    tokens = tokenize_text(text)
    tokens.sort()
    filtered_text = ' '.join(tokens)
    return filtered_text

def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def unescape_html(parser, text):

    return parser.unescape(text)


def normalize_corpus(corpus, lemmatize=True,
                     only_text_chars=False,
                     tokenize=False, sort_text=False):
    global html_parser

    normalized_corpus = []
    
    for text in corpus:
        text = html_parser.unescape(text)
        text = expand_contractions(text, CONTRACTION_MAP)
        if lemmatize:
            text = lemmatize_text(text)
        else:
            text = text.lower()
        text = remove_stopwords(text)    
        text = remove_special_characters(text)
        if sort_text:
            text = sort_terms(text)
        if only_text_chars:
            text = keep_text_characters(text)

        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)

    return normalized_corpus


def parse_document(document):
    document = re.sub('\n', ' ', document)
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize('NFKD', document).encode('ascii', 'ignore')
    else:
        raise ValueError('Document is not string or unicode!')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences
