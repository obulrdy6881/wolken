import pandas as pd
import html.parser
import re
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from sklearn.externals import joblib

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import session
# App config.
csv = pd.read_csv("cases.csv") # give csv path

print(csv.head())

csv.dropna(inplace=True)

stopwords_list = [word.strip() for word in open("stopwords-en.txt", 'r', encoding="utf8")]

#print(stopwords_list)


def data_cleaning(text):
    def escaping_html_chars(text_html):
        # print(text_html)
        html_parser = html.parser.HTMLParser()
        parsed_text = html_parser.unescape(text_html)
        return parsed_text

    def decode_text(text_decode):
        decoded_text = text_decode.encode('ascii', 'ignore').decode('utf-8')
        return str(decoded_text)

    def remove_punctuations_and_expressions(text_non_char):
        normalize_punctuations = ' '.join([word.lstrip(punctuation.replace(".", "")).rstrip(punctuation).strip() for word in text_non_char.split() if word not in punctuation])
        normalize_punctuations = ' '.join([re.sub("[{" + punctuation.replace("-", "").replace(".", "") + "}]", " ", word) for word in normalize_punctuations.split()])
        return normalize_punctuations

    def remove_stopwords(text_stopwords):
        stopwords_removed_text = ' '.join([word for word in text_stopwords.split() if word.lower() not in stopwords_list])
        return stopwords_removed_text

    def remove_numbers(text_num):
        numbers_removed_text = re.sub('\d+', " ", text_num)
        return numbers_removed_text

    def root_word_transform(text_root_form):
        transformed_text = ' '.join([text.vocab_info['rootForm'][word.lower().strip()] if word.lower().strip() in text.vocab_info['rootForm'] else word for word in text_root_form.split()])
        return transformed_text

    mod_text = escaping_html_chars(text)
    mod_text = decode_text(mod_text)
    mod_text = remove_punctuations_and_expressions(mod_text)
    text = remove_numbers(text)
    mod_text = remove_stopwords(mod_text)
    # mod_text = root_word_transform(mod_text)

    return mod_text.lower()


labels_with_more_data = [key for key, value in Counter(list(csv['Product'].values)).most_common() if value > 100]
labels, cleaned_data = [], []

for index, zip_data in enumerate(zip(csv['Product'], csv['Description'])):
    label, datum = zip_data
    #if index <= 1000:
    if label in labels_with_more_data:
        labels.append(label)
        cleaned_data.append(data_cleaning(datum))

print(len(labels), len(cleaned_data)) 

# for cd_index, cd in enumerate(cleaned_data):
#     if cd_index > 100:
#         break
#     print(cd)

X_train, X_test, y_train, y_test = train_test_split(cleaned_data, labels, random_state=42, test_size=0.40, stratify=labels, shuffle=True)

print(len(X_train), len(y_train))
print(len(X_test), len(y_test))

svc_clf = LinearSVC() # will go with this for now
# svc_clf = RandomForestClassifier()
#svc_clf = LogisticRegression()

vec = CountVectorizer(ngram_range=(1, 2))
# try (2, 3) and (1, 3)

model = Pipeline([
    ("feat_extractor", vec),
    ('classifier', svc_clf)
])
print("fitting model")
model.fit(X_train, y_train)
joblib.dump(model,'wolken.pkl')


print(model.score(X_test, y_test))
print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))




