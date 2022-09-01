import pandas as pd
import streamlit as st
import requests
from models import SupervisedModel, UnSupervisedModel,DirectMatching
from preprocess import preprocess_body,preprocess_title

def title_tokens(title):
    return ['a','b']

def question_tokens(title):
    return ['d','e','c']

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

unsup_model = UnSupervisedModel()
sup_model  = SupervisedModel()
direct_matcher = DirectMatching()


def main():
    API_URI = 'http://127.0.0.1:8000/tags'

    st.title('Inputs')

    #question title
    title = st.text_input("Question title")

    #question 
    question = st.text_area("Question")

    #prediction
    predict_btn = st.button('Guess Tags')
    tags = None
    if predict_btn:
        title_tokens  = preprocess_title(title)
        quest_tokens  = preprocess_body(question)
        data = title_tokens + quest_tokens
        unsup_tags = unsup_model.predict(data)
        sup_tags   = sup_model.predict(data)
        direct_tags = direct_matcher.predict(data)

        tags = list(set(unsup_tags).union(sup_tags).union(direct_tags)) #request_prediction(API_URI, data)[0]

    #output
    st.title('Suggested tags:')
    if tags is not None:
        if len(tags) > 0 :
            st.write(f'{tags}')
        else:
            st.write('No tags found')


if __name__ == '__main__':
    main()
