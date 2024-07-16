import json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

sys_template = '''
You are a helpful assistant who helps to create a customer review analysis system.
Read each review and determine whether the customer sentiment is positive or negative.
If the review is positive , create a thank you email to the customer and also recommend a new product for them to try.
If the review is negative , create an apology email to the customer and only for negative reviews , create an internal mail to be send to a senior customer service representative to address the customer concern.
'''

Prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_template),
        ("user", "{item}")
    ]
)

out = StrOutputParser()

chain = Prompt | llm | out

def load_reviews(file_path):
    with open(file_path, 'r') as file:
        reviews = json.load(file)
    return reviews

def load_products(file_path):
    with open(file_path, 'r') as file:
        products = json.load(file)
    return products

def save_reviews(reviews, file_path):
    with open(file_path, 'w') as file:
        json.dump(reviews, file, indent=4)

st.title("Customer Review Analysis System")

reviews_file = st.file_uploader("Upload the reviews.json file", type="json")
products_file = st.file_uploader("Upload the products.json file", type="json")

if reviews_file and products_file:
    reviews = json.load(reviews_file)
    products = json.load(products_file)

    processed_reviews = []

    for review in reviews:
        chats = [HumanMessage(content=review['text'])]
        response = chain.invoke({"chats": chats, "item": review['text']})
        sentiment, email = response.split('\n', 1) 

        
        st.write(f"Email to {review['customer_email']} : {email}")

        processed_reviews.append(review)

    save_reviews(processed_reviews, 'reviews_with_sentiment.json')

    st.success("Reviews have been processed and saved to reviews_with_sentiment.json")
