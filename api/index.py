from flask import Flask,jsonify, request
import openai
import os
os.environ["OPENAI_API_KEY"]  = OPENAI_API_KEY

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

llm =  ChatOpenAI()
memory = ConversationBufferMemory()

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/conversation',methods=['POST'])
def conversation_func(user_input):
    template = """ You are a conversational Bot. Respond politely in just one to two short sentences to human input.
    Current conversation:
    {history}
    Human: {input}
    Bot:
    """
    prompt_template = PromptTemplate(input_variables=["input","history"], template=template)

    conversation = ConversationChain(
        llm=llm,
        prompt=prompt_template,
        verbose=False,
        memory=memory
    )
    return jsonify(conversation.predict(input=user_input))