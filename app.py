from flask import Flask
import speech_recognition as sr
import boto3
import random

app = Flask(__name__)
 
 
@app.route("/")
def hello():
    return "Hello World!"
