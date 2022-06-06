from flask import Flask, jsonify, request
import speech_recognition as sr
from googletrans import Translator, constants
import boto3
import random

app = Flask(__name__)

# Define region name
region_name = 'me-south-1'
#  Define table name
TableName = 'recognised_audio_collection'

# Define success message function
def success_message(data, message, code):
    return {'status': "success", 'data': data, 'message': message, 'code': code}

# Define success error function
def error_message(message, code):
    return {'status': "error", 'data': [], 'message': message, 'code': code}

# Define function to upload audio to S3
def upload_audio_to_s3(audio_file):

    # Generate random file name with .wav extension
    file_name = str(random.randint(100, 10000))+'.wav'

    path = 'audio/' + file_name
    file_url = "https://recognizedaudios.s3.me-south-1.amazonaws.com/" + path
    bucket = 'recognizedaudios'

    #  Get the S3 object
    s3 = boto3.resource('s3', region_name=region_name)
    object = s3.Object(bucket, path)
    #  Upload the audio to the bucket
    s3_upload = object.put(Body=audio_file,
                           Metadata={'filename': file_name}
                           )
    # Check if the audio is uploaded to S3 bucket then respond accordingly
    if s3_upload['ResponseMetadata']['HTTPStatusCode'] == 200:
        # Prepare response dictionary
        file_dictionary = {"file_name": file_name, "file_url": file_url}

        return success_message(file_dictionary, "Audio file uploaded", 200)
    else:
        return error_message("Audio file not uploaded to S3", 409)

# Define function to upload details to DynamoDB
def upload_to_dynamoDB(file_url, google_response,english_prediction):

    #  Get the DynamoDB object
    dynamodb = boto3.client('dynamodb', region_name=region_name)
    #  Upload the data to the table
    dynamodb_response = dynamodb.put_item(
        TableName=TableName,
        Item={
            'source': {'S': file_url},
            'urdu': {'S': google_response},
            'english': {'S': english_prediction}
        }
    )

    # Check if the data is uploaded to S3 bucket then respond accordingly
    if dynamodb_response['ResponseMetadata']['HTTPStatusCode'] == 200:
        # Prepare response dictionary
        response_dictionary = {"file_url": file_url,
                               "urdu": google_response,
                               "english": english_prediction}

        return success_message(response_dictionary, "Record succesfully added to DynamoDB", 200)
    else:
        return error_message("Record does not added to DynamoDB", 409)

# Define function to get the prediction
def google_prediction(audio):

    # this try catch check if the audio is valid or not
    try:

        predicted_text_urdu = ""
        predicted_text_eng = ""
        AUDIO_FILE = '/tmp/audio.wav'
        # save audio in temporary file
        audio.save(AUDIO_FILE)
        # this try catch handle any exception while reading the audio file
        try:
            # Initialis the recognizer
            recognizer = sr.Recognizer()
            # read the audio file from temporary file
            with sr.AudioFile(AUDIO_FILE) as source:
                audio = recognizer.record(source)  # read the entire audio file

            # this try catch handle any exception while converting audio to text
            try:
                # Get the text from audio
                predicted_text_urdu = recognizer.recognize_google(
                    audio, language="ur-PK")

                translator = Translator()
                predicted_text_eng = translator.translate(predicted_text_urdu)
                predicted_text_eng = predicted_text_eng.text

                # Prepare response dictionary
                response_dictionary = {"prediction": predicted_text_urdu,"english_prediction":predicted_text_eng}

                return success_message(response_dictionary, "Audio converted succesfully", 200)
            except sr.UnknownValueError:
                return error_message("Could not understand Audio", 409)
            except sr.RequestError as e:
                return error_message("Error: {0}".format(e), 409)
        # handle any exception while reading the audio file
        except Exception as e:
            print(e)
            return error_message("Error: {0}".format(e), 409)
    # handle any exception while saving the audio file
    except Exception as e:
        return error_message("File is might not in correct format or File is corrupt ", 409)

# Define function to get all data from DynamoDB
def get_data_from_dynamodb():
    #  Get the DynamoDB object
    dynamodb = boto3.client('dynamodb', region_name=region_name)
    #  Get all data from the table
    dynamodb_response = dynamodb.scan(
        TableName=TableName
    )
    #  Filter the response object
    dynamodb_response = dynamodb_response['Items']
    #  check if the data is present in the table
    if dynamodb_response:
        return success_message(dynamodb_response, "Record fetched succesfully", 200)
    else:
        return error_message("No record found", 404)

# root route
@app.route("/")
def init():
    return jsonify(success_message([], "Hey there, I am running", 200)), 200

# Define route to get the prediction
@app.route('/predict', methods=['POST'])
def predictresult():
    #  Get the audio file from the request
    audio_file = request.files.get('audio')
    # convert audio file to text
    google_response = google_prediction(audio_file)
    # check if the audio is converted to text
    if google_response['code'] == 200:
        # get the text from the response
        prediction = google_response['data']['prediction']
        english_prediction = google_response['data']['english_prediction']
        # upload the audio to S3
        s3_upload_response = upload_audio_to_s3(audio_file)
        # check if the audio is uploaded to S3 bucket
        if s3_upload_response['code'] == 200:
            # get the audio file url from the response
            file_url = s3_upload_response['data']['file_url']
            # upload the file URL and prediction to DynamoDB
            dynamoDB_upload_response = upload_to_dynamoDB(
                file_url, prediction,english_prediction)

            return jsonify(dynamoDB_upload_response), dynamoDB_upload_response['code']
        else:
            return jsonify(s3_upload_response), s3_upload_response['code']
    else:
        return jsonify(google_response), google_response['code']

# Define route to get all data from DynamoDB
@app.route("/data", methods=['GET'])
def data():
    #  Get the data from DynamoDB
    dynamoDB_response = get_data_from_dynamodb()

    return jsonify(dynamoDB_response), dynamoDB_response['code']
