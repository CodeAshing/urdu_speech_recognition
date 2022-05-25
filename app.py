from flask import Flask, jsonify
import speech_recognition as sr
import boto3
import random

app = Flask(__name__)
 
 
region_name = 'me-south-1'


def success_message(data, message, code):
    return {'status': "success", 'data': data, 'message': message, 'code': code}


def error_message(message, code):
    return {'status': "error", 'data': [], 'message': message, 'code': code}


async def upload_audio_to_s3():

    audio_file = request.files.get('audio')

    file_name = str(random.randint(100, 10000))+'.wav'

    path = 'audio/' + file_name
    file_url = "https://recognizedaudios.s3.me-south-1.amazonaws.com/" + path
    bucket = 'recognizedaudios'

    s3 = boto3.resource('s3', region_name=region_name)

    object = s3.Object(bucket, path)
    s3_upload = object.put(Body=audio_file,
                           Metadata={'filename': file_name}
                           )

    if s3_upload['ResponseMetadata']['HTTPStatusCode'] == 200:

        file_dictionary = {"file_name": file_name, "file_url": file_url}

        return success_message(file_dictionary, "Audio file uploaded", 200)

    else:
        return error_message("Audio file not uploaded to S3", 409)


def upload_to_dynamoDB(file_url, google_response):

    TableName = 'recognised_audio_collection'

    dynamodb = boto3.client('dynamodb', region_name=region_name)

    dynamodb_response = dynamodb.put_item(
        TableName=TableName,
        Item={
            'source': {'S': file_url},
            'prediction': {'S': google_response}
        }
    )
    if dynamodb_response['ResponseMetadata']['HTTPStatusCode'] == 200:

        response_dictionary = {"file_url": file_url,
                               "prediction": google_response}

        return success_message(response_dictionary, "Record succesfully added to DynamoDB", 200)

    else:
        return error_message("Record does not added to DynamoDB", 409)


def google_prediction():

    try:

        predicted_text_urdu = ""
        AUDIO_FILE = '/tmp/audio.wav'

        audio = request.files.get('audio')
        audio.save(AUDIO_FILE)

        recognizer = sr.Recognizer()

        with sr.AudioFile(AUDIO_FILE) as source:
            audio = recognizer.record(source)  # read the entire audio file

        # recognize speech using Google Speech Recognition
        try:
            predicted_text_urdu = recognizer.recognize_google(
                audio, language="ur")

            response_dictionary = {"prediction": predicted_text_urdu}

            return success_message(response_dictionary, "Audio converted succesfully", 200)

        except sr.UnknownValueError:
            return error_message("Could not understand Audio", 409)
        except sr.RequestError as e:
            return error_message("Error: {0}".format(e), 409)

    except Exception as e:
        print(e)
        return error_message("Error: {0}".format(e), 409)


def get_data_from_dynamodb():

    TableName = 'recognised_audio_collection'

    dynamodb = boto3.client('dynamodb', region_name=region_name)

    dynamodb_response = dynamodb.scan(
        TableName=TableName
    )
    dynamodb_response = dynamodb_response['Items']

    if dynamodb_response:

        return success_message(dynamodb_response, "Record fetched succesfully", 200)

    else:
        return error_message("No record found", 404)


@app.route("/")
def init():
    return ('Hey there, I am running')


@app.route('/predict', methods=['POST'])
async def predictresult():

    google_response = google_prediction()

    if google_response['code'] == 200:
        prediction = google_response['data']['prediction']

        s3_upload_response = await upload_audio_to_s3()

        if s3_upload_response['code'] == 200:

            file_url = s3_upload_response['data']['file_url']

            dynamoDB_upload_response = upload_to_dynamoDB(
                file_url, prediction)

            return jsonify(dynamoDB_upload_response), dynamoDB_upload_response['code']

        else:
            return jsonify(s3_upload_response), s3_upload_response['code']

    else:
        return jsonify(google_response), google_response['code']


@app.route("/data", methods=['GET'])
def data():

    dynamoDB_response = get_data_from_dynamodb()

    return jsonify(dynamoDB_response), dynamoDB_response['code']


