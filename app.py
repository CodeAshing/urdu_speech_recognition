from flask import Flask, jsonify, request
import speech_recognition as sr
import boto3
import random
import torch
import os 
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)

# create app and load the trained Model
app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        ا 2
        ب 3
        پ 4
        ت 5
        ٹ 6
        ث 7
        ج 8
        چ 9
        ح 10
        خ 11
        د 12
        ڈ 13
        ذ 14
        ر 15
        ڑ 16
        ز 17
        ژ 18
        س 19
        ش 20
        ص 21
        ض 22
        ط 23
        ظ 24
        ع 25
        غ 26
        ف 27
        ق 28
        ک 29
        گ 30
        ل 31
        م 32
        ن 33
        ں 34
        و 35
        ہ 36
        ھ 37
        ی 38
        ے 39
        ئ 40
        ء 41
        آ 42
        اً 43
        ؤ 44
         ّ 45
         ٔ 46
         ً 47
         ُ 48
         َ 49
         ِ 50
         \u200c 51
        """
        self.char_map = {}
        self.index_map = {}
        
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
text_transform = TextTransform()


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class SpeechRecognitionModel(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

# Specify a path
PATH = "Trained_Model.pt"

# Load
trained_model = torch.load(PATH,map_location ='cpu')
trained_model.eval()

def GreedyDecoder(output, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes

def save_file():

    response = ""
    success = False

    # check if the post request has the file part
    if 'file' not in request.files:
        response = jsonify({'message' : 'No file part in the request'})
        response.status_code = 400
        return response
 
    files = request.files.getlist('file')
     
    for file in files:      
        if file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "download.wav"))
            success = True

    if not success:
        response = jsonify({'message' : 'Cannot find file in the request'})
        response.status_code = 500
        return response

def our_model_prediction(audio_file):

    # this try catch check if the audio is valid or not
    try:

        AUDIO_FILE = '/tmp/audio.wav'
        # save audio in temporary file
        audio_file.save(AUDIO_FILE)

        # wav, _ = sf.read(os.path.join(app.config['UPLOAD_FOLDER'], "download.wav"), dtype='float32')
        _, wav = wavfile.read(AUDIO_FILE,mmap=True)
        wav = wav.astype(np.float32, order='C') / 32768.0

        data = torch.from_numpy(wav)
        spectrograms = []
        spec = train_audio_transforms(data).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)

        output = trained_model(spectrograms)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)

        decoded_preds = GreedyDecoder(output.transpose(0, 1))

        print(decoded_preds)

        # Prepare response dictionary
        response_dictionary = {"prediction": decoded_preds[0]}

        return success_message(response_dictionary, "Audio converted succesfully", 200)

        # handle any exception while saving the audio file
    except Exception as e:
        return error_message("File is might not in correct format or File is corrupt ", 409)


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
def upload_to_dynamoDB(file_url, google_response):

    #  Get the DynamoDB object
    dynamodb = boto3.client('dynamodb', region_name=region_name)
    #  Upload the data to the table
    dynamodb_response = dynamodb.put_item(
        TableName=TableName,
        Item={
            'source': {'S': file_url},
            'prediction': {'S': google_response}
        }
    )

    # Check if the data is uploaded to S3 bucket then respond accordingly
    if dynamodb_response['ResponseMetadata']['HTTPStatusCode'] == 200:
        # Prepare response dictionary
        response_dictionary = {"file_url": file_url,
                               "prediction": google_response}

        return success_message(response_dictionary, "Record succesfully added to DynamoDB", 200)
    else:
        return error_message("Record does not added to DynamoDB", 409)


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
    prediction_response = our_model_prediction(audio_file)
    # check if the audio is converted to text
    if prediction_response['code'] == 200:
        # get the text from the response
        prediction = prediction_response['data']['prediction']
        # upload the audio to S3
        s3_upload_response = upload_audio_to_s3(audio_file)
        # check if the audio is uploaded to S3 bucket
        if s3_upload_response['code'] == 200:
            # get the audio file url from the response
            file_url = s3_upload_response['data']['file_url']
            # upload the file URL and prediction to DynamoDB
            dynamoDB_upload_response = upload_to_dynamoDB(
                file_url, prediction)

            return jsonify(dynamoDB_upload_response), dynamoDB_upload_response['code']
        else:
            return jsonify(s3_upload_response), s3_upload_response['code']
    else:
        return jsonify(prediction_response), prediction_response['code']

# Define route to get all data from DynamoDB
@app.route("/data", methods=['GET'])
def data():
    #  Get the data from DynamoDB
    dynamoDB_response = get_data_from_dynamodb()

    return jsonify(dynamoDB_response), dynamoDB_response['code']


if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()