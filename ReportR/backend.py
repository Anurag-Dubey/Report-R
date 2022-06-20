import tensorflow_hub as hub
from official.nlp import optimization
import tensorflow as tf
import torch
import tensorflow_text
import socket

import os
from flask import Flask,jsonify,Response,json, request
from flask_cors import CORS, cross_origin
from werkzeug.datastructures import ImmutableMultiDict
from werkzeug.utils import secure_filename
from googletrans import Translator
import numpy as np
import tweepy
import pymongo

CONSUMER_KEY='1FUBvVQoujMdqrqtWwdyQKbLb'
CONSUMER_SECRET='TZ7QPTgkw9whW8894cEqReDQ95qfzzkDFWZqU5dKdRXd4aKJv3'
BEARER_TOKEN='AAAAAAAAAAAAAAAAAAAAAPjadwEAAAAAypz6Td6sswWl%2BiXCob3O6e2cxl8%3DGKPknk8z7baDHiten0P4yUsaykv1tFwcZNu3Rv4vM3EtN61hCV'
ACCESS_TOKEN='1301426261156225024-MoZdD7DrIZsiTX68wysvFwkZUdJaVe'
ACCESS_TOKEN_SECRET='ytdNJ2WaNSQUA5VsPXHWznBh1DbtRL51iOPrML76URxg9'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

client = pymongo.MongoClient("mongodb+srv://demo:test@cluster0.lhsd2.mongodb.net/?retryWrites=true&w=majority", tls=True, tlsAllowInvalidCertificates=True)
db = client.test
dbname = client.get_database('sentiments')
collection_name = dbname["records"]


translator = Translator()

UPLOAD_FOLDER = './store/'

app = Flask(__name__,template_folder='template',static_url_path='/',static_folder='./')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs_count = 6#defining_epochs_count
steps_each_epoch = 500
number_of_training_steps = steps_each_epoch * epochs_count#defining_Training_steps
number_of_warmup_steps = int(0.1*number_of_training_steps)#defining_warmup_steps
init_lr = 3e-5 #deifning learning rate
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=number_of_training_steps,
                                          num_warmup_steps=number_of_warmup_steps,
                                          optimizer_type='adamw')


loaded_model = tf.keras.models.load_model('model_keras.h5', custom_objects={'KerasLayer':hub.KerasLayer , 'AdamWeightDecay': optimizer})


            
def recognise(text):
    text=translator.translate(text, dest='en').text
    print(text)
    try:
        result = loaded_model.predict([text])
        index_max = np.argmax(result[0])
        socket.getaddrinfo('localhost', 8080)
    except:
        print('Error, Try Again!')
    return 'hate' if index_max == 0 else 'offensive' if index_max == 1 else 'neutral'

@app.route('/report', methods=['POST'])
def report():
    # try:
    data=request.get_json()
    link=data['link']
    if('twitter' in link):
        id=link.split('/')[-1]
        fetched=api.get_status(id)
        push_to_mongo=recognise(fetched.text)
        if(collection_name.find_one({'id':id})):
            collection_name.update_one(
            {"id": id},
            { '$inc': { 'count': 1 } },
            upsert=True
        )
        else:
            collection_name.insert_one({"id":id, "text": fetched.text, "report_count":0, "severity": push_to_mongo})
    # except:
    #     print('Error, Try Again!')
    return jsonify({'status':'200'})

if __name__ == '__main__':
       app.run(host="0.0.0.0", port=8080)

