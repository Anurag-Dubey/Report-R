#-------------- For Transfomers --------------

pip install tensorflow-text==2.4.1
import tensorflow as tf%tensorflow_version 2.x
import tensorflow as tf
pip install transformers
import os
import shutil
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy.stats import spearmanr
from math import floor, ceil
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Dropout, Activation, Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
import shutil

!pip install tf-models-official
from official.nlp import optimization

#-------------- For Glove --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import keras
from keras.models import Sequential
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau,CSVLogger
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential,LSTM,BatchNormalization,Bidirectional
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Embedding,Conv1D, GlobalMaxPooling1D, 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,classification_report
import shutil

#-------------- For TF-IDF --------------
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
import pandas as pd
import numpy as np
from itertools import chain
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import Embedding, BatchNormalization, Dense, Flatten, Dropout, Bidirectional,Flatten, GlobalMaxPool1D,Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau,CSVLogger
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from keras.utils import np_utils
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import xgboost as xgb
from sklearn import svm