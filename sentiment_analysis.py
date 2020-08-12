vocab={}
word_encoding=1
def bag_of_words(text):
    global word_encoding
    
    words=text.lower().split(" ")
    bag={}
    
    for word in words:
        if word in vocab:
            encoding=vocab[word]
            
        else:
            vocab[word]=word_encoding
            encoding=word_encoding
            word_encoding+=1
        if encoding in bag:
            bag[encoding]+=1
        else:
            bag[encoding]=1
    return bag
text="kabhi kabhi lagta hai apun hi bhagwan hai"
bag=bag_of_words(text)            
print(bag)
print(vocab)

vocab = {}  # maps word to integer representing it
word_encoding = 1
def bag_of_words(text):
  global word_encoding

  words = text.lower().split(" ")  # create a list of all of the words in the text, well assume there is no grammar in our text for this example
  bag = {}  # stores all of the encodings and their frequency

  for word in words:
    if word in vocab:
      encoding = vocab[word]  # get encoding from vocab
    else:
      vocab[word] = word_encoding
      encoding = word_encoding
      word_encoding += 1
    
    if encoding in bag:
      bag[encoding] += 1
    else:
      bag[encoding] = 1
  
  return bag

text = "this is a test to see if this test will work is is test a a"
bag = bag_of_words(text)
print(bag)
print(vocab)


vocab={}
word_encoding=1
def bag_of_words(text):
    global word_encoding
    
    words=text.lower().split(" ")
    bag={}
    for word in words:
        if word in vocab:
            encoding = vocab[word]
        else:
            vocab[word]=word_encoding
            encoding=word_encoding
            word_encoding += 1
        if encoding in bag:
            bag[encoding]+=1
        else:
            bag[encoding]=1
    return bag
text="kabhi kabhi lagta hai apunich bhagwan hai"
bag=bag_of_words(text)
print(bag)
print(vocab)


vocab = {}  
word_encoding = 1
def one_hot_encoding(text):
  global word_encoding

  words = text.lower().split(" ") 
  encoding = []  

  for word in words:
    if word in vocab:
      code = vocab[word]  
      encoding.append(code) 
    else:
      vocab[word] = word_encoding
      encoding.append(word_encoding)
      word_encoding += 1
  
  return encoding

text = "this is a test to see if this test will work is is test a a"
encoding = one_hot_encoding(text)
print(encoding)
print(vocab)

vocab={}
word_encoding=1
def one_hot_encoding(test):
    global word_encoding
    
    words=text.lower().split(" ")
    encoding=[]
    
    for word in words:
        if word in vocab:
            code=vocab[word]
            encoding.append(code)
        else:
            vocab[word]=word_encoding
            encoding.append(word_encoding)
            word_encoding+=1
    return encoding
text="this is to see if this test is to see is going to be successfull to be honest"
bag=one_hot_encoding(text)
print(bag)
print(vocab)

positive_review = "I thought the movie was going to be bad but it was actually amazing"
negative_review = "I thought the movie was going to be amazing but it was actually bad"

pos_encode = one_hot_encoding(positive_review)
neg_encode = one_hot_encoding(negative_review)

print("Positive:", pos_encode)
print("Negative:", neg_encode)


from keras.datasets import imdb
from keras.preprocessing import sequence

import tensorflow as tf
import tensorflow.keras as keras
import os
import numpy as np

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=VOCAB_SIZE)

train_data = sequence.pad_sequences(train_data,MAXLEN)
test_data = sequence.pad_sequences(test_data,MAXLEN)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE,32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.summary()

model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=1, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print(results)

word_index = imdb.get_word_index()

def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return sequence.pad_sequences([tokens], MAXLEN)[0]

text = "the i is film movie was just amazing, so amazing, all time blockbuster,greatest,epic, hero, 5 stars"
encoded = encode_text(text)
print(encoded)


# while were at it lets make a decode function

reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " "

    return text[:-1]
  
print(decode_integers(encoded))


def predict(text):
    encoded_text=encode_text(text)
    pred=np.zeros((1,250))
    pred[0]=encoded_text
    result=model.predict(pred)
    return (result[0])
positive_review = "this is the most beautiful movie that i have ever seen"
print(predict(positive_review))

negative_review = "this is the worst movie"
print(predict(negative_review))
