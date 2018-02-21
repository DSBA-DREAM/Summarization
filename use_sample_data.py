import numpy as np
import pandas as pd
from konlpy.tag import Kkma
from gensim.models import Word2Vec

raw_data = pd.read_csv("C:/Users/Mo/Desktop/sample_data.csv", engine='python')

kkma = Kkma()
raw_data['pos_tag'] = raw_data['sentence'].apply(lambda x: [word for (word, pos) in kkma.pos(x)])

# raw_data.to_csv("C:/Users/mo/Desktop/t.csv", index=False)
num_sen = 0
for i in list(set(raw_data['doc'])):
    tmp_num_sen = sum(raw_data['doc']==i)
    if tmp_num_sen >= num_sen:
        num_sen = tmp_num_sen

num_word = 0
for i in range(len(raw_data)):
    tmp_num_word = len(raw_data['pos_tag'][i])
    if tmp_num_word >= num_word:
        num_word = tmp_num_word

data = [row.pos_tag for index, row in raw_data.iterrows()]
gensim_model = Word2Vec(data, size=100, window=5, min_count=1, sg=1)

cal_data = np.array([], dtype=np.float32).reshape(0,21,119,100)
tmp_cal_data = np.zeros((1,21,119,100))
label_data = np.array([], dtype=np.float32).reshape(0,21,2)
tmp_label_data = np.expand_dims(np.array([[0, 1]]*21, dtype=np.float32),axis=0)

count = 0
for doc_n in list(set(raw_data['doc'])): # doc(0~31)
    print("doc_n :", doc_n)
    for sen_n in range(sum(raw_data['doc'] == doc_n)): #sen
        pos = raw_data.loc[(raw_data['doc'] == doc_n) & (raw_data['sen_idx'] == sen_n)]['pos_tag'][count]
        label = raw_data.loc[(raw_data['doc'] == doc_n) & (raw_data['sen_idx'] == sen_n)]['sum_content'][count]
        if label == 1:
            tmp_label_data[0][sen_n] = np.asarray([1, 0])
        count += 1
        for word_n, word in enumerate(pos):
            tmp_cal_data[0][sen_n][word_n] = gensim_model[word]

    cal_data = np.vstack((cal_data,tmp_cal_data))
    tmp_cal_data = np.zeros((1, 21, 119, 100))
    label_data = np.vstack((label_data,tmp_label_data))
    tmp_label_data = np.expand_dims(np.array([[0, 1]] * 21, dtype=np.float32), axis=0)

np.save("C:/Users/Mo/Desktop/cal_data.npy", cal_data)
np.save("C:/Users/Mo/Desktop/label_data.npy", label_data)
