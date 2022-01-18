from flask import Flask, request, session, g, redirect, url_for, abort, render_template
import tensorflow as tf
import os
import numpy as np
from keras.models import  load_model
import cv2
from tensorflow.nn import leaky_relu
from tensorflow.keras import layers, Model, Input
from PIL import Image
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from itertools import combinations

# RankNet構築
class RankNet(Model):
    def __init__(self):
        super().__init__()
        self.dense = [layers.Dense(16, activation=leaky_relu), layers.Dense(8, activation=leaky_relu)]
        self.o = layers.Dense(1, activation='linear')
        self.oi_minus_oj = layers.Subtract()
    
    def call(self, inputs):
        xi, xj = inputs
        densei = self.dense[0](xi)
        densej = self.dense[0](xj)
        for dense in self.dense[1:]:
            densei = dense(densei)
            densej = dense(densej)
        oi = self.o(densei)
        oj= self.o(densej)
        oij = self.oi_minus_oj([oi, oj])
        output = layers.Activation('sigmoid')(oij)
        return output
    
    def build_graph(self):
        x = [Input(shape=(10)), Input(shape=(10))]
        return Model(inputs=x, outputs=self.call(x))


# ranknet
ranknet = RankNet()
years = [2020,2019,2018,2017,2016,2015,2014,2013,2012,2011]
df = pd.read_csv("./documents/有馬記念Data.csv")
df["タイム指数2-3"] = df["タイム指数2"] - df["タイム指数3"]

index_num = 0
xi = []
xj = []
pij = []
pair_ids = []
pair_query_id = []

for year in years:
    one_year_Data = df[df['年数'] == year]

    index_list = [i for i in range(len(one_year_Data))]
    random.shuffle(index_list)
    for pair_id in combinations(index_list, 2):
        pair_query_id.append(year)
        pair_ids.append(pair_id)
        i = pair_id[0]
        j = pair_id[1]
        xi.append([one_year_Data.at[i+index_num,"タイム指数2"],one_year_Data.at[i+index_num,"タイム指数2-3"],one_year_Data.at[i+index_num,"上り"]])
        xj.append([one_year_Data.at[j+index_num,"タイム指数2"],one_year_Data.at[j+index_num,"タイム指数2-3"],one_year_Data.at[j+index_num,"上り"]])

        if one_year_Data.at[i+index_num,"順位"]  == one_year_Data.at[j+index_num,"順位"] :
            pij_com = 0.5

        elif one_year_Data.at[i+index_num,"順位"]  > one_year_Data.at[j+index_num,"順位"] :
            pij_com = 0

        else:
            pij_com = 1

        pij.append(pij_com)
    index_num += len(one_year_Data)
    index_list.clear()

xi = np.array(xi)
xj = np.array(xj)
pij = np.array(pij)
pair_query_id = np.array(pair_query_id)

xi_train, xi_test, xj_train, xj_test, pij_train, pij_test, pair_id_train, pair_id_test = train_test_split(
    xi, xj, pij, pair_ids, test_size=0.2, stratify=pair_query_id)

ranknet.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
# ranknet.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])
ranknet.fit([xi_train, xj_train], pij_train, epochs=85, batch_size=4, validation_data=([xi_test, xj_test], pij_test))

# resnetの読み込み
resnet =load_model("./documents/hourse_resnet.h5")

# 予想順に並べる
def bubble_sort_hourse(data_arr, hourse_list):
    for i in range(len(data_arr)):
        for j in range(len(data_arr)-i-1):
            if ranknet.predict([np.array([data_arr[j]]), np.array([data_arr[j+1]])]) < 0.5:
                data_arr[j], data_arr[j + 1] = data_arr[j + 1], data_arr[j]
                hourse_list[j],hourse_list[j + 1] = hourse_list[j + 1], hourse_list[j]

# 画像の予測
def image_predict(hourse_list):
    img_index = hourse_list[0]
    hourse_max = 0
    for i in range(5):
        img = cv2.imread('./uploads/h'+str(hourse_list[i])+'_img_file.jpg')
        img_rot = cv2.resize(img,(64,64))
        #色成分を分割
        b,g,r = cv2.split(img_rot)
        #色成分を結合
        img_cha = cv2.merge([r,g,b])
        img_exp = np.expand_dims(img_cha,axis=0)
        y_p = resnet.predict(img_exp)[0][0]
        if y_p > hourse_max:
            hourse_max = y_p
            img_index = hourse_list[i]
    return img_index

# 画像をアップロードするフォルダー
upload_folder = './uploads'
# 画像の拡張子の制限
# set()で重複した要素を取り除く
allowed_extenstions = set(["png","jpg","jpeg"])

# お決まり
app = Flask(__name__)
app.secret_key = "hogehoge"

# 設定の保存
# upload_folderの設定を保存
UPLOAD_FOLDER = './uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# configの読み込み
app.config.from_object(__name__)

# main画面
@app.route("/main_page",methods = ["GET", "POST"])
def main_page():
    text = "データ入力ページ"
    return render_template("main_page.html",text = text)

# アップロードしたファイルの処理
@app.route('/result_page',methods = ["GET", "POST"])
# ファイルを表示する
def result_page():
    text = "予想結果"
    #データが届いたら
    if request.method == "POST":
        # 値のデータを取得
        hourse_num = request.form['hourse_num']
        data_arr = [[int(request.form['h'+str(i+1)+'_time_num1']),int(request.form['h'+str(i+1)+'_time_num1'])-int(request.form['h'+str(i+1)+'_time_num2']),float(request.form['h'+str(i+1)+'_agari'])] for i in range(int(hourse_num))]
        hourse_list = [i+1 for i in range(int(hourse_num))]
        bubble_sort_hourse(data_arr, hourse_list)

        # ファイルを読み込む
        for i in range(5):
            img_file = request.files['h'+str(hourse_list[i])+'_img_file']
            filename = 'h'+str(hourse_list[i])+'_img_file.jpg'
            # 画像のアップロード先URLを生成する
            img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
            # 画像をアップロード先に保存する
            img_file.save(img_url)

        # 画像解析
        index = image_predict(hourse_list)

        return render_template('result_page.html',text = text , hourse_list=hourse_list, index=index)

## おまじない
if __name__ == "__main__":
    app.run(debug=True)