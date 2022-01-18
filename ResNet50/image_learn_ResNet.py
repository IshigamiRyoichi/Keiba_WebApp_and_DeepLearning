###############
#画像の機械学習
###############
import os
import cv2
import numpy as np
from keras.layers import Dense, Dropout, Input,Activation, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator

hourse_list = ["有馬記念/前蹄/上位","有馬記念/前蹄/下位"]

# 教師データのラベル付け
X_train = [] 
Y_train = []
i = 0

for name in hourse_list:
    #ファルダーの中身の画像を一覧にする
    img_file_name_list=os.listdir("./data/"+name)
    #確認
    print(len(img_file_name_list))
    #画像ファイルごとに処理
    for img_file_name in img_file_name_list:
        #パスを結合
        n=os.path.join("./data/"+name+"/"+img_file_name)
        img = cv2.imread(n)
        #色成分を分割
        b,g,r = cv2.split(img)
        #色成分を結合
        img = cv2.merge([r,g,b])
        X_train.append(img)
        Y_train.append(i)
    i += 1

# テストデータのラベル付け
X_test = [] # 画像データ読み込み
Y_test = [] # ラベル（名前）
#飲み物の名前ごとに処理する
i = 0
for name in hourse_list:
    img_file_name_list=os.listdir("./test/"+name)
    #確認
    print(len(img_file_name_list))
    #ファイルごとに処理
    for img_file_name in img_file_name_list:
        n=os.path.join("./test/" + name + "/" + img_file_name)
        img = cv2.imread(n)
        #色成分を分割
        b,g,r = cv2.split(img)
        #色成分を結合
        img = cv2.merge([r,g,b])
        X_test.append(img)
        # ラベルは整数値
        Y_test.append(i)
    i += 1
#配列化
X_train=np.array(X_train)
X_test=np.array(X_test)

#ラベルをone-hotベクトルにする？
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

# trainデータとtestデータを整える
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
 horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train,batch_size=32)
val_generator = val_datagen.flow(X_test, y_test, batch_size=32)

# ResnNetの準備
input_tensor = Input(shape=(64, 64, 3))
resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層の準備
fc_model = Sequential()
fc_model.add(Flatten(input_shape=resnet50.output_shape[1:]))
fc_model.add(Dense(512, activation='relu'))
fc_model.add(Dropout(0.3))
fc_model.add(Dense(2, activation='sigmoid'))

# modelの準備
resnet50_model = Model(resnet50.input, fc_model(resnet50.output))
# #ResNet50の一部の重みを固定
for layer in resnet50_model.layers[:100]:
    layer.trainable = False
resnet50_model.summary()

# コンパイル
resnet50_model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 学習
history = resnet50_model.fit(train_generator, batch_size=4, 
                    epochs=50, verbose=1, validation_data=val_generator)#validation_data=(X_test, y_test)

# 汎化制度の評価・表示
score = resnet50_model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#モデルを保存
resnet50_model.save("./hourse_resnet.tf")
resnet50_model.save("./hourse_resnet.h5")
