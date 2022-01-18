######################
#画像の水増し作業を行う
######################
import os
import cv2
import glob
from scipy import ndimage
import numpy as np
from IPython.display import Image, display

def mosaic(img, scale):
    h, w = img.shape[:2]  # 画像の大きさ

    # 画像を scale (0 < scale <= 1) 倍に縮小する。
    dst = cv2.resize(
        img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )

    # 元の大きさに拡大する。
    dst = cv2.resize(dst, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    return dst


hourse_list = ["有馬記念/前蹄/上位","有馬記念/前蹄/下位"]
for hourse_name in hourse_list:

    #フォルダーを作成している
    os.makedirs("./data/" + hourse_name, exist_ok=True)

    #読み取る画像のフォルダを指定
    in_dir = "./resize/" + hourse_name + "/*.jpg"
    #出力先のフォルダを指定
    out_dir = "./data/" + hourse_name

    #./assets_kora/drink/のフォルダの画像のディレクトリをすべて配列に格納している
    img_jpg = glob.glob(in_dir)

    #画像の個数分繰り返し作業を行う
    for i in range(len(img_jpg)):
        img =cv2.imread(str(img_jpg[i]))
        #64×64サイズにしてる
        img_rot = cv2.resize(img,(64,64))
        #パスを結合している
        fileName = os.path.join(out_dir, str(i) + ".jpg")
        #画像を出力
        cv2.imwrite(str(fileName),img_rot)
        #cv2.imwrite('data/dst/lena_cv_flip_lr.jpg', img_flip_lr)
        #--------
        #閾値処理
        #--------
        #閾値を変更している
        #閾値を決め、値化の方法(今回はTHRESH_TOZERO)を決めている
        for j in [80,90,100,110]:
            img_thr = cv2.threshold(img_rot, j, 255, cv2.THRESH_TOZERO)[1]
            #パスを結合
            fileName = os.path.join(out_dir, str(i)+ "_" + str(j) +"thr.jpg")
            #画像を出力
            cv2.imwrite(str(fileName),img_thr)
        #----------
        #ぼかし処理
        #----------
        #カーネルサイズ(5×5)とガウス関数を指定する
        #カーネルサイズはモザイクの粗さ的なもの
        #ガウス関数はよくわからない
        for j in [1,3,5,7,9]:
            img_filter = cv2.GaussianBlur(img_rot, (j, j), 0)
            #パスを結合
            fileName = os.path.join(out_dir, str(i) + "_" + str(j) +"filter.jpg")
            #画像を出力
            cv2.imwrite(str(fileName), img_filter)

        #--------
        #ノイズ除去
        #--------
        for j in [3,5,7]:
            img_median = cv2.medianBlur(img_rot,j)
            #パスを結合
            fileName = os.path.join(out_dir, str(i) + "_" + str(j) +"median.jpg")
            #画像を出力
            cv2.imwrite(str(fileName), img_median)
        
        #------------
        #コントラスト
        #------------
        for j in [0.2,0.6,1.4,1.8]:
            img_res = cv2.convertScaleAbs(img_rot,alpha = j,beta = 0.0)
            #パスを結合
            fileName = os.path.join(out_dir, str(i) + "_" + str(j) +"res.jpg")
            #画像を出力
            cv2.imwrite(str(fileName), img_res)

        #------------
        #モザイク
        #------------
        for j in [0.1,0.2,0.3,0.4]:
            # モザイク処理を行う。
            img_dst = mosaic(img,j)
            #パスを結合
            fileName = os.path.join(out_dir, str(i) + "_" + str(j) +"dst.jpg")
            #画像を出力
            cv2.imwrite(str(fileName), img_res)