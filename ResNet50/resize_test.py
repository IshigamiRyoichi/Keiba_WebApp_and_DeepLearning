import os
import cv2
import glob

hourse_list = ["有馬記念/前蹄/上位","有馬記念/前蹄/下位"]
for hourse_name in hourse_list:

    #フォルダーを作成している
    os.makedirs("./resize/" + hourse_name, exist_ok=True)

    #読み取る画像のフォルダを指定
    in_dir = "./gray/" + hourse_name + "/*.jpg"
    #出力先のフォルダを指定
    out_dir = "./resize/" + hourse_name

    #./assets_kora/drink/のフォルダの画像のディレクトリをすべて配列に格納している
    img_jpg = glob.glob(in_dir)
    #./assets_kora/drink/のファイルを一覧にする

    #画像の個数分繰り返し作業を行う
    for i in range(len(img_jpg)):
        img =cv2.imread(str(img_jpg[i]))
        #64×64サイズにしてる
        img_rot = cv2.resize(img,(64,64))
        #パスを結合している
        fileName = os.path.join(out_dir, "img_" + str(i) + ".jpg")
        #画像を出力
        cv2.imwrite(str(fileName),img_rot)