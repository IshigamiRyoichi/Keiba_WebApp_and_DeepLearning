import cv2
import glob
import os

hourse_list = ["有馬記念/前蹄/上位","有馬記念/前蹄/下位"]
for hourse_name in hourse_list:
    in_dir = "./images/" + hourse_name + "/*.jpg"
    #フォルダーを作成している
    os.makedirs("./gray/" + hourse_name, exist_ok=True)
    out_dir = "./gray/" + hourse_name
    img_list = glob.glob(in_dir)
    for i in range(len(img_list)):
        img_bgr = cv2.imread(img_list[i])
        # グレースケースに変える
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        #パスを結合している
        fileName = os.path.join(out_dir, "img_" + str(i) + ".jpg")
        i += 1
        cv2.imwrite(str(fileName), img_gray)
