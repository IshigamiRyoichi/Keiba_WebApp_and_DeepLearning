import shutil
import random
import glob
import os
hourse_list = ["有馬記念/前蹄/上位","有馬記念/前蹄/下位"]
os.makedirs("./test", exist_ok=True)

for hourse in hourse_list:
    in_dir = "./data/"+hourse+"/*"
    in_jpg=glob.glob(in_dir)
    img_file_name_list=os.listdir("./data/"+hourse+"/")
    #img_file_name_listをシャッフル、そのうち2割をtest_imageディテクトリに入れる
    random.shuffle(in_jpg)
    os.makedirs('./test/' + hourse, exist_ok=True)
    for t in range(len(in_jpg)//5):
        shutil.move(str(in_jpg[t]), "./test/"+hourse)