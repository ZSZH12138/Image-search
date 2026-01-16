import csv
import os
import requests
from tqdm import tqdm

CSV_PATH=r"C:\Users\LENOVO\Desktop\project5-SZU-python\assignments\data.csv"

SAVE_DIR=r"D:\tmp_imgs"
os.makedirs(SAVE_DIR,exist_ok=True)
fail_list=[]

with open(CSV_PATH,newline='',encoding="utf-8") as f:
    reader=csv.DictReader(f)
    proc=[]
    for idx,row in enumerate(reader):
        proc.append((idx,row))
    for idx,row in tqdm(proc):
        url=row["image_url"]
        try:
            resp=requests.get(url,timeout=10)
            resp.raise_for_status()
            filename=f"{idx}.jpg"
            save_path=os.path.join(SAVE_DIR,filename)
            with open(save_path,"wb") as img_f:
                img_f.write(resp.content)
        except Exception as e:
            fail_list.append((idx,url,str(e)))

for fail in fail_list:
    print(fail)