import os
from tqdm import tqdm
import numpy as np
from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

my_path=r"D:\tmp_imgs\downloaded_images"
proc_list=os.listdir(my_path)
weights=np.load("./vit-dinov2-base.npz")
vit=Dinov2Numpy(weights)
features={}
fail_list=[]
suc_sum=0
fail_sum=0
for item in tqdm(proc_list):
    if item.endswith(".jpg"):
        try:
            main_path=os.path.join(my_path,item)
            pixel_values=center_crop(main_path)
            feat=vit(pixel_values)
            features[item]=feat
            suc_sum+=1
        except Exception as e:
            fail_list.append((item,str(e)))
            tqdm.write(f"{item} processed failed. {str(e)}")
            fail_sum+=1
np.save("features.npy",features)
for x in fail_list:
    print(x)
print(f"successful items:{suc_sum}")
print(f"failed items:{fail_sum}")