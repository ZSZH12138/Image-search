import numpy as np

from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

weights = np.load("vit-dinov2-base.npz")
vit = Dinov2Numpy(weights)

cat_pixel_values = center_crop("./demo_data/cat.jpg")
cat_feat = vit(cat_pixel_values) # 获取猫的特征

dog_pixel_values = center_crop("./demo_data/dog.jpg")
dog_feat = vit(dog_pixel_values) # 获取狗的特征

ref_feat = np.load("./demo_data/cat_dog_feature.npy", allow_pickle=True)
cat_ref = ref_feat[0]
dog_ref = ref_feat[1]
# 获取标准猫狗的特征

cat_diff = np.linalg.norm(cat_feat - cat_ref)
dog_diff = np.linalg.norm(dog_feat - dog_ref)

print("Cat diff:", cat_diff)
print("Dog diff:", dog_diff)
# 差异值0.5-0.8 不影响实际使用

print(np.abs(cat_feat - cat_ref).max())  # 最大绝对误差不超过0.1