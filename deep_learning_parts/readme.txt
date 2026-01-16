1. How to debug
- Complete dinov2_numpy.py
- Run debug.py
- Check output, i.e., compare your extracted features with the reference (./demo_data/cat_dog_feature.npy). Make sure the difference is within a small numerical tolerance.

2. Image retrieval
- Cownload 10,000+ web images (data.csv) to build the gallery set
- Finish 'resize_short_side' in preprocess_image.py. The function must correctly resize images of different resolutions such that the shorter side becomes the target size (e.g., 224). Meanwhile, both sides should be the multiple of 14
- Extract features for all gallery images via your ViT (dinov2_numpy.py)
- When user upload an image, preprocess → extract features, compute similarity with all gallery features (e.g., cosine similarity or L2 distance), and return the Top-10 most similar images as search results
----------------------------------------------------------------------------------------------------------------

运行download.py会将本文件夹中的csv文件中的链接全部都下载下来，为了节省空间，这里已经把csv文件删去，如果想要复现，可以自己添加一个csv文件。

get_feature.py的my_path变量存储的便是你的图片文件夹路径，运行该文件可以获取所有图片的特征并保存为.npy文件。注意一定要修改my_path路径才可以。

注意：由于文件vit-dinov2-base.npz太大，无法上传至GitHub，如果想要复现实验过程，请将vit-dinov2-base.npz文件添加至本文件夹。
