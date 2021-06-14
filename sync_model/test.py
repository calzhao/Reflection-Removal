import os
import numpy as np
from PIL import Image
from os.path import join
import sync_models
import torchvision.transforms as transforms
import cv2

to_tensor = transforms.ToTensor()
def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor.float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy

B_path="b.jpg"
R_path="r.jpg"

t_img = Image.open(B_path).convert('RGB')
r_img = Image.open(R_path).convert('RGB')
print(np.array(t_img).shape)

BaseModel=sync_models.BaseSythesis_1(coef=0.8)
SynModel1=sync_models.ReflectionSythesis_1(kernel_sizes=[11], low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3)
SynModel2=sync_models.ReflectionSythesis_2()
SynModel3=sync_models.ReflectionSythesis_3()

b0,r0,syn0=BaseModel(t_img,r_img)
print(np.max(syn0),np.min(syn0))
b1,r1,syn1=SynModel1(t_img,r_img)
b2,r2,syn2=SynModel2(t_img,r_img)
b3,r3,syn3=SynModel3(t_img,r_img)

# b,r=cv2.imread("b.jpg"),cv2.imread("r.jpg")
# b3,r3,syn3=SynModel3(b,r)

a0=tensor2im(to_tensor(syn0)).astype(np.uint8)
Image.fromarray(a0).save('syn0.png')
a1=tensor2im(to_tensor(syn1)).astype(np.uint8)
Image.fromarray(a1).save('syn1.png')
a2=tensor2im(to_tensor(syn2)).astype(np.uint8)
Image.fromarray(a2).save('syn2.png')
a3=tensor2im(to_tensor(syn3)).astype(np.uint8)
Image.fromarray(a3).save('syn3.png')