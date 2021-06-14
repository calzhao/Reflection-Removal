from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import scipy.stats as st
import cv2
import numbers
import types
import collections
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from scipy.signal import convolve2d

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def quantization(img): # in (0,1) out(0,255)
    img = np.clip(img, 0, 1)
    img_oct = img * 255
    img_oct = img_oct.astype(np.uint8)
    return img_oct

def photometric_pano(img):
    #br, gr, rr = cv2.split(img)
    br, gr, rr = img[:,:,2],img[:,:,1],img[:,:,0]
    r=img
    print(np.array(br).shape)
    p1_b = np.poly1d([20.28, -48.43, 41.34, -14.53, 2.36, -0.04053])
    p1_g = np.poly1d([15.28, -35.39, 29.76, -10.42, 1.778, -0.01514])
    p1_r = np.poly1d([11.17, -23.39, 17.8, -5.499, 0.9215, -0.004502])

    br = quantization(p1_b(br / 255.0))
    gr = quantization(p1_g(gr / 255.0))
    rr = quantization(p1_r(rr / 255.0))

    r[:,:,2],r[:,:,1],r[:,:,0] = br, gr, rr
    return r

class BaseSythesis_1(object):
    """
        I=aB+(1-a)*R
    """
    def __init__(self, coef=0.8):
        self.coef=coef

    def __call__(self, B, R):
        if not _is_pil_image(B):
            raise TypeError('B should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))
        
        B_ = np.asarray(B, np.float32) / 255.
        R_ = np.asarray(R, np.float32) / 255.

        B = B_ * self.coef
        print(np.max(B_),np.min(B_))
        R = R_ * (1-self.coef)
        print(np.max(R_),np.min(R_))
        M = B + R
        
        return B, R, M

class ReflectionSythesis_1(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of ICCV 2017 paper *"A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing"*    
    """
    def __init__(self, kernel_sizes=None, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
        self.kernel_sizes = kernel_sizes or [11]
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.low_gamma = low_gamma
        self.high_gamma = high_gamma
        print('[i] reflection sythesis model: {}'.format({
            'kernel_sizes': kernel_sizes, 'low_sigma': low_sigma, 'high_sigma': high_sigma,
            'low_gamma': low_gamma, 'high_gamma': high_gamma}))

    def __call__(self, B, R):
        if not _is_pil_image(B):
            raise TypeError('B should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))
        
        B_ = np.asarray(B, np.float32) / 255.
        R_ = np.asarray(R, np.float32) / 255.

        kernel_size = np.random.choice(self.kernel_sizes)
        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        gamma = np.random.uniform(self.low_gamma, self.high_gamma)
        R_blur = R_
        kernel = cv2.getGaussianKernel(11, sigma)
        kernel2d = np.dot(kernel, kernel.T)

        for i in range(3):
            R_blur[...,i] = convolve2d(R_blur[...,i], kernel2d, mode='same')

        M_ = B_ + R_blur
        
        if np.max(M_) > 1:
            m = M_[M_ > 1]
            m = (np.mean(m) - 1) * gamma
            R_blur = np.clip(R_blur - m, 0, 1)
            M_ = np.clip(R_blur + B_, 0, 1)
        
        return B_, R_blur, M_

class ReflectionSythesis_2(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of CVPR 2018 paper *"Single Image Reflection Separation with Perceptual Losses"*
    """
    def __init__(self, kernel_sizes=None):
        self.kernel_sizes = kernel_sizes or np.linspace(1,5,80)
    
    @staticmethod
    def gkern(kernlen=100, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        kernel = kernel/kernel.max()
        return kernel

    def __call__(self, t, r):        
        t = np.float32(t) / 255.
        r = np.float32(r) / 255.
        ori_t = t
        # create a vignetting mask
        g_mask=self.gkern(560,3)
        g_mask=np.dstack((g_mask,g_mask,g_mask))
        sigma=self.kernel_sizes[np.random.randint(0, len(self.kernel_sizes))]

        t=np.power(t,2.2)
        r=np.power(r,2.2)
        
        sz=int(2*np.ceil(2*sigma)+1)
        
        r_blur=cv2.GaussianBlur(r,(sz,sz),sigma,sigma,0)
        blend=r_blur+t
        
        att=1.08+np.random.random()/10.0
        
        for i in range(3):
            maski=blend[:,:,i]>1
            mean_i=max(1.,np.sum(blend[:,:,i]*maski)/(maski.sum()+1e-6))
            r_blur[:,:,i]=r_blur[:,:,i]-(mean_i-1)*att
        r_blur[r_blur>=1]=1
        r_blur[r_blur<=0]=0

        h,w=r_blur.shape[0:2]
        neww=np.random.randint(0, 560-w-10)
        newh=np.random.randint(0, 560-h-10)
        alpha1=g_mask[newh:newh+h,neww:neww+w,:]
        alpha2 = 1-np.random.random()/5.0
        r_blur_mask=np.multiply(r_blur,alpha1)
        blend=r_blur_mask+t*alpha2
        
        t=np.power(t,1/2.2)
        r_blur_mask=np.power(r_blur_mask,1/2.2)
        blend=np.power(blend,1/2.2)
        blend[blend>=1]=1
        blend[blend<=0]=0
        
        return np.float32(ori_t), np.float32(r_blur_mask), np.float32(blend)

class ReflectionSythesis_3(object):
    """
        I=aB+(1-a)*R
    """
    def __init__(self, low_gamma=1.3, high_gamma=1.3):
        self.low_gamma,self.high_gamma=low_gamma,high_gamma

    def __call__(self, B, R):

        B_b = np.asarray(B, np.float32) 
        R_b = np.asarray(R, np.float32)       
        
        B_ = B_b / 255.
        R_blur = photometric_pano(R_b) / 255.

        alpha2 = (np.random.random()+1)/2.0
        R_blur = R_blur*alpha2
        M_ = B_ + R_blur
        gamma = np.random.uniform(self.low_gamma, self.high_gamma)

        if np.max(M_) > 1:
            m = M_[M_ > 1]
            m = (np.mean(m) - 1) * gamma
            R_blur = np.clip(R_blur - m, 0, 1)
            M_ = np.clip(R_blur + B_, 0, 1)
        
        return B_, R_blur, M_