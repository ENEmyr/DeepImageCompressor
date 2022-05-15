import cv2
import numpy as np
from typing import Union, List, Tuple 
from os import path, listdir

def check_file_res(fpath: str) -> Tuple[int, int, int]:
    ''' return Tuple(height, width, channel) '''
    if path.exists(fpath):
        img = cv2.imread(fpath)
        return img.shape
    else:
        raise Exception("File not found! : {}".format(fpath))

def file_res_summary(dirpath: str) -> List[Tuple[int, int, int]]:
    summary_dict = {}
    count = 1
    if path.exists(path.abspath(dirpath)):
        for f in listdir(path.abspath(dirpath)):
            shape = check_file_res(f'{path.abspath(dirpath)}/{f}')
            if shape not in summary_dict.values():
                summary_dict[count] = shape
                count += 1
    return list(summary_dict.values())

def find_min_max_res(
        shapes: list, 
        include_channel: bool = False
        ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    ''' return Tuple[Tuple[min_res_h, min_res_w, min_res_c], Tuple[max_res_h, max_res_w, max_res_c]] '''
    min_res, max_res = shapes[0], shapes[0]
    for shape in shapes:
        if include_channel:
            min_res = shape if shape[0]*shape[1]*shape[2] < min_res[0]*min_res[1]*min_res[2] else min_res
            max_res = shape if shape[0]*shape[1]*shape[2] > max_res[0]*max_res[1]*max_res[2] else max_res
        else:
            min_res = shape if shape[0]*shape[1] < min_res[0]*min_res[1] else min_res
            max_res = shape if shape[0]*shape[1] > max_res[0]*max_res[1] else max_res
    return (min_res, max_res)

def find_min_max_hw(shapes: list) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    ''' return Tuple[Tuple[min_h, min_w], Tuple[max_h, max_w]] '''
    min_h, min_w, max_h, max_w = np.inf, np.inf, 0, 0
    for shape in shapes:
        min_h = shape[0] if min_h > shape[0] else min_h
        min_w = shape[1] if min_w > shape[1] else min_w
        max_h = shape[0] if max_h < shape[0] else max_h
        max_w = shape[1] if max_w < shape[1] else max_w
    return ((min_h, min_w), (max_h, max_w))

def resize(img_arr:np.ndarray, target_res:Tuple[int,int]) -> np.ndarray:
    ''' return resized ndarray '''
    bigger = img_arr.shape[0] if img_arr.shape[0] > img_arr.shape[1] else img_arr.shape[1]
    ratio = (target_res[0] if target_res[0] > target_res[1] else target_res[1])/bigger
    new_w = int(img_arr.shape[1]*ratio)
    new_h = int(img_arr.shape[0]*ratio)
    #if new_w > new_h:
    #    new_w = new_w if new_w == target_res[1] else target_res[1]
    #else:
    #    new_h = new_h if new_h == target_res[0] else target_res[0]
    img_arr = cv2.resize(img_arr, (new_w, new_h))
    return img_arr

def padd_border(img_arr:np.ndarray, target_res:Tuple[int,int]) -> Union[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    ''' return padded ndarray of image and the origin/end of padding coordinates '''
    if img_arr.shape[0] > target_res[0] or img_arr.shape[1] > target_res[1]:
        img_arr = resize(img_arr, target_res)
    h_border = target_res[0]-img_arr.shape[0]
    w_border = target_res[1]-img_arr.shape[1]
    padd = np.pad(img_arr, pad_width = [(h_border//2, h_border-h_border//2), (w_border//2, w_border-w_border//2), (0, 0)], mode='edge')
    # padd, (start_x, start_y), (end_x, end_y)
    return padd, (w_border//2, h_border//2), (img_arr.shape[1]+h_border//2, img_arr.shape[0]+w_border//2)

def add_noise(
        img_arr:np.ndarray,
        sigma=.15,
        mu=0
        ) -> np.ndarray:
    ''' return noisy version of input image'''
    sd = sigma # np.random.exponential(.15)
    noise_layer = np.random.normal(loc=mu, scale=sd, size=img_arr.shape)
    # noise_layer = np.random.normal(loc=loc, scale=scale, size=img_arr.shape)
    appied_noise = np.clip(img_arr+noise_layer, 0, 1)
    return appied_noise

def random_crop(img:np.ndarray, random_crop_size:Tuple[int, int]) -> "np.ndarray":
    height, width = img.shape[:-1]
    dx, dy = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:y+dy, x:x+dx]

def generator(batches, noise_sd=.15, random_crop_size=None):
    for batch_x, batch_y in batches:
        preprocessed_batch_x = batch_x
        if random_crop_size != None:
            preprocessed_batch_x = random_crop(preprocessed_batch_x, random_crop_size)
        if noise_sd != 0:
            preprocessed_batch_x = add_noise(preprocessed_batch_x, sigma=noise_sd)
        yield(preprocessed_batch_x, batch_y)
