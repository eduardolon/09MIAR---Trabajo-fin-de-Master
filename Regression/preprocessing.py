import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops
from skimage.measure import label
from skimage.feature import graycomatrix, graycoprops
from scipy import stats

def extract_attributes(image):
    
    # Resize once and convert to grayscale
    image_resized = cv2.resize(image, (500, 300))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

    # Thresholding and hole filling
    _, img = cv2.threshold(image_gray, 0, 1, cv2.THRESH_OTSU)
    img = 1 - img
    img = binary_fill_holes(img)

    # Calculate largest bounding box using regionprops
    lab, num = label(img, return_num=True)
    max_area = 0
    bbox = []

    for i in range(1, num + 1):
        object_region = (lab == i).astype('uint8')
        prop = regionprops(object_region)[0]
        area = prop.area
        if area > max_area:
            max_area = area
            bbox = prop.bbox

            
    # If max_area is too small, skip processing (early exit)
    if max_area < 1000:
        print(f"Max area too small: {max_area}")
        return None

    # Crop the image and apply the mask
    img_cropped = image_resized[bbox[0]: bbox[2], bbox[1]: bbox[3]]
    mask_cropped = img[bbox[0]: bbox[2], bbox[1]:bbox[3]]
    img_cropped = img_cropped * mask_cropped[..., None]

    # Shape features
    prop = regionprops(mask_cropped.astype('uint8'))[0]
    area = prop.area
    ex = prop.eccentricity
    extent = prop.extent

    shape_features = [area, ex, extent]  


    # Texture features (GLCM)
    glcm = graycomatrix(image_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, "correlation")[0, 0]
    glcm_dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
    glcm_homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    glcm_asm = graycoprops(glcm, "ASM")[0, 0]
    glcm_energy = graycoprops(glcm, "energy")[0, 0]
    glcm_correlation = graycoprops(glcm, "correlation")[0, 0]

    texture_features = [glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_asm, glcm_energy, glcm_correlation]
    
    # Color features
    # RGB histogram calculation

    img_r = img_cropped[:,:,0]
    img_g = img_cropped[:,:,1]
    img_b = img_cropped[:,:,2]

    mask_r = img_r > 0
    mask_g = img_g > 0
    mask_b = img_b > 0

    hist_r = cv2.calcHist([img_r], [0], mask_r.astype("uint8"), [25], [0, 250]).flatten()
    hist_g = cv2.calcHist([img_g], [0], mask_g.astype("uint8"), [25], [0, 250]).flatten()
    hist_b = cv2.calcHist([img_b], [0], mask_b.astype("uint8"), [25], [0, 250]).flatten()
    
    # HSV histogram calculation
    img_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)

    img_v = img_cropped[:,:,2]

    mask_v = img_v > 0

    hist_h = cv2.calcHist([img_hsv], [0], None, [25], [0, 256]).flatten()
    hist_s = cv2.calcHist([img_hsv], [1], None, [25], [0, 256]).flatten()
    hist_v = cv2.calcHist([img_v], [0], mask_v.astype("uint8"), [25], [0, 250]).flatten()

    color_features = [hist_r, hist_g, hist_b, hist_h, hist_s, hist_v]

    attr = shape_features + texture_features + color_features

    return [item for sublist in attr for item in (sublist if isinstance(sublist, np.ndarray) else [sublist])]

def get_all_columns():

    return ['area', 'eccentricity', 'extent', 'GLCM Contrast', 'GLCM dissimilarity', 'GLCM homogeneity', 'GLCM ASM', 'GLCM energy', 'GLCM correlation', 'hist_r1', 'hist_r2', 'hist_r3', 'hist_r4', 'hist_r5', 'hist_r6', 'hist_r7', 'hist_r8', 'hist_r9', 'hist_r10', 'hist_r11', 'hist_r12', 'hist_r13', 'hist_r14', 'hist_r15', 'hist_r16', 'hist_r17', 'hist_r18', 'hist_r19', 'hist_r20', 'hist_r21', 'hist_r22', 'hist_r23', 'hist_r24', 'hist_r25', 'hist_g1', 'hist_g2', 'hist_g3', 'hist_g4', 'hist_g5', 'hist_g6', 'hist_g7', 'hist_g8', 'hist_g9', 'hist_g10', 'hist_g11', 'hist_g12', 'hist_g13', 'hist_g14', 'hist_g15', 'hist_g16', 'hist_g17', 'hist_g18', 'hist_g19', 'hist_g20', 'hist_g21', 'hist_g22', 'hist_g23', 'hist_g24', 'hist_g25', 'hist_b1', 'hist_b2', 'hist_b3', 'hist_b4', 'hist_b5', 'hist_b6', 'hist_b7', 'hist_b8', 'hist_b9', 'hist_b10', 'hist_b11', 'hist_b12', 'hist_b13', 'hist_b14', 'hist_b15', 'hist_b16', 'hist_b17', 'hist_b18', 'hist_b19', 'hist_b20', 'hist_b21', 'hist_b22', 'hist_b23', 'hist_b24', 'hist_b25', 'hist_h1', 'hist_h2', 'hist_h3', 'hist_h4', 'hist_h5', 'hist_h6', 'hist_h7', 'hist_h8', 'hist_h9', 'hist_h10', 'hist_h11', 'hist_h12', 'hist_h13', 'hist_h14', 'hist_h15', 'hist_h16', 'hist_h17', 'hist_h18', 'hist_h19', 'hist_h20', 'hist_h21', 'hist_h22', 'hist_h23', 'hist_h24', 'hist_h25', 'hist_s1', 'hist_s2', 'hist_s3', 'hist_s4', 'hist_s5', 'hist_s6', 'hist_s7', 'hist_s8', 'hist_s9', 'hist_s10', 'hist_s11', 'hist_s12', 'hist_s13', 'hist_s14', 'hist_s15', 'hist_s16', 'hist_s17', 'hist_s18', 'hist_s19', 'hist_s20', 'hist_s21', 'hist_s22', 'hist_s23', 'hist_s24', 'hist_s25', 'hist_v1', 'hist_v2', 'hist_v3', 'hist_v4', 'hist_v5', 'hist_v6', 'hist_v7', 'hist_v8', 'hist_v9', 'hist_v10', 'hist_v11', 'hist_v12', 'hist_v13', 'hist_v14', 'hist_v15', 'hist_v16', 'hist_v17', 'hist_v18', 'hist_v19', 'hist_v20', 'hist_v21', 'hist_v22', 'hist_v23', 'hist_v24', 'hist_v25']

def get_columns():
    return ['area', 'eccentricity', 'extent', 'GLCM Contrast', 'GLCM dissimilarity',
       'GLCM homogeneity', 'GLCM ASM', 'GLCM energy', 'GLCM correlation',
       'hist_r5', 'hist_r6', 'hist_r7', 'hist_r8', 'hist_r9', 'hist_r10',
       'hist_r11', 'hist_r12', 'hist_r13', 'hist_r14', 'hist_r15', 'hist_r16',
       'hist_r17', 'hist_r18', 'hist_r19', 'hist_r20', 'hist_g4', 'hist_g5',
       'hist_g6', 'hist_g7', 'hist_g8', 'hist_g9', 'hist_g10', 'hist_g11',
       'hist_g12', 'hist_g13', 'hist_g14', 'hist_g15', 'hist_g16', 'hist_g17',
       'hist_g18', 'hist_b5', 'hist_b6', 'hist_b7', 'hist_b8', 'hist_b9',
       'hist_b10', 'hist_b11', 'hist_b12', 'hist_b13', 'hist_b14', 'hist_b15',
       'hist_b16', 'hist_b17', 'hist_h1', 'hist_h2', 'hist_h3', 'hist_h12',
       'hist_h13', 'hist_h14', 'hist_h15', 'hist_h16', 'hist_h17', 'hist_h18',
       'hist_s1', 'hist_s2', 'hist_s3', 'hist_s4', 'hist_s5', 'hist_s6',
       'hist_s7', 'hist_s8', 'hist_s9', 'hist_s10', 'hist_s11', 'hist_s12',
       'hist_s13', 'hist_s14', 'hist_v5', 'hist_v6', 'hist_v7', 'hist_v8',
       'hist_v9', 'hist_v10', 'hist_v11', 'hist_v12', 'hist_v13', 'hist_v14',
       'hist_v15', 'hist_v16', 'hist_v17', 'hist_v18']