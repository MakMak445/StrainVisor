import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def reloadim(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def showim(img):
    plt.imshow(img)
    plt.title('Cat Image')
    plt.axis('off')
    plt.show()

def turn_greyscale(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    '''fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[1].imshow(gray, cmap='gray')
    axs[1].set_title('Grayscale Image')
    axs[0].axis('off')
    axs[1].axis('off')
    plt.show()'''
    return gray

def mask_image(img):
    mask=np.zeros(img.shape[:2], dtype=np.uint8)
    height, width=img.shape[:2]
    centre=(width//4, height//4)
    radius=min(centre[0], centre[1])
    cv.circle(mask, centre, radius, (255, 255, 255), -1)
    masked_image=cv.bitwise_and(img, img, mask=mask)
    masked_image=cv.cvtColor(masked_image, cv.COLOR_BGR2RGB)
    cv.imshow('Masked Image', masked_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def split_image(img):
    r, g, b=cv.split(img)
    fig, axs=plt.subplots(1, 3, figsize=(10,10))
    axs[0].imshow(b, cmap='Blues')
    axs[0].axis('off')
    axs[0].set_title('Blue')
    axs[1].imshow(g, cmap='Greens')
    axs[1].axis('off')
    axs[2].imshow(r, cmap='Reds')
    axs[2].axis('off')
    plt.show()

def merge_image(img):
    r, g, b=cv.split(img)
    merged_im=cv.merge([g,b,r])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(merged_im)
    ax1.set_title('Merged')
    ax2.imshow(img)
    ax2.set_title('Original')
    ax1.axis('off')
    ax2.axis('off')
    plt.show()

def grey_hist(img):
    grey_img=cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    hist = cv.calcHist([grey_img], [0], None, [256], [0, 256])
    hist = hist.squeeze()
    plt.plot(hist)
    plt.title('Greyscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.show()

def colour_hist(img):
    colour=('r', 'g', 'b')
    for i, col in enumerate(colour):
        hist=cv.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=col)
    plt.show()

def equal_hist(img):
    gray = turn_greyscale(img)
    eqd_gray=cv.equalizeHist(gray)
    fig, axis=plt.subplots(2, 2, figsize=(10,10))
    axis[0,0].imshow(gray, cmap='gray')
    axis[0,0].set_title('Gray')
    axis[0,1].imshow(eqd_gray,cmap='gray')
    axis[0,1].set_title('Equalised')
    axis[1,0].hist(gray.ravel(), 256, [0,256])
    axis[1,0].set_title('Histogram')
    axis[1,1].hist(eqd_gray.ravel(), 256, [0,256])
    axis[1,1].set_title('Equalised Histogram')
    axis[0,0].axis('off')
    axis[0,1].axis('off')
    plt.show()

def blurs(img):
    gaus_blur=cv.GaussianBlur(img, (25,25), 25.0, None, 25.0)
    bilat_blur=cv.bilateralFilter(img, 25, 75, 150)
    median_blur=cv.medianBlur(img, 25)
    fig, axs=plt.subplots(2, 2, figsize=(10,10))
    axs[0,0].imshow(img)
    axs[0,0].set_title('original')
    axs[1,0].imshow(gaus_blur)
    axs[1,0].set_title('Gaussian Blur')
    axs[0,1].imshow(bilat_blur)
    axs[1,1].imshow(median_blur)
    axs[0,1].set_title('bilateral blur')
    axs[1,1].set_title('median blur')
    for ax in axs.flat: ax.axis('off')
    plt.show()

def threshold(img):
    gray=turn_greyscale(img)
    retOTSU, dstOTSU=cv.threshold(gray, 159, 255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)
    retBIN, dstBIN=cv.threshold(gray, 127, 255, type=cv.THRESH_BINARY)
    dstADAPT=cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    retTRI, dstTRI=cv.threshold(gray, 127, 255, cv.THRESH_BINARY+cv.THRESH_TRIANGLE)
    fig, axs=plt.subplots(3,2, figsize=(13,13))
    axs[0,0].imshow(gray, cmap='gray')
    axs[0,0].set_title('Original')

    axs[0,1].imshow(dstOTSU, cmap='gray')
    axs[0,1].set_title(f"OTSU ret = {retOTSU}")

    axs[1,1].imshow(dstADAPT, cmap='gray')
    axs[1,1].set_title('Adapt')

    axs[1,0].imshow(dstTRI, cmap='gray')
    axs[1,0].set_title(f"TRI ret = {retTRI}")

    axs[2,0].imshow(dstBIN, cmap='gray')
    axs[2,0].set_title("Binary ret = 127")
    plt.show()

def ygradient(img):
    gray=img.copy()
    gray=cv.equalizeHist(gray)
    Iy=cv.Scharr(gray, cv.CV_64F, 0, 1)
    Ix = cv.Scharr(gray, cv.CV_64F, dx=1, dy=0)
    grad_mag=np.sqrt(Iy**2+Ix**2)
    grad_mag_norm=cv.normalize(grad_mag, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    fig, axs=plt.subplots(2,1, figsize=(15,15))
    axs[0].imshow(gray, cmap='gray')
    axs[1].imshow(grad_mag_norm, cmap='gray')
    plt.show()

def canny_edge(img):
    gray = img.copy()
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges_blur = cv.Canny(blurred, 100, 200)
    edges_unblur = cv.Canny(gray, 100, 200)
    fig, axs=plt.subplots(3,1, figsize=(15,15))
    axs[0].imshow(gray, cmap='gray')
    axs[1].imshow(edges_blur, cmap='gray')
    axs[2].imshow(edges_unblur, cmap='gray')
    plt.show()

def contours(img):
    #_, binotsu=cv.threshold(img, 50, 255, type=cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def contours_tree(img):
    #_, binotsu=cv.threshold(img, 50, 255, type=cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def morph_ops(img):
    img = reloadim(img)
    grey = turn_greyscale(img)
    ret, otsuthresh = cv.threshold(grey, 127, 255, type=cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    dialation=cv.dilate(otsuthresh, kernel, iterations=1)
    erosion=cv.erode(otsuthresh, kernel, iterations=1)
    opening=cv.morphologyEx(otsuthresh, cv.MORPH_OPEN, kernel)
    closing=cv.morphologyEx(otsuthresh, cv.MORPH_CLOSE, kernel)
    fig, axs = plt.subplots(3, 2, figsize = (25, 25))
    axs[0,0].imshow(dialation, cmap='gray')
    axs[0,0].set_title('dialation')
    axs[0,1].imshow(erosion, cmap='gray')
    axs[0,1].set_title('erosion')
    axs[1,0].imshow(opening, cmap='gray')
    axs[1,0].set_title('opening')
    axs[1,1].imshow(closing, cmap='gray')
    axs[1,1].set_title('closing')
    axs[2,0].imshow(img)
    axs[2,0].set_title('original ')
    axs[2,1].imshow(otsuthresh, cmap='gray')
    axs[2,1].set_title('otsu threshold')
    plt.show()