"""PROJECT: DAI
Processes medical images and extracts regions of interest, converting the images into a value that can be used to estimate whether the ROI Diagnosis is a diseases or not.

Create a class called AIDiagnosis that will process the image and produce a diagnostic.
- Base this class off of the NLP from spacy module."""
from skimage import data, io, filters
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    img = io.imread(r'C:\Users\Benjamin\Documents\Programming\Github\MLpython\Sample_Data\Medical_Imaging\CT_Scan\tiff_images\ID_0000_AGE_0060_CONTRAST_1_CT.tif')
    print('img shape: ', img.shape)
    signal = np.fft.fft2(img)
    new_img = np.fft.ifft2(signal)
    print('new img shape: ', new_img.shape)
    plt.figure(1)
    plt.imshow(img)
    plt.figure(2)
    plt.plot(signal)
    #plt.figure(3)
    #plt.imshow(new_img)
    plt.show()


if __name__ == "__main__":
    main()
