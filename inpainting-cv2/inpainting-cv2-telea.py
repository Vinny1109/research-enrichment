import cv2
import numpy as np
 
# reading the damaged image
# EDIT PATH
damaged_img = cv2.imread(filename=r"C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research enrichment\inpainting\KIR2.png") 
 
# get the shape of the image
height, width = damaged_img.shape[0], damaged_img.shape[1]
 
# Converting all pixels greater than zero to black while black becomes white
for i in range(height):
    for j in range(width):
        if damaged_img[i, j].sum() > 0:
            damaged_img[i, j] = 0
        else:
            damaged_img[i, j] = [255, 255, 255]
 
# saving the mask 
mask = damaged_img
# EDIT PATH
cv2.imwrite(r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research enrichment\inpainting\KIR2_mask.png', mask)
 
# displaying mask
cv2.imshow("damaged image mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===============================
# Open the image.
# EDIT PATH
img = cv2.imread(r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research enrichment\inpainting\KIR2.png')
 
# Load the mask.
# EDIT PATH
mask = cv2.imread(r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research enrichment\inpainting\KIR2_mask.png', 0)
 
# Inpaint.
dst = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
# coba pake telea, disini radius nya 1, dan pakai telea
 
# Write the output.
# EDIT PATH
cv2.imwrite(r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research enrichment\inpainting\KIR2_telea.png', dst)

# displaying result
cv2.imshow("result inpainting", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
