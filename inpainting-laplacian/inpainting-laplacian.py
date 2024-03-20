import numpy as np #array-matrices

# Computes the relative squared error between two arrays.
# antara gambar asli dengan gambar yg di inpaint
def compute_rse(var, var_hat):
    return np.linalg.norm(var - var_hat, 2) / np.linalg.norm(var, 2)

# Generates a 1D Laplacian kernel with the specified length T and parameter tau
# mendeteksi area tepi pada gambar
def laplacian(T, tau):
    ell = np.zeros(T)
    ell[0] = 2 * tau
    for k in range(tau):
        ell[k + 1] = -1
        ell[-k - 1] = -1
    return ell

# Computes the proximal operator for a 2D signal.
# memperbarui nilai piksel gambar
def prox_2d(z, w, lmbda, denominator):
    M, N = z.shape
    temp = np.fft.fft2(lmbda * z - w) / denominator
    temp1 = 1 - M * N / (lmbda * np.abs(temp))
    temp1[temp1 <= 0] = 0
    return np.fft.ifft2(temp * temp1).real

# Updates the variable z using the observed and missing parts of the signal.
# ?memperbarui variabel z menggunakan bagian yang diamati dan hilang dari sinyal.
def update_z(y_train, pos_train, x, w, lmbda, eta):
    z = x + w / lmbda
    z[pos_train] = (lmbda / (lmbda + eta) * z[pos_train] 
                    + eta / (lmbda + eta) * y_train)
    return z

#  Updates the variable w.
# memperbarui variabel w, yang merupakan bagian dari proses iteratif untuk menemukan solusi yang optimal.
def update_w(x, z, w, lmbda):
    return w + lmbda * (x - z)

# Performs Laplacian convolution-based inpainting for 2D signals.
# melakukan inpainting berbasis konvolusi Laplacian untuk sinyal 2D. Ini mengatur proses iteratif 
# dan memanggil fungsi lainnya untuk memperbarui gambar.
def laplacian_conv_2d(y_true, y, lmbda, gamma, eta, tau, maxiter = 50):
    M, N = y.shape
    pos_train = np.where(y != 0)
    y_train = y[pos_train]
    pos_test = np.where((y_true != 0) & (y == 0))
    y_test = y_true[pos_test]
    z = y.copy()
    w = y.copy()
    ell_1 = laplacian(M, tau)
    ell_2 = laplacian(N, tau)
    denominator = lmbda + gamma * np.fft.fft2(np.outer(ell_1, ell_2)) ** 2
    del y_true, y
    show_iter = 10
    for it in range(maxiter):
        x = prox_2d(z, w, lmbda, denominator)
        z = update_z(y_train, pos_train, x, w, lmbda, eta)
        w = update_w(x, z, w, lmbda)
        if (it + 1) % show_iter == 0:
            print(it + 1)
            print(compute_rse(y_test, x[pos_test]))
            print()
    return x

# masking 90% pixels as missing values
import numpy as np
np.random.seed(1)

import matplotlib.pyplot as plt
from skimage import color #converting image to grayscale
from skimage import io #reading writing image
img = io.imread(r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research enrichment\inpainting\laplacian\test3.png') #png
io.imshow(img)
plt.show()

# convert image to grayscale
if img.shape[2] == 4:
    img = img[:, :, :3]    
imgGray = color.rgb2gray(img)

# create the masked image
M, N = imgGray.shape
missing_rate = 0.9
sparse_img = imgGray * np.round(np.random.rand(M, N) + 0.5 - missing_rate)

io.imshow(sparse_img, cmap='gray')
plt.axis('off')
plt.show()
# supaya bs save
sparse_img_uint8 = (sparse_img * 255).astype(np.uint8)
io.imsave(r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research enrichment\inpainting\laplacian\test3_masked_image.png', sparse_img_uint8)

# proses inpainting
lmbda = 5e-3 * M * N
gamma = 1 * lmbda
eta = 100 * lmbda
tau = 2
maxiter = 100
mat_hat = laplacian_conv_2d(imgGray, sparse_img, lmbda, gamma, eta, tau, maxiter)

mat_hat[mat_hat < 0] = 0
mat_hat[mat_hat > 1] = 1
io.imshow(mat_hat)
plt.axis('off')
plt.imsave(r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research enrichment\inpainting\laplacian\test3_gray_recovery.png'.format(tau), 
           mat_hat, cmap = plt.cm.gray)
plt.show()
