import matplotlib.pyplot as plt
from skimage import color #converting image to grayscale
from skimage import io #reading writing image
import glob
import os
import time
import numpy as np #array-matrices
np.random.seed(1)

# susunan folder
# laplacian
# --> ori
#       --> p1-nomor1_ori.png
# --> defect
#       --> p1-nomor1_defect.png
# --> mask
#       --> p1-nomor1_mask.png
# --> inp
#       --> p1-nomor1_inp.png


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
        # print("iteration ke", it)
        # print("show iter: ", show_iter)
        # print("iteration lap2d")
        # print("\n")
        x = prox_2d(z, w, lmbda, denominator)
        z = update_z(y_train, pos_train, x, w, lmbda, eta)
        w = update_w(x, z, w, lmbda)
        
        if (it + 1) % show_iter == 0:
            print(it + 1)
            print(compute_rse(y_test, x[pos_test]))
            print()
            # io.imshow(z)
            # plt.axis('off')
            # plt.show()

    return x

def number_file(path):
    # path: c:\user\.....\laplacian\defect\p1-nomor1_defect.png
    # mengambil nama file + type + extension
    filename = os.path.basename(path) 
    print("filename ",filename)
    # hasilnya p1-nomor1_ori.png
    
    # mengambil nama file
    filenumber = filename.rsplit('_', 1)[0]
    print("filenumber",filenumber) 
    # hasilnya p1-nomor1
    
    return filenumber

def base_path(path):
    # path: c:\user\.....\laplacian\defect\p1-nomor1_defect.png
    # mundur 1 dir, c:\....\laplacian\defect
    filebase =  os.path.dirname(path)   
    print("filebase:", filebase) 
    
    # mundur 1 dir, c:\....\laplacian
    filebase = os.path.dirname(filebase)
    print("filebase:", filebase) 

    return filebase

def process_file(path):
    start_time = time.time()
    img = io.imread(path)
    print("path: ", path)
    # path: c:\user\.....\laplacian\defect\p1-nomor1_defect.png

    # convert image to grayscale
    if img.shape[2] == 4:
        img = img[:, :, :3]    
    imgGray = color.rgb2gray(img)
    print("img shape: ", img.shape)
   
    M, N = imgGray.shape
    missing_rate = 0.9
    mask = imgGray * np.round(np.random.rand(M, N)+ 0.5 - missing_rate)

    print("sparse shape: ", mask.shape)
    # io.imshow(mask, cmap='gray')
    # plt.axis('off')
    # plt.show()
    filemask = os.path.join(base_path(path), 'mask', number_file(path) + '_mask.png')
    print("filemask:", filemask)
    # hasil C:\Users......\laplacian\mask\p1-nomor1_mask.png

    # supaya bs save
    sparse_img_uint8 = (mask * 255).astype(np.uint8)
    io.imsave(filemask, sparse_img_uint8)
    end_time = time.time()
    processing_time_mask = end_time - start_time
    print("Waktu pemrosesan mask:", processing_time_mask, "detik")
 
    start_time = time.time()
    # proses inpainting
    lmbda = 5e-3 * M * N
    gamma = 1 * lmbda
    eta = 100 * lmbda
    tau = 2
    maxiter = 100
    mat_hat = laplacian_conv_2d(imgGray, mask, lmbda, gamma, eta, tau, maxiter)
    
    mat_hat[mat_hat < 0] = 0
    mat_hat[mat_hat > 1] = 1
    # io.imshow(mat_hat)
    # plt.axis('off')
    
    fileinp = os.path.join(base_path(path), 'inp', number_file(path) + '_inp.png')
    print("fileinp:", fileinp)
    # hasil C:\Users......\laplacian\inp\p1-nomor1_inp.png

    plt.imsave(fileinp.format(tau), mat_hat, cmap = plt.cm.gray)
    end_time = time.time()
    processing_time_inpaint = end_time - start_time
    print("Waktu pemrosesan inpaint:", processing_time_inpaint, "detik")

    #bikin plot
    fig, axes = plt.subplots(ncols=2, nrows=2)
    ax = axes.ravel()

    ax[0].set_title('Original image')
    fileori = os.path.join(base_path(path), 'ori', number_file(path) + '_ori.png')
    print("fileori:", fileori) 
    # hasil C:\Users\.......\laplacian\ori\p1-nomor1_ori.png
    ax[0].imshow(io.imread(fileori))

    ax[1].set_title('Mask')
    ax[1].imshow(mask, cmap=plt.cm.gray)

    ax[2].set_title('Defected image')
    ax[2].imshow(img)

    ax[3].set_title('Inpainted image')
    ax[3].imshow(io.imread(fileinp))
    
    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()

def main():
    file_onepath = r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research-enrichment\inpainting\laplacian\defect\p1-nomor50_defect.png'
    folder_path = 'C:\\Users\\vinny\\OneDrive - Bina Nusantara\\Documents\\Research-enrichment\\inpainting\\laplacian\\defect'
    
    # comment yang tidak digunakan
    file_use = folder_path 
    # file_use = file_onepath 
    n = 1
 
    if os.path.isfile(file_use):
        print("file")
        process_file(file_use)
    elif os.path.isdir(file_use):
        file_type = '*.png'  
        # Menggunakan glob untuk mencocokkan pola file
        file_list = glob.glob(os.path.join(folder_path, file_type))

        # Iterasi semua file yang cocok
        for file_path in file_list:
            print("file pathdisinii: ", file_path)
            absolute_path = os.path.abspath(file_path)
            print("file pathdisinii: ", file_path)
            process_file(absolute_path)
            print("=====================")
            print("process file ke: ", n)
            print("=====================")
            n = n + 1
            

if __name__ == "__main__":
    main()
