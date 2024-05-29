import glob
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage import laplace
import skimage
from skimage.measure import label, regionprops
from skimage import io

# susunan folder
# bi-laplacian
# --> ori
#       --> p1-nomor1_ori.png
# --> defect
#       --> p1-nomor1_defect.png
# --> mask
#       --> p1-nomor1_mask.png
# --> inp
#       --> p1-nomor1_inp.png

def numbergrid(mask):
    n = np.sum(mask)
    G1 = np.zeros_like(mask, dtype=np.uint32)
    G1[mask] = np.arange(1, n+1)
    return G1

def delsq_laplacian(G):
    [m, n] = G.shape
    G1 = G.flatten()
    p = np.where(G1)[0]
    N = len(p)
    i = G1[p] - 1
    j = G1[p] - 1
    s = 4 * np.ones(N)
    #for all four neighbor of center points
    for offset in [-1, m, 1, -m]:
        #indices of all possible neighbours in sparse matrix
        Q = G1[p+offset]
        #filter inner indices
        q = np.where(Q)[0]
       
        #generate neighbour coordinate
        i = np.concatenate([i, G1[p[q]]-1])
        j = np.concatenate([j, Q[q]-1])
        s = np.concatenate([s, -np.ones(q.shape)])

    sp = sparse.csr_matrix((s, (i,j)), (N,N))
    return sp

def delsq_bilaplacian(G):
    [n, m] = G.shape
    G1 = G.flatten()
    p = np.where(G1)[0]
    N = len(p)
    i = G1[p] - 1
    j = G1[p] - 1
    s = 20 * np.ones(N)
    #for all four neighbor of center points
    coeffs  = np.array([1, 2, -8, 2, 1, -8, -8, 1, 2, -8, 2, 1])
    offsets = np.array([-2*m, -m-1, -m, -m+1, -2, -1, 1, 2, m-1, m, m+1, 2*m])
    loop_break = False

    for coeff, offset in zip(coeffs, offsets):  
        #indices of all possible neighbours in sparse matrix    
        if np.all((p+offset) < len(G1)):
            Q = G1[p+offset]
            #filter inner indices
            q = np.where(Q)[0]
            #generate neighbour coordinate
            i = np.concatenate([i, G1[p[q]]-1])
            j = np.concatenate([j, Q[q]-1])
            s = np.concatenate([s, coeff*np.ones(q.shape)])
        else:
            loop_break = True
        if loop_break:
            break
        
    sp = sparse.csr_matrix((s, (i,j)), (N,N))
    return sp

def generate_stencials():
    stencils = []
    for i in range(5):
        for j in range(5):
            A = np.zeros((5, 5))
            A[i,j]=1
            S = laplace(laplace(A))
            x_range = np.array([i-2, i+3]).clip(0,5)
            y_range = np.array([j-2, j+3]).clip(0,5)
            S = S[x_range[0]:x_range[1], y_range[0]:y_range[1]]
            stencils.append(S)

    return stencils

def _inpaint_biharmonic_single_channel(mask, out, limits):
    # Initialize sparse matrices
    matrix_unknown = sparse.lil_matrix((np.sum(mask), out.size))
    matrix_known = sparse.lil_matrix((np.sum(mask), out.size))

    # Find indexes of masked points in flatten array
    mask_i = np.ravel_multi_index(np.where(mask), mask.shape)

    G = numbergrid(mask)
    L = delsq_bilaplacian(G)
    out[mask] = 0
    B = -laplace(laplace(out))
    b = B[mask]
    result = spsolve(L, b)
    # Handle enormous values
    result = np.clip(result, *limits)
    result = result.ravel()
    out[mask] = result
    return out

def dilate_rect(rect, d, nd_shape):
    rect[0:2] = (rect[0:2] - d).clip(min = 0)
    rect[2:4] = (rect[2:4] + d).clip(max = nd_shape)
    return rect

def k_inpaint_biharmonic(image, mask, multichannel=False):
    if image.ndim < 1:
        raise ValueError('Input array has to be at least 1D')
    img_baseshape = image.shape[:-1] if multichannel else image.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')

    if np.ma.isMaskedArray(image):
        raise TypeError('Masked arrays are not supported')

    image = skimage.img_as_float(image)
    mask = mask.astype(bool)

    # Split inpainting mask into independent regions
    kernel = ndi.morphology.generate_binary_structure(mask.ndim, 1)
    mask_dilated = ndi.morphology.binary_dilation(mask, structure=kernel)
    mask_labeled, num_labels = label(mask_dilated, return_num=True)
    mask_labeled *= mask
    if not multichannel:
        image = image[..., np.newaxis]

    out = np.copy(image)

    props = regionprops(mask_labeled)
    comp_out_imgs = []
    comp_masks = []
    for i in range(num_labels):
        rect = np.array(props[i].bbox)
        rect = dilate_rect(rect, 2, image.shape[:2])
        out_sub_img = out[rect[0]:rect[2], rect[1]:rect[3], :]
        comp_mask   = mask[rect[0]:rect[2], rect[1]:rect[3]]
        comp_out_imgs.append(out_sub_img)
        comp_masks.append(comp_mask)

    for idx_channel in range(image.shape[-1]):
        known_points = image[..., idx_channel][~mask]
        limits = (np.min(known_points), np.max(known_points))
        for i in range(num_labels):
            _inpaint_biharmonic_single_channel(comp_masks[i], comp_out_imgs[i][..., idx_channel], limits)

    if not multichannel:
        out = out[..., 0]

    return out


def number_file(path):
    # path: c:\user\.....\bi-laplacian\defect\p1-nomor1_defect.png
    # mengambil nama file + type + extension
    filename = os.path.basename(path) 
    print("filename ",filename)
    # hasilnya p1-nomor1_defect.png
    
    # mengambil nama file
    filenumber = filename.rsplit('_', 1)[0]
    print("filenumber",filenumber) 
    # hasilnya p1-nomor1
    
    return filenumber

def base_path(path):
    # path: c:\user\.....\bi-laplacian\defect\p1-nomor1_defect.png
    # mundur 1 dir, c:\....\bi-laplacian\defect
    filebilap =  os.path.dirname(path)   
    print("filebilap:", filebilap) 
    
    # mundur 1 dir, c:\....\bi-laplacian
    filebilap = os.path.dirname(filebilap)
    print("filebilap:", filebilap) 

    return filebilap

def process_file(path):
    start_time = time.time()
    image_defect = io.imread(path) 
    print("path: ", path)
    # path: c:\user\.....\bi-laplacian\defect\p1-nomor1_defect.png
    print("img df sh: ", image_defect.shape)
    print("img df sh: ", image_defect.shape[2])
    if len(image_defect.shape) == 3 or image_defect.shape[2] == 4:
        # Jika gambar memiliki 4 saluran, konversi ke 3 saluran dengan menghapus saluran alpha
        image_defect = image_defect[:, :, :3]
    print("img df sh: ", image_defect.shape)
    print("img df sh: ", image_defect.shape[2])

    height, width = image_defect.shape[0], image_defect.shape[1]
    mask = np.zeros(image_defect.shape[:-1])

    # Converting all pixels greater than zero to black while black becomes white
    for i in range(height):
        for j in range(width):
            if image_defect[i, j].sum() > 0:
                mask[i, j] = 0
            else:
                mask[i, j] = 1
    # io.imshow(mask)
    # plt.show()

    filemask = os.path.join(base_path(path), 'mask', number_file(path) + '_mask.png')
    print("filemask:", filemask) 
    # hasil C:\Users......\bi-laplacian\mask\p1-nomor1_mask.png
   
    save_mask = (mask * 255).astype(np.uint8)
    io.imsave(filemask, save_mask)
    end_time = time.time()
    processing_time_mask = end_time - start_time
    print("Waktu pemrosesan mask:", processing_time_mask, "detik")
        
    start_time = time.time()
    image_result = k_inpaint_biharmonic(image_defect, mask, multichannel=True)
    sparse_img_uint8 = (image_result * 255).astype(np.uint8)
    
    fileinp = os.path.join(base_path(path), 'inp', number_file(path) + '_inp.png')
    print("fileinp:", fileinp)
    io.imsave(fileinp, sparse_img_uint8)
    # hasil C:\Users......\bi-laplacian\inp\p1-nomor1_inp.png
    
    end_time = time.time()
    processing_time_inpaint = end_time - start_time
    print("Waktu pemrosesan inpaint:", processing_time_inpaint, "detik")

    #bikin plot
    fig, axes = plt.subplots(ncols=2, nrows=2)
    ax = axes.ravel()

    ax[0].set_title('Original image')
    fileori = os.path.join(base_path(path), 'ori', number_file(path) + '_ori.png')
    print("fileori:", fileori) 
    # hasil C:\Users\.......\bi-laplacian\ori\p1-nomor1_ori.png

    ax[0].imshow(io.imread(fileori))

    ax[1].set_title('Mask')
    ax[1].imshow(mask, cmap=plt.cm.gray)

    ax[2].set_title('Defected image')
    ax[2].imshow(image_defect)

    ax[3].set_title('Inpainted image')
    ax[3].imshow(image_result)
    
    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()



def main():
    file_onepath = r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research-enrichment\inpainting\bi-laplacian\defect\p2-nomor99_defect.png'
    folder_path = 'C:\\Users\\vinny\\OneDrive - Bina Nusantara\\Documents\\Research-enrichment\\inpainting\\bi-laplacian\\defect'

    # comment yang tidak digunakan
    # file_use = folder_path 
    file_use = file_onepath 
    n = 1
 
    if os.path.isfile(file_use):
        print("file")
        process_file(file_use)
    elif os.path.isdir(file_use):
        file_type = '*.png'  
        file_list = glob.glob(os.path.join(folder_path, file_type))
        
        # iterasi semua file yang cocok
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
