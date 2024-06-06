import cv2
import time
import glob
import os
import matplotlib.pyplot as plt

# susunan folder
# cv2-ns
# --> ori
#       --> p1-nomor1_ori.png
# --> defect
#       --> p1-nomor1_defect.png
# --> mask
#       --> p1-nomor1_mask.png
# --> inp
#       --> p1-nomor1_inp.png

def number_file(path):
    # path: c:\user\.....\cv2-ns\defect\p1-nomor1_defect.png
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
    # path: c:\user\.....\cv2-ns\defect\p1-nomor1_defect.png
    # mundur 1 dir, c:\....\cv2-ns\defect
    filebase =  os.path.dirname(path)   
    print("filebase:", filebase) 
    
    # mundur 1 dir, c:\....\cv2-ns
    filebase = os.path.dirname(filebase)
    print("filebase:", filebase) 

    return filebase

def process_file(path):
    start_time = time.time()
    damaged_img = cv2.imread(path) 
    print("path: ", path)
    # path: c:\user\.....\cv2-ns\defect\p1-nomor1_defect.png
    
    height, width = damaged_img.shape[0], damaged_img.shape[1]
    mask = damaged_img.copy()
    
    # Converting all pixels greater than zero to black while black becomes white
    for i in range(height):
        for j in range(width):
            if damaged_img[i, j].sum() > 0:
                mask[i, j] = 0
            else:
                mask[i, j] = [255, 255, 255]

    filemask = os.path.join(base_path(path), 'mask', number_file(path) + '_mask.png')
    print("filemask:", filemask) 
    # hasil C:\Users......\cv2-ns\mask\p1-nomor1_mask.png
    cv2.imwrite(filemask, mask)
    
    end_time = time.time()
    processing_time_mask = end_time - start_time
    print("Waktu pemrosesan mask:", processing_time_mask, "detik")
        
    mask = cv2.imread(filemask, 0)
    start_time = time.time()
    inp = cv2.inpaint(damaged_img, mask, 1, cv2.INPAINT_NS)
 
    fileinp = os.path.join(base_path(path), 'inp', number_file(path) + '_inp.png')
    print("fileinp:", fileinp)
    # hasil C:\Users......\cv2-ns\inp\p1-nomor1_inp.png
    cv2.imwrite(fileinp, inp)

    end_time = time.time()
    processing_time_inpaint = end_time - start_time
    print("Waktu pemrosesan inpaint:", processing_time_inpaint, "detik")

    #bikin plot
    fig, axes = plt.subplots(ncols=2, nrows=2)
    ax = axes.ravel()

    ax[0].set_title('Original image')
    fileori = os.path.join(base_path(path), 'ori', number_file(path) + '_ori.png')
    print("fileori:", fileori) 
    # hasil C:\Users\.......\cv2-ns\ori\p1-nomor1_ori.png
    ax[0].imshow(cv2.cvtColor(cv2.imread(fileori), cv2.COLOR_BGR2RGB))

    ax[1].set_title('Mask')
    ax[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)

    ax[2].set_title('Defected image')
    ax[2].imshow(cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB))

    ax[3].set_title('Inpainted image')
    ax[3].imshow((cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)))
    
    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()

def main():
    file_onepath = r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research-enrichment\inpainting\cv2-ns\defect\p1-nomor1_defect.png'
    folder_path = 'C:\\Users\\vinny\\OneDrive - Bina Nusantara\\Documents\\Research-enrichment\\inpainting\\cv2-ns\\defect'

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
