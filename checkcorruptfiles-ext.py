import os #interaksi dg sistem operasi
import glob #mencari file
from multiprocessing import Pool #pemrosesan paralel
from PIL import Image #memanipulasi gambar

#cek satu persatu gambar
def check_one(f): 
    try:
        im = Image.open(f)
        im.verify() #cek corrupt gak
        im.close()
        return None  # No corruption detected
    except (IOError, OSError, Image.DecompressionBombError) as e:
        print(f"Error: {e}")
        return f  # File is corrupt

if __name__ == '__main__':
    # kalau  mau folder dari input user
    folder_path = input("Masukkan path folder yang ingin Anda periksa: ")
    if not os.path.isdir(folder_path):
        print("Folder tidak ditemukan.")
    
    # kalau folder langsung tembak
    # folder_path = r'C:\Users\vinny\OneDrive - Bina Nusantara\Documents\Research enrichment\jpg'
    p = Pool()

    # glob.glob untuk mencari file dengan format jpg, bisa ganti gif atau png
    files = [os.path.join(folder_path, f) for f in glob.glob(os.path.join(folder_path, "*.jpg"))]

    # print(f"Files to be checked: {len(files)}")

    # Map the list of files to check onto the Pool
    result = p.map(check_one, files)

    print(f"Files to be checked: {len(files)}")
    # Filter out None values (non-corrupt files)
    corrupt_files = list(filter(None, result))
    print(f"Number of corrupt files: {len(corrupt_files)}")
