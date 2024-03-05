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
    png_files = [os.path.join(folder_path, f) for f in glob.glob(os.path.join(folder_path, "*.png"))]
    jpg_files = [os.path.join(folder_path, f) for f in glob.glob(os.path.join(folder_path, "*.jpg"))]
    gif_files = [os.path.join(folder_path, f) for f in glob.glob(os.path.join(folder_path, "*.gif"))]

    files = png_files + jpg_files + gif_files

    print(f"Files to be checked: {len(files)}")    
    # Map the list of files to check onto the Pool
    result = p.map(check_one, files)

    # Counting corrupt files for each format
    corrupt_png = sum(1 for file in result if file in png_files)
    corrupt_jpg = sum(1 for file in result if file in jpg_files)
    corrupt_gif = sum(1 for file in result if file in gif_files)

    print(f"Number of PNG files: {len(png_files)}")
    print(f"Number of JPG files: {len(jpg_files)}")
    print(f"Number of GIF files: {len(gif_files)}")
    print(f"Number of corrupt PNG files: {corrupt_png}")
    print(f"Number of corrupt JPG files: {corrupt_jpg}")
    print(f"Number of corrupt GIF files: {corrupt_gif}")