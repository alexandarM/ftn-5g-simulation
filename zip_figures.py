import os
import zipfile

def zip_figures(output_dir="figures", zip_name=None):
    
    #Zips the ENTIRE figures folder (all subfolders and files) and downloads it.

    if not os.path.exists(output_dir):
        print(f"[ERROR] Folder '{output_dir}' does not exist")
        return

    if zip_name is None:
        zip_name = f"{output_dir}.zip"

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=os.path.dirname(output_dir))
                zipf.write(file_path, arcname)

    print(f"[OK] ZIP created: {zip_name}")
    print(f"   Size: {os.path.getsize(zip_name) / 1024:.2f} KB")

    
    try:
        from google.colab import files
        files.download(zip_name)
        print(f"[OK] Downloading: {zip_name}")
    except ImportError:
        print("[INFO] Not ran in Colab - saved locally")

