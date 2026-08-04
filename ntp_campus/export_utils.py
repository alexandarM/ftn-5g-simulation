"""
export_utils.py
=================

Utility for packaging all generated plots (from the `figures/` folder) into
a ZIP archive and downloading it in a Google Colab environment.
"""

import os
import zipfile


def download_figures_folder(folder_path="figures", zip_name="figures.zip"):
    """
    Creates a ZIP archive from all files in `folder_path`. If running in
    Google Colab, triggers a browser download. Outside Colab, the ZIP is
    created locally and a message is printed.

    Parameters
    ----------
    folder_path : str
        Folder to archive (e.g., "figures").
    zip_name : str
        Name of the output ZIP file.
    """
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder '{folder_path}' does not exist!")
        return

    with zipfile.ZipFile(zip_name, "w") as zipf:
        for root, dirs, files_list in os.walk(folder_path):
            for file in files_list:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=os.path.dirname(folder_path))
                zipf.write(file_path, arcname)
                print(f"  Added: {file_path}")

    print(f"\n[OK] ZIP archive created: {zip_name}")
    print(f"   Size: {os.path.getsize(zip_name) / 1024:.2f} KB")

    try:
        from google.colab import files
        files.download(zip_name)
        print(f"[OK] Download started: {zip_name}")
    except ImportError:
        print("[INFO] Not running in Google Colab — ZIP created locally, "
              "skipping automatic download.")
