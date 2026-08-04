"""
scene_setup.py
===============

Environment preparation for Sionna RT: installation if missing, downloading
campus 3D meshes from Google Drive, and loading the scene.

Designed for Google Colab (hence the `!gdown` calls and `google.colab` imports),
but factored into functions so they can be easily skipped or replaced with
local paths when running outside Colab.
"""

import os
import zipfile

from . import config


def ensure_sionna_installed():
    """
    Checks whether `sionna.rt` is available; if not, installs `sionna-rt`
    and `plyfile` via pip. Replicates the try/except ImportError block from
    the first cell of the original notebook.
    """
    try:
        import sionna.rt  # noqa: F401
    except ImportError:
        print("sionna.rt not found — installing sionna-rt and plyfile...")
        os.system("pip install sionna-rt plyfile")
        import sionna.rt  # noqa: F401
    print("[OK] Sionna RT is available.")


def enable_colab_widgets():
    """
    Enables interactive ipywidgets in Google Colab (needed for 3D scene/path
    previews within the notebook). Safely skipped if not running in Colab.
    """
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
        print("[OK] Google Colab custom widget manager enabled.")
    except ImportError:
        print("[INFO] Not running in Google Colab — skipping enable_custom_widget_manager().")


def download_campus_meshes(gdrive_id=config.CAMPUS_MESHES_GDRIVE_ID,
                            zip_path=config.MESHES_ZIP_PATH,
                            extract_dir=config.MESHES_DIR):
    """
    Downloads the campus 3D mesh ZIP archive from Google Drive (`gdown`) and
    extracts it to `extract_dir`.

    Parameters
    ----------
    gdrive_id : str
        Google Drive file ID (the part after `/d/` in the URL).
    zip_path : str
        Local path where the ZIP archive will be saved.
    extract_dir : str
        Folder to extract the archive into.
    """
    print(f"Downloading campus meshes (Google Drive ID: {gdrive_id})...")
    os.system(f"gdown {gdrive_id}")

    print("Extracting meshes...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"[OK] Meshes extracted to '{extract_dir}/'.")


def load_campus_scene(scene_path=config.SCENE_XML_PATH,
                       frequency_hz=config.SCENE_FREQUENCY_HZ,
                       merge_shapes=False):
    """
    Loads the Mitsuba/Sionna XML campus scene and sets the operating frequency.

    Parameters
    ----------
    scene_path : str
        Path to the `.xml` scene file (Mitsuba format).
    frequency_hz : float
        Operating frequency in Hz (e.g., 3.5e9 for n78).
    merge_shapes : bool
        Whether Sionna should merge geometrically identical meshes for
        performance. Set to False here because we need separate objects
        (each building/material must remain individually addressable).

    Returns
    -------
    sionna.rt.Scene
        Loaded scene with frequency set.
    """
    from sionna.rt import load_scene

    print(f"Loading campus scene from '{scene_path}'...")
    scene = load_scene(scene_path, merge_shapes=merge_shapes)
    scene.frequency = frequency_hz
    print(f"[OK] Scene loaded. Frequency set to {scene.frequency / 1e9:.2f} GHz")
    return scene


def load_empty_scene(frequency_hz=config.SCENE_FREQUENCY_HZ):
    """
    Creates an empty Sionna scene (no buildings) – used in `verification.py`
    for clean (occlusion‑/reflection‑free) antenna pattern verification,
    isolated from the campus geometry.

    Returns
    -------
    sionna.rt.Scene
        Empty scene with frequency set.
    """
    from sionna.rt import load_scene

    print("Creating empty scene for clean antenna verification...")
    try:
        scene = load_scene(None)  # Sionna RT: None -> empty scene
        print("[OK] Empty scene created via load_scene(None).")
    except Exception as e:
        print(f"load_scene(None) failed ({e}), trying load_scene() without arguments...")
        scene = load_scene()

    scene.frequency = frequency_hz
    print(f"Frequency: {scene.frequency / 1e9:.2f} GHz")
    print(f"Objects in scene: {len(scene.objects)} (should be 0 or close to 0)")
    return scene
