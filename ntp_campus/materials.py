"""
materials.py
=============

Utilities for:
  1) reading building geometry (bounding boxes) from campus PLY mesh files,
  2) defining custom radio materials (concrete, brick, glass) with scattering
     parameters for diffuse reflection,
  3) assigning those materials to scene objects based on filename/object name
     suffixes (`_brick` or `_concrete`),
  4) printing / verifying material assignments.
"""

import os

import numpy as np
from plyfile import PlyData
from sionna.rt import RadioMaterial

from . import config

# Names of custom radio materials that have scattering_coefficient > 0.
# Used to check whether the scene is ready for diffuse scattering.
DIFFUSION_MATERIAL_NAMES = ("concrete_scattering", "brick_scattering", "glass_scattering")


# =========================================================================
# 1. READING BUILDING GEOMETRY FROM PLY FILES
# =========================================================================
def _read_ply_vertices(filepath):
    """Loads (x, y, z) coordinates of all vertices from a single PLY file."""
    ply_data = PlyData.read(filepath)
    vertex_data = ply_data["vertex"]
    return np.array([(v[0], v[1], v[2]) for v in vertex_data])


def load_building_dimensions(ply_folder=config.PLY_FOLDER):
    """
    Iterates over all `.ply` files in `ply_folder` (skipping `Plane.ply`,
    i.e., the ground) and computes bounding box (width/depth/height) and
    centre for each. Files with the same "base name" (without `_brick` or
    `_concrete` suffix) represent different material parts of the same
    physical building, and can be grouped later via `group_buildings`.

    Returns
    -------
    list[dict]
        One dict per PLY file with keys: filename, base_name,
        width, depth, height, center_x/y/z, min_x/y/z, max_x/y/z,
        num_vertices.
    """
    building_data = []
    print("\n--- Building dimensions from PLY files ---")

    for filename in os.listdir(ply_folder):
        if not filename.endswith(".ply") or filename.lower().startswith("plane"):
            continue

        filepath = os.path.join(ply_folder, filename)
        try:
            vertices = _read_ply_vertices(filepath)
            if len(vertices) == 0:
                continue

            min_xyz = vertices.min(axis=0)
            max_xyz = vertices.max(axis=0)
            dims = max_xyz - min_xyz

            base_name = filename.replace(".ply", "")
            if base_name.endswith("_concrete") or base_name.endswith("_brick"):
                base_name = base_name.rsplit("_", 1)[0]

            building_data.append({
                "filename": filename,
                "base_name": base_name,
                "width": dims[0], "depth": dims[1], "height": dims[2],
                "center_x": (min_xyz[0] + max_xyz[0]) / 2,
                "center_y": (min_xyz[1] + max_xyz[1]) / 2,
                "center_z": (min_xyz[2] + max_xyz[2]) / 2,
                "min_x": min_xyz[0], "min_y": min_xyz[1], "min_z": min_xyz[2],
                "max_x": max_xyz[0], "max_y": max_xyz[1], "max_z": max_xyz[2],
                "num_vertices": len(vertices),
            })
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    return building_data


def group_buildings(building_data):
    """
    Groups `load_building_dimensions` output by `base_name`, so that all
    material parts (e.g., `_brick` and `_concrete` versions) of the same
    building are in one list. Returns `dict[base_name] -> list[dict]`.
    """
    groups = {}
    for item in building_data:
        groups.setdefault(item["base_name"], []).append(item)
    return groups


def print_building_summary(building_data):
    """Prints a summary table of individual PLY files and grouped statistics per building."""
    if not building_data:
        print("No PLY files were successfully processed.")
        return

    print(f"{'Name':<45} {'Width (m)':<12} {'Depth (m)':<12} {'Height (m)':<12} {'Vertices':<8}")
    print("-" * 95)
    for data in building_data:
        print(f"{data['base_name']:<45} {data['width']:<12.2f} {data['depth']:<12.2f} "
              f"{data['height']:<12.2f} {data['num_vertices']:<8}")

    print("\n--- Statistics by building (grouped) ---")
    groups = group_buildings(building_data)

    print(f"{'Building name':<45} {'Width (m)':<12} {'Depth (m)':<12} {'Height (m)':<12} {'Materials':<10}")
    print("-" * 95)

    widths_all, depths_all, heights_all = [], [], []
    for name, items in groups.items():
        avg_width = np.mean([d["width"] for d in items])
        avg_depth = np.mean([d["depth"] for d in items])
        avg_height = np.mean([d["height"] for d in items])
        mats = ", ".join(["brick" if "brick" in d["filename"] else "concrete" for d in items])

        widths_all.append(avg_width)
        depths_all.append(avg_depth)
        heights_all.append(avg_height)

        print(f"{name:<45} {avg_width:<12.2f} {avg_depth:<12.2f} {avg_height:<12.2f} {mats:<10}")

    print("\n--- Overall statistics ---")
    print(f"Number of unique buildings: {len(groups)}")
    print(f"Number of PLY files (material parts): {len(building_data)}")
    print(f"Average width: {np.mean(widths_all):.2f} m")
    print(f"Average depth: {np.mean(depths_all):.2f} m")
    print(f"Average height: {np.mean(heights_all):.2f} m")
    print(f"Maximum height: {np.max(heights_all):.2f} m")
    print(f"Minimum height: {np.min(heights_all):.2f} m")


def print_building_group_details(groups, ply_folder=config.PLY_FOLDER, filter_name=None):
    """
    Detailed printout of dimensions/coordinates for all (or filtered)
    building groups, including the overall bounding box when a building
    has multiple material parts.

    Parameters
    ----------
    groups : dict
        Output of `group_buildings`.
    ply_folder : str
        Folder where the original `.ply` files are located (for re-reading
        vertex data when computing the overall bounding box).
    filter_name : str or None
        If given, only buildings whose `base_name` contains this substring
        are displayed.
    """
    items_to_print = {
        name: items for name, items in sorted(groups.items())
        if filter_name is None or filter_name in name
    }

    if not items_to_print:
        print(f"No buildings match filter '{filter_name}'.")
        return

    print("\n" + "=" * 110)
    print("DETAILED DIMENSIONS AND COORDINATES" + (f" (filter: {filter_name})" if filter_name else ""))
    print("=" * 110)

    for name, items in items_to_print.items():
        print(f"\n{'─' * 110}")
        print(f"BUILDING: {name}")
        print(f"{'─' * 110}")

        for item in items:
            mat = "BRICK" if "brick" in item["filename"] else "CONCRETE"
            print(f"\n  Material: {mat}")
            print(f"     Width (X): {item['width']:.2f} m")
            print(f"     Depth (Y): {item['depth']:.2f} m")
            print(f"     Height (Z): {item['height']:.2f} m")
            print(f"     Centre: ({item['center_x']:.2f}, {item['center_y']:.2f}, {item['center_z']:.2f}) m")
            print(f"     Bounding box min: ({item['min_x']:.2f}, {item['min_y']:.2f}, {item['min_z']:.2f}) m")
            print(f"     Bounding box max: ({item['max_x']:.2f}, {item['max_y']:.2f}, {item['max_z']:.2f}) m")
            print(f"     Vertex count: {item['num_vertices']}")

        if len(items) > 1:
            all_verts = [_read_ply_vertices(os.path.join(ply_folder, item["filename"])) for item in items]
            all_verts = np.vstack(all_verts)
            min_xyz = all_verts.min(axis=0)
            max_xyz = all_verts.max(axis=0)
            total_dims = max_xyz - min_xyz
            total_center = (min_xyz + max_xyz) / 2

            print(f"\n  OVERALL (all parts):")
            print(f"     Total width (X): {total_dims[0]:.2f} m")
            print(f"     Total depth (Y): {total_dims[1]:.2f} m")
            print(f"     Total height (Z): {total_dims[2]:.2f} m")
            print(f"     Centre: ({total_center[0]:.2f}, {total_center[1]:.2f}, {total_center[2]:.2f}) m")
            print(f"     Total vertex count: {len(all_verts)}")

    print(f"\n{'=' * 110}")
    print(f"TOTAL BUILDINGS: {len(items_to_print)}")
    print("=" * 110)


# =========================================================================
# 2. DEFINING AND ASSIGNING RADIO MATERIALS (FOR DIFFUSE SCATTERING)
# =========================================================================
def define_radio_materials(diffusion_cfg=None):
    """
    Defines three custom `RadioMaterial` objects (concrete, brick, glass)
    with Lambertian diffuse scattering parameters. The objects are
    automatically registered with Sionna RT upon construction.

    Parameters
    ----------
    diffusion_cfg : run_config.DiffusionConfig or None
        If given, scattering coefficients and `scattering_pattern` are taken
        from it. If None, default values are used (concrete=0.3, brick=0.4,
        glass=0.05, pattern="lambertian").

    Returns
    -------
    dict
        {"concrete": RadioMaterial, "brick": RadioMaterial, "glass": RadioMaterial}
    """
    if diffusion_cfg is not None:
        concrete_sc = diffusion_cfg.concrete_scattering_coefficient
        brick_sc = diffusion_cfg.brick_scattering_coefficient
        glass_sc = diffusion_cfg.glass_scattering_coefficient
        pattern = diffusion_cfg.scattering_pattern
    else:
        concrete_sc, brick_sc, glass_sc, pattern = 0.3, 0.4, 0.05, "lambertian"

    concrete = RadioMaterial(
        name="concrete_scattering",
        relative_permittivity=5.24, conductivity=0.123, thickness=0.3,
        scattering_coefficient=concrete_sc, scattering_pattern=pattern,
    )
    brick = RadioMaterial(
        name="brick_scattering",
        relative_permittivity=3.91, conductivity=0.029, thickness=0.25,
        scattering_coefficient=brick_sc, scattering_pattern=pattern,
    )
    glass = RadioMaterial(
        name="glass_scattering",
        relative_permittivity=2.5, conductivity=0.0, thickness=0.01,
        scattering_coefficient=glass_sc, scattering_pattern=pattern,
    )
    print(f"[OK] Materials defined (scattering: concrete={concrete_sc}, "
          f"brick={brick_sc}, glass={glass_sc}, pattern={pattern!r}).")
    return {"concrete": concrete, "brick": brick, "glass": glass}


def assign_materials_to_scene(scene, groups, materials):
    """
    Assigns `concrete` or `brick` materials to each scene object based on
    suffix in the name (`_brick` or `_concrete`); if missing, defaults to
    concrete. Objects whose name contains "plane" (i.e., ground) are skipped.

    Parameters
    ----------
    scene : sionna.rt.Scene
        The campus scene (already loaded).
    groups : dict
        Output of `group_buildings` – used to check that the object belongs
        to a known building.
    materials : dict
        Output of `define_radio_materials`.

    Returns
    -------
    int
        Number of objects assigned.
    """
    print("\n--- Assigning materials to objects ---")
    assigned_count = 0

    for obj_name, obj in scene.objects.items():
        if "plane" in obj_name.lower():
            continue

        base_name = obj_name
        for suffix in ["_brick", "_concrete"]:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break
        if base_name.endswith(".ply"):
            base_name = base_name[:-4]

        if base_name not in groups:
            continue

        if obj_name.endswith("_brick.ply") or obj_name.endswith("_brick"):
            material, mat_label = materials["brick"], "BRICK"
        elif obj_name.endswith("_concrete.ply") or obj_name.endswith("_concrete"):
            material, mat_label = materials["concrete"], "CONCRETE"
        else:
            material, mat_label = materials["concrete"], "CONCRETE (default)"

        obj.radio_material = material
        assigned_count += 1
        print(f"  [OK] {obj_name} -> {mat_label}")

    print(f"\n[OK] Assigned materials to {assigned_count} objects.")
    return assigned_count


def fix_remaining_itu_concrete(scene, concrete_material):
    """
    Patch for objects that still have the default built-in ITU "concrete"
    material (from the XML scene, named "itu_concrete" or just "concrete"
    depending on the Sionna RT version) – reassigns them to our custom
    `concrete_material` so they also get a scattering coefficient.

    Note: `obj.radio_material` is a `RadioMaterial` object, not a string,
    so it must be compared via its `.name` attribute rather than directly
    against a string (a direct `==` comparison against a string is always
    False and would silently skip every object).
    """
    default_concrete_names = {"itu_concrete", "concrete"}
    for obj_name, obj in scene.objects.items():
        mat = obj.radio_material
        mat_name = mat.name if mat is not None else None
        if mat_name in default_concrete_names:
            obj.radio_material = concrete_material
            print(f"[OK] {obj_name} -> concrete_scattering")


def print_scene_material_summary(scene):
    """Prints all defined radio materials and each object's assigned material (excluding ground)."""
    print("--- Currently defined materials in XML ---")
    for mat_name in scene.radio_materials.keys():
        print(f"- {mat_name}")

    print("\n--- Objects (buildings/ground) and their current materials ---")
    for obj_name, obj in scene.objects.items():
        print(f"Object: {obj_name} -> Material: {obj.radio_material.name}")


def diffusion_materials_are_assigned(scene, expected_material_names=DIFFUSION_MATERIAL_NAMES):
    """
    Checks whether ALL objects in the scene (except ground/"plane") have been
    assigned one of our custom scattering materials (`concrete_scattering`,
    `brick_scattering`, `glass_scattering`) – i.e., whether
    `assign_materials_to_scene()` (+ `fix_remaining_itu_concrete()`) has been
    called.

    Objects still on built‑in (XML) materials (e.g., `itu_concrete` with
    scattering_coefficient=0) will silently disable diffusion – this check
    provides an explicit, fast test before running the solver with
    `diffuse_reflection=True`.

    Returns
    -------
    tuple (bool, list[tuple[str, str]])
        (True, []) if all objects are ready;
        (False, [(object_name, current_material_name), ...]) otherwise.
    """
    missing = []
    for obj_name, obj in scene.objects.items():
        if "plane" in obj_name.lower():
            continue
        mat_name = obj.radio_material.name if obj.radio_material is not None else "None"
        if mat_name not in expected_material_names:
            missing.append((obj_name, mat_name))
    return (len(missing) == 0), missing


def ensure_diffusion_ready(scene):
    """
    Guard function – called automatically from `radio_map.compute_radio_map`
    and `path_solver.run_path_solver` when `diffuse_reflection=True`.
    Raises a clear, actionable error if not all objects have a scattering
    material, instead of silently returning results identical to those
    without diffusion (a common pitfall).

    Raises
    ------
    RuntimeError
        If any (non‑"plane") object in the scene lacks a custom scattering material.
    """
    ok, missing = diffusion_materials_are_assigned(scene)
    if not ok:
        preview = ", ".join(f"{name} ({mat})" for name, mat in missing[:5])
        more = f", ... (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise RuntimeError(
            "Diffuse scattering requested (diffuse_reflection=True), but "
            f"{len(missing)} object(s) in the scene still use the default "
            f"(XML) radio material with no scattering coefficient: {preview}{more}.\n"
            "Before enabling diffusion, call (in order):\n"
            "  1) materials.define_radio_materials()\n"
            "  2) materials.assign_materials_to_scene(scene, groups, materials)\n"
            "  3) materials.fix_remaining_itu_concrete(scene, materials['concrete'])\n"
            "(see run_config.py -> DiffusionConfig note, or main.py step 8)."
        )
    print("[OK] Radio materials for diffuse scattering are correctly assigned - diffusion is ready.")


def verify_material_assignment(scene):
    """
    Counts how many objects received which material (and whether any object
    is left without a material). 
    """
    print("Materials in scene:")
    for name, obj in scene.objects.items():
        if "plane" not in name.lower():
            print(f"  {name}: {obj.radio_material.name}")

    material_counts = {}
    for obj_name, obj in scene.objects.items():
        mat_name = obj.radio_material.name if obj.radio_material is not None else "None"
        material_counts[mat_name] = material_counts.get(mat_name, 0) + 1

    print("\n--- Assignment check ---")
    for mat_name, count in material_counts.items():
        print(f"  {mat_name}: {count} objects")
    print(f"\nTotal objects in scene: {len(scene.objects)}")

    return material_counts
