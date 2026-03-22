# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

import bpy
import bmesh
import time

from .idtech3lib.ID3VFS import Q3VFS
from .idtech3lib.BSP import BSP_READER as BSP
from .idtech3lib import GamePacks, MAP
from .idtech3lib.ID3Model import ID3Model as MODEL
from .idtech3lib.ImportSettings import Surface_Type
from math import atan, radians
from numpy import dot, sqrt
from mathutils import Vector

from . import BlenderImage, QuakeShader, QuakeLight
from . import MD3, TIKI

# import cProfile


class ImportLogger:
    """Kompaktes Logging-System für BSP-Import mit Zeitmessung"""

    def __init__(self):
        self.start_time = time.time()
        self.task_times = {}
        self.current_task = None
        self.task_start = None

    def start_task(self, task_name):
        """Startet eine neue Aufgabe und loggt den Beginn"""
        if self.current_task:
            self.end_task()

        self.current_task = task_name
        self.task_start = time.time()
        print(f"🔄 {task_name}...")

    def end_task(self):
        """Beendet die aktuelle Aufgabe und loggt die Zeit"""
        if self.current_task and self.task_start:
            elapsed = time.time() - self.task_start
            self.task_times[self.current_task] = elapsed
            print(f"✅ {self.current_task} completed in {elapsed:.2f}s")
            self.current_task = None
            self.task_start = None

    def log_loop_start(self, loop_name, count):
        """Loggt den Start einer Schleife"""
        print(f"🔄 {loop_name} ({count} items)...")

    def log_loop_end(self, loop_name, processed_count=None):
        """Loggt das Ende einer Schleife"""
        if processed_count is not None:
            print(f"✅ {loop_name} completed ({processed_count} items processed)")
        else:
            print(f"✅ {loop_name} completed")

    def log_progress(self, message, count=None):
        """Loggt Fortschritt ohne Zeitmessung"""
        if count is not None:
            print(f"📊 {message} ({count} items)")
        else:
            print(f"📊 {message}")

    def log_summary(self):
        """Loggt eine Zusammenfassung aller Zeiten"""
        total_time = time.time() - self.start_time
        print(f"\n📋 Import Summary (Total: {total_time:.2f}s):")
        for task, elapsed in self.task_times.items():
            percentage = (elapsed / total_time) * 100
            print(f"  • {task}: {elapsed:.2f}s ({percentage:.1f}%)")
        print(f"  • Other: {total_time - sum(self.task_times.values()):.2f}s")


def create_meshes_from_models(models):
    if models is None:
        return None
    return_meshes = {}
    for model in models:
        if model is None:
            continue
        name = model.name
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(model.positions.get_indexed(), [], model.indices)
        for texture_instance in model.material_names:
            mat = bpy.data.materials.get(texture_instance)
            if mat is None:
                mat = bpy.data.materials.new(name=texture_instance)
            mesh.materials.append(mat)
        mesh.polygons.foreach_set("material_index", model.material_id)

        if bpy.app.version < (4, 1, 0):
            mesh.use_auto_smooth = True

        for poly, smooth in zip(mesh.polygons, model.face_smooth):
            poly.use_smooth = smooth
        
        # Validate mesh before setting custom normals
        mesh.validate()
        mesh.update()
        
        unindexed_normals = model.vertex_normals.get_unindexed()
        if unindexed_normals is not None and len(unindexed_normals) > 0:
            # Convert normals to flat array format expected by Blender
            # normals_split_custom_set expects a flat list of floats: [x, y, z, x, y, z, ...]
            flat_normals = []
            for normal in unindexed_normals:
                if normal is not None:
                    # Handle both tuple/list and single value cases
                    if isinstance(normal, (list, tuple)) and len(normal) >= 3:
                        try:
                            # Extract and validate normal values (no NaN or inf)
                            nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
                            
                            # Check for invalid values
                            if (nx != nx or ny != ny or nz != nz or  # NaN check
                                abs(nx) == float('inf') or abs(ny) == float('inf') or abs(nz) == float('inf')):
                                # Invalid normal, use default (0, 0, 1)
                                flat_normals.append(0.0)
                                flat_normals.append(0.0)
                                flat_normals.append(1.0)
                            else:
                                # Normalize the normal vector
                                normal_vec = Vector((nx, ny, nz))
                                length = normal_vec.length
                                if length > 0.0001:  # Avoid division by zero
                                    normal_vec.normalize()
                                    flat_normals.append(float(normal_vec.x))
                                    flat_normals.append(float(normal_vec.y))
                                    flat_normals.append(float(normal_vec.z))
                                else:
                                    # Zero-length normal, use default
                                    flat_normals.append(0.0)
                                    flat_normals.append(0.0)
                                    flat_normals.append(1.0)
                        except (ValueError, TypeError, IndexError):
                            # Conversion error, use default
                            flat_normals.append(0.0)
                            flat_normals.append(0.0)
                            flat_normals.append(1.0)
                    elif isinstance(normal, (int, float)):
                        # Single float value - this shouldn't happen, but handle it
                        flat_normals.append(0.0)
                        flat_normals.append(0.0)
                        flat_normals.append(1.0)
                    else:
                        # Invalid format, use default
                        flat_normals.append(0.0)
                        flat_normals.append(0.0)
                        flat_normals.append(1.0)
                else:
                    # None value, use default
                    flat_normals.append(0.0)
                    flat_normals.append(0.0)
                    flat_normals.append(1.0)
            
            # Only set normals if the count matches the number of loops
            num_loops = len(mesh.loops)
            expected_count = num_loops * 3
            if len(flat_normals) == expected_count:
                try:
                    # Ensure we have a proper list of floats (not numpy array or other types)
                    normals_list = [float(v) for v in flat_normals]
                    mesh.normals_split_custom_set(normals_list)
                except Exception:
                    # TODO: ANATOLI hier muss ich was machen beim splitten der normals
                    # Continue without custom normals
                    pass
            # else: Normal count mismatch - skip setting normals

        for uv_layer in model.uv_layers:
            uvs = []
            for uv in model.uv_layers[uv_layer].get_unindexed(tuple):
                uvs.append(uv[0])
                uvs.append(uv[1])
            mesh.uv_layers.new(do_init=False, name=uv_layer)
            mesh.uv_layers[uv_layer].data.foreach_set("uv", uvs)
        for vert_color in model.vertex_colors:
            colors = []
            for color in model.vertex_colors[vert_color].get_unindexed():
                colors.append(color[0] / 255.0)
                colors.append(color[1] / 255.0)
                colors.append(color[2] / 255.0)
                colors.append(color[3] / 255.0)
            mesh.vertex_colors.new(name=vert_color)
            mesh.vertex_colors[vert_color].data.foreach_set("color", colors)
        if bpy.app.version >= (2, 92, 0):
            for vert_att in model.vertex_data_layers:
                try:
                    vert_data = model.vertex_data_layers[vert_att].get_indexed(int)
                    # Ensure data is a flat list
                    if vert_data and isinstance(vert_data, list):
                        # Flatten if nested
                        flat_data = []
                        for item in vert_data:
                            if isinstance(item, (list, tuple)):
                                flat_data.extend(item)
                            else:
                                flat_data.append(int(item))
                        # Validate count matches number of vertices
                        num_verts = len(mesh.vertices)
                        if len(flat_data) == num_verts:
                            mesh.attributes.new(name=vert_att, type="INT", domain="POINT")
                            mesh.attributes[vert_att].data.foreach_set("value", flat_data)
                except Exception:
                    pass
            
            for face_att in model.face_data_layers:
                try:
                    face_data = model.face_data_layers[face_att]
                    # Ensure data is a flat list
                    if face_data and isinstance(face_data, list):
                        # Flatten if nested
                        flat_data = []
                        for item in face_data:
                            if isinstance(item, (list, tuple)):
                                flat_data.extend(item)
                            else:
                                flat_data.append(int(item))
                        # Validate count matches number of faces
                        num_faces = len(mesh.polygons)
                        if len(flat_data) == num_faces:
                            mesh.attributes.new(name=face_att, type="INT", domain="FACE")
                            mesh.attributes[face_att].data.foreach_set("value", flat_data)
                except Exception:
                    pass
        elif bpy.app.version >= (2, 91, 0):
            for vert_att in model.vertex_data_layers:
                try:
                    vert_data = model.vertex_data_layers[vert_att].get_indexed(int)
                    # Ensure data is a flat list
                    if vert_data and isinstance(vert_data, list):
                        # Flatten if nested
                        flat_data = []
                        for item in vert_data:
                            if isinstance(item, (list, tuple)):
                                flat_data.extend(item)
                            else:
                                flat_data.append(int(item))
                        # Validate count matches number of vertices
                        num_verts = len(mesh.vertices)
                        if len(flat_data) == num_verts:
                            mesh.attributes.new(name=vert_att, type="INT", domain="VERTEX")
                            mesh.attributes[vert_att].data.foreach_set("value", flat_data)
                except Exception:
                    pass
            
            for face_att in model.face_data_layers:
                try:
                    face_data = model.face_data_layers[face_att]
                    # Ensure data is a flat list
                    if face_data and isinstance(face_data, list):
                        # Flatten if nested
                        flat_data = []
                        for item in face_data:
                            if isinstance(item, (list, tuple)):
                                flat_data.extend(item)
                            else:
                                flat_data.append(int(item))
                        # Validate count matches number of faces
                        num_faces = len(mesh.polygons)
                        if len(flat_data) == num_faces:
                            mesh.attributes.new(name=face_att, type="INT", domain="POLYGON")
                            mesh.attributes[face_att].data.foreach_set("value", flat_data)
                except Exception:
                    pass
        else:
            for vert_att in model.vertex_data_layers:
                try:
                    vert_data = model.vertex_data_layers[vert_att].get_indexed(int)
                    # Ensure data is a flat list
                    if vert_data and isinstance(vert_data, list):
                        # Flatten if nested
                        flat_data = []
                        for item in vert_data:
                            if isinstance(item, (list, tuple)):
                                flat_data.extend(item)
                            else:
                                flat_data.append(int(item))
                        # Validate count matches number of vertices
                        num_verts = len(mesh.vertices)
                        if len(flat_data) == num_verts:
                            mesh.vertex_layers_int.new(name=vert_att)
                            mesh.vertex_layers_int[vert_att].data.foreach_set("value", flat_data)
                except Exception:
                    pass

        mesh.validate()
        mesh.update()

        if name in return_meshes:
            print("Double mesh name found! Mesh did not get added: " + name)
            continue
        return_meshes[name] = mesh, model.vertex_groups
    if len(return_meshes) > 0:
        return return_meshes
    return None


def load_mesh(VFS, mesh_name, zoffset, bsp):
    blender_mesh = None
    vertex_groups = {}
    if mesh_name.endswith(".md3"):
        try:
            blender_mesh = MD3.ImportMD3(VFS, mesh_name, zoffset)[0]
        except Exception as e:
            print("Could not get model for mesh ", mesh_name, e)
            return blender_mesh, vertex_groups
    elif mesh_name.endswith(".tik"):
        try:
            blender_mesh = TIKI.ImportTIK(VFS, "models/{}".format(mesh_name), zoffset)[
                0
            ]
        except Exception as e:
            print("Could not get model for mesh ", mesh_name, e)
            return blender_mesh, vertex_groups
    elif mesh_name.startswith("*") and bsp is not None:
        model_id = None

        mesh = bpy.data.meshes.get(mesh_name)
        if mesh != None:
            mesh.name = mesh_name + "_prev.000"

        try:
            model_id = int(mesh_name[1:])
            new_blender_mesh = create_meshes_from_models([bsp.get_bsp_model(model_id)])
            blender_mesh, vertex_groups = next(iter(new_blender_mesh.values()))
        except Exception as e:
            print("Could not get model for mesh ", mesh_name, e)
            return blender_mesh, vertex_groups
    elif mesh_name == "box":
        blender_mesh = bpy.data.meshes.get("box")
        if blender_mesh is None:
            ent_object = bpy.ops.mesh.primitive_cube_add(size=8.0, location=([0, 0, 0]))
            ent_object = bpy.context.object
            ent_object.name = "box"

            material_name = "Object Color"
            mat = bpy.data.materials.get(material_name)
            if mat == None:
                mat = bpy.data.materials.new(name=material_name)
                mat.use_nodes = True
                node = mat.node_tree.nodes["Principled BSDF"]
                object_node = mat.node_tree.nodes.new(type="ShaderNodeObjectInfo")
                object_node.location = (node.location[0] - 400, node.location[1])
                mat.node_tree.links.new(
                    object_node.outputs["Color"], node.inputs["Base Color"]
                )
                if bpy.app.version >= (4, 0, 0):
                    mat.node_tree.links.new(
                        object_node.outputs["Color"], node.inputs["Emission Color"]
                    )
                    node.inputs["Emission Strength"].default_value = 1.0
                else:
                    mat.node_tree.links.new(
                        object_node.outputs["Color"], node.inputs["Emission"]
                    )
            ent_object.data.materials.append(mat)

            blender_mesh = ent_object.data
            blender_mesh.name = "box"
            bpy.data.objects.remove(ent_object, do_unlink=True)

    return blender_mesh, vertex_groups


def load_map_entity_surfaces(VFS, obj, import_settings):
    materials = []
    surfaces = obj.custom_parameters.get("surfaces")
    if surfaces is None:
        return None, None
    for surf in surfaces:
        if surf.type == "BRUSH":
            for plane in surf.planes:
                mat = plane.material
                if mat not in materials:
                    materials.append(mat)
    material_sizes = QuakeShader.get_shader_image_sizes(VFS, import_settings, materials)
    new_blender_mesh = create_meshes_from_models(
        [MAP.get_entity_brushes(obj, material_sizes, import_settings)]
    )
    if new_blender_mesh is None:
        return None, None
    blender_mesh, vertex_groups = next(iter(new_blender_mesh.values()))
    return blender_mesh, vertex_groups


def set_custom_properties(import_settings, blender_obj, bsp_obj):
    if bpy.app.version < (3, 0, 0):
        # needed for custom descriptions and data types prior 3.0
        rna_ui = blender_obj.get("_RNA_UI")
        if rna_ui is None:
            blender_obj["_RNA_UI"] = {}
            rna_ui = blender_obj["_RNA_UI"]

    class_dict_keys = {}
    class_model_forced = False
    classname = bsp_obj.custom_parameters.get("classname").lower()
    if classname in import_settings.entity_dict:
        class_dict_keys = import_settings.entity_dict[classname]["Keys"]
        if "Color" in import_settings.entity_dict[classname]:
            color_info = [*import_settings.entity_dict[classname]["Color"], 1.0]
            blender_obj.color = (
                pow(color_info[0], 2.2),
                pow(color_info[1], 2.2),
                pow(color_info[2], 2.2),
                pow(color_info[3], 2.2),
            )
        if "Model" in import_settings.entity_dict[classname]:
            class_model_forced = (
                import_settings.entity_dict[classname]["Model"] != "box"
            )
        if (
            bsp_obj.mesh_name == "box"
            and import_settings.entity_dict[classname]["Model"] == "box"
        ):
            maxs = import_settings.entity_dict[classname]["Maxs"]
            mins = import_settings.entity_dict[classname]["Mins"]
            blender_obj.delta_scale[0] = (maxs[0] - mins[0]) / 8.0
            if blender_obj.delta_scale[0] == 0:
                blender_obj.delta_scale[0] = 1.0
            else:
                blender_obj.delta_scale[1] = (maxs[1] - mins[1]) / 8.0
                blender_obj.delta_scale[2] = (maxs[2] - mins[2]) / 8.0
                blender_obj.delta_location[0] = (maxs[0] + mins[0]) * 0.5
                blender_obj.delta_location[1] = (maxs[1] + mins[1]) * 0.5
                blender_obj.delta_location[2] = (maxs[2] + mins[2]) * 0.5

    skip_properties = ("surfaces", "first_line")
    for property in bsp_obj.custom_parameters:
        if property in skip_properties:
            continue
        blender_obj[property] = bsp_obj.custom_parameters[property]

        if property not in class_dict_keys:
            continue
        property_dict = class_dict_keys[property]
        property_subtype = "NONE"
        property_desc = ""
        key_type = "STRING"
        if "Description" in property_dict:
            property_desc = property_dict["Description"]
        if "Type" in property_dict:
            key_type = property_dict["Type"]
            if key_type in GamePacks.TYPE_MATCHING:
                property_subtype = GamePacks.TYPE_MATCHING[key_type]
        if bpy.app.version < (3, 0, 0):
            descr_dict = {}
            descr_dict["description"] = property_desc
            if property_subtype != "NONE":
                descr_dict["subtype"] = property_subtype
            rna_ui[property] = descr_dict
        else:
            id_props = blender_obj.id_properties_ui(property)
            if id_props is None:
                continue
            if property_desc != "":
                id_props.update(description=str(property_desc))
            if property_subtype != "NONE":
                id_props.update(subtype=property_subtype)

    spawnflag = bsp_obj.spawnflags
    if spawnflag % 2 == 1:
        blender_obj.q3_dynamic_props.b1 = True
    if spawnflag & 2 > 1:
        blender_obj.q3_dynamic_props.b2 = True
    if spawnflag & 4 > 1:
        blender_obj.q3_dynamic_props.b4 = True
    if spawnflag & 8 > 1:
        blender_obj.q3_dynamic_props.b8 = True
    if spawnflag & 16 > 1:
        blender_obj.q3_dynamic_props.b16 = True
    if spawnflag & 32 > 1:
        blender_obj.q3_dynamic_props.b32 = True
    if spawnflag & 64 > 1:
        blender_obj.q3_dynamic_props.b64 = True
    if spawnflag & 128 > 1:
        blender_obj.q3_dynamic_props.b128 = True
    if spawnflag & 256 > 1:
        blender_obj.q3_dynamic_props.b256 = True
    if spawnflag & 512 > 1:
        blender_obj.q3_dynamic_props.b512 = True

    if (
        bsp_obj.mesh_name is not None
        and bsp_obj.mesh_name != "box"
        and not class_model_forced
    ):
        blender_obj.q3_dynamic_props.model = bsp_obj.mesh_name
        blender_obj["model"] = bsp_obj.mesh_name
    if bsp_obj.model2 != "":
        blender_obj.q3_dynamic_props.model2 = bsp_obj.model2
        blender_obj["model2"] = bsp_obj.model2


def add_light_drivers(light):
    light_data = light.data
    light_value = light.get("light")
    scale_value = light.get("scale")
    color_value = light.get("_color")

    if light_value is None and color_value is None:
        return

    if light_value is not None:
        driver = light.data.driver_add("energy")

        new_var = driver.driver.variables.get("light")
        if new_var is None:
            new_var = driver.driver.variables.new()
            new_var.name = "light"
        new_var.type = "SINGLE_PROP"
        new_var.targets[0].id = light
        new_var.targets[0].data_path = '["light"]'
        if scale_value != None:
            new_var = driver.driver.variables.get("scale")
            if new_var is None:
                new_var = driver.driver.variables.new()
                new_var.name = "scale"
            new_var.type = "SINGLE_PROP"
            new_var.targets[0].id = light
            new_var.targets[0].data_path = '["scale"]'

        light_type_scale = {
            "SUN": 0.1,
            "SPOT": 750.0,
            "POINT": 750.0,
            "AREA": 0.1,  # should not be used by the addon
        }

        if scale_value != None:
            expression = "float(light) * float(scale) * {}"
        else:
            expression = "float(light) * {}"

        driver.driver.expression = expression.format(light_type_scale[light_data.type])

    if color_value is not None:
        driver = light.data.driver_add("color")

        new_var = driver[0].driver.variables.get("color")
        if new_var is None:
            new_var = driver[0].driver.variables.new()
            new_var.name = "color"
        new_var.type = "SINGLE_PROP"
        new_var.targets[0].id = light
        new_var.targets[0].data_path = '["_color"]'
        driver[0].driver.expression = "color[0]"

        new_var = driver[1].driver.variables.get("color")
        if new_var is None:
            new_var = driver[1].driver.variables.new()
            new_var.name = "color"
        new_var.type = "SINGLE_PROP"
        new_var.targets[0].id = light
        new_var.targets[0].data_path = '["_color"]'
        driver[1].driver.expression = "color[1]"

        new_var = driver[2].driver.variables.get("color")
        if new_var is None:
            new_var = driver[2].driver.variables.new()
            new_var.name = "color"
        new_var.type = "SINGLE_PROP"
        new_var.targets[0].id = light
        new_var.targets[0].data_path = '["_color"]'
        driver[2].driver.expression = "color[2]"


def create_blender_light(import_settings, bsp_object, objects):
    intensity = 300.0
    color = [1.0, 1.0, 1.0]
    vector = [0.0, 0.0, -1.0]
    angle = 3.141592 / 2.0
    properties = bsp_object.custom_parameters
    if "light" in properties:
        intensity = float(properties["light"])
    if "scale" in properties:
        intensity *= float(properties["scale"])
    if "_color" in properties:
        color = properties["_color"]
    if "target" in properties:
        if properties["target"] in objects:
            target_origin = objects[properties["target"]].position
            vector = bsp_object.position - target_origin
            sqr_length = dot(vector, vector)
            if sqr_length == 0.0:
                sqr_length = 1.0
            radius = 64.0
            if "radius" in properties:
                radius = float(properties["radius"])
            angle = 2 * (atan(radius / sqrt(sqr_length)))
        if "_sun" in properties and properties["_sun"] == 1:
            light = QuakeLight.add_light(
                bsp_object.name, "SUN", intensity, color, vector, radians(1.5)
            )
        else:
            light = QuakeLight.add_light(
                bsp_object.name, "SPOT", intensity, color, vector, angle
            )
    else:
        light = QuakeLight.add_light(
            bsp_object.name, "POINT", intensity, color, vector, angle
        )
    light.location = bsp_object.position

    set_custom_properties(import_settings, light, bsp_object)
    # Add driver for the blender light intensity based on the entity data
    add_light_drivers(light)
    return light


def create_light_marker(import_settings, bsp_object, objects):
    """Creates an empty marker object for a light entity instead of a real
    Blender light.  All entity key/value pairs are stored as custom
    properties so they survive FBX export and can be read in Unity."""
    properties = bsp_object.custom_parameters

    # Determine the light type string for Unity
    light_type = "Point"
    if "target" in properties:
        if "_sun" in properties and properties["_sun"] == 1:
            light_type = "Sun"
        else:
            light_type = "Spot"

    marker = bpy.data.objects.new(bsp_object.name, None)
    marker.empty_display_type = 'SPHERE'
    marker.empty_display_size = 0.25
    marker.location = bsp_object.position

    bpy.context.collection.objects.link(marker)

    set_custom_properties(import_settings, marker, bsp_object)
    marker["light_type"] = light_type

    return marker


def is_object_valid_for_preset(bsp_object, import_settings):
    # import every entity in editing preset
    preset = import_settings.preset
    if preset == "EDITING":
        return True

    classname = bsp_object.custom_parameters.get("classname")
    mesh_name = bsp_object.mesh_name

    if classname is not None:
        if preset == "ONLY_LIGHTS":
            return classname == "light"
        if preset == "RENDERING" and classname == "light":
            return True
        if preset == "RENDERING" and classname == "misc_model":
            return False
        if preset == "UNITY" and classname == "light":
            return import_settings.import_lights

    class_dict = {}
    if classname in import_settings.entity_dict:
        class_dict = import_settings.entity_dict[classname]

    if "Model" in class_dict and mesh_name is None:
        mesh_name = class_dict["Model"]
    elif mesh_name is None:
        mesh_name = "box"

    if mesh_name == "box":
        return bsp_object.model2 != ""

    if classname is None:
        return False

    static_property = bsp_object.custom_parameters.get("make_static")
    if static_property is not None and preset == "RENDERING":
        return static_property == 0

    return True


def create_custom_worldspawn_collection(main_collection):
    """Erstellt eine custom_worldspawn Collection für aufgeteilte Worldspawn-Objekte"""
    col_name = "{}_worldspawn".format(main_collection.name)
    worldspawn_collection = bpy.data.collections.get(col_name)
    if worldspawn_collection is None:
        worldspawn_collection = bpy.data.collections.new(name=col_name)
        main_collection.children.link(worldspawn_collection)
    return worldspawn_collection


def create_object_type_collection(main_collection, object_type):
    """Erstellt eine Collection für einen bestimmten Objekttyp"""
    col_name = "{}_{}".format(main_collection.name, object_type)
    type_collection = bpy.data.collections.get(col_name)
    if type_collection is None:
        type_collection = bpy.data.collections.new(name=col_name)
        main_collection.children.link(type_collection)
    return type_collection


def cleanup_bsp_mesh(mesh, merge_threshold=0.001):
    """Cleans up a BSP brush mesh: merges duplicate vertices, removes degenerate faces and loose geometry."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_threshold)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=merge_threshold)
    # Delete loose verts and edges
    loose_verts = [v for v in bm.verts if not v.link_faces]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')
    loose_edges = [e for e in bm.edges if not e.link_faces]
    if loose_edges:
        bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def split_object_into_individual_surfaces(
    obj, bsp, import_settings, parent_collection, object_type="worldspawn"
):
    """Teilt ein BSP-Objekt in einzelne Surfaces auf - jede Surface wird ein separates Objekt in Material-Gruppen"""
    if obj.mesh_name is None or not obj.mesh_name.startswith("*"):
        return []

    # Extrahiere Model-ID aus dem mesh_name (z.B. "*0" -> 0)
    try:
        model_id = int(obj.mesh_name[1:])
    except (ValueError, IndexError):
        print("Could not extract model ID from mesh_name: {}".format(obj.mesh_name))
        return []

    # Hole das BSP-Model
    bsp_model = bsp.lumps["models"][model_id]
    first_face = bsp_model.face
    n_faces = bsp_model.n_faces

    print(
        f"  🔄 Splitting {object_type} into {n_faces} individual surfaces... from {obj.name}"
    )
    surface_start_time = time.time()

    # Gruppiere Surfaces nach Material für bessere Organisation
    surface_groups = {}
    shader_names = {}

    for i in range(n_faces):
        face_id = first_face + i
        if face_id >= len(bsp.lumps["surfaces"]):
            continue

        face = bsp.lumps["surfaces"][face_id]
        texture_id = face.texture

        # Cache Shader-Namen
        if texture_id not in shader_names:
            if texture_id >= 0 and texture_id < len(bsp.lumps["shaders"]):
                shader_name = bsp.lumps["shaders"][texture_id].name.decode("latin-1")
            else:
                shader_name = "unknown_material_{}".format(texture_id)
            shader_names[texture_id] = shader_name
        else:
            shader_name = shader_names[texture_id]

        if shader_name not in surface_groups:
            surface_groups[shader_name] = []
        surface_groups[shader_name].append(face_id)

    individual_objects = []
    processed_count = 0
    material_groups_processed = 0

    # Erstelle Collections für jede Material-Gruppe
    material_collections = {}

    print(f"  📊 Grouped into {len(surface_groups)} material groups from {obj.name}")

    # PROFILING: Zeitmessung pro Operation
    _t_model_build = 0.0
    _t_mesh_create = 0.0
    _t_obj_create = 0.0
    _t_props = 0.0
    _t_link = 0.0
    _t_collection = 0.0

    merge_by_material = getattr(import_settings, "merge_surfaces_by_material", False)

    # Erstelle ein separates Objekt für jede Surface oder ein Objekt pro Material-Gruppe
    for material_name, face_ids in surface_groups.items():
        material_groups_processed += 1
        if material_groups_processed % 10 == 0:
            elapsed = time.time() - surface_start_time
            print(
                f"    Processing material group {material_groups_processed}/{len(surface_groups)} ({elapsed:.1f}s elapsed)..."
            )
            print(
                f"      ⏱ model_build={_t_model_build:.1f}s mesh_create={_t_mesh_create:.1f}s obj_create={_t_obj_create:.1f}s props={_t_props:.1f}s link={_t_link:.1f}s collection={_t_collection:.1f}s"
            )

        # Erstelle Collection für diese Material-Gruppe
        _tc = time.time()
        safe_material_name = (
            material_name.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(".", "_")
        )
        group_collection_name = "group_{}".format(safe_material_name)

        group_collection = bpy.data.collections.get(group_collection_name)
        if group_collection is None:
            group_collection = bpy.data.collections.new(name=group_collection_name)
            parent_collection.children.link(group_collection)

        material_collections[material_name] = group_collection
        _t_collection += time.time() - _tc

        if merge_by_material:
            _t0 = time.time()
            model = MODEL("*{}_material_{}".format(model_id, safe_material_name))
            model.init_bsp_face_data_single(bsp, import_settings)

            for face_id in face_ids:
                face = bsp.lumps["surfaces"][face_id]
                surface_type = Surface_Type.bsp_value(face.type)

                if surface_type in (
                    Surface_Type.PLANAR,
                    Surface_Type.TRISOUP,
                    Surface_Type.FAKK_TERRAIN,
                ):
                    model.add_bsp_surface(bsp, face, import_settings)
                elif surface_type == Surface_Type.PATCH:
                    model.add_bsp_patch(bsp, face, import_settings)

            _t_model_build += time.time() - _t0

            if model.current_index > 0:
                _t1 = time.time()
                blender_meshes = create_meshes_from_models([model])
                _t_mesh_create += time.time() - _t1

                if blender_meshes:
                    for mesh_name, (mesh, vertex_groups) in blender_meshes.items():
                        if mesh is None:
                            continue

                        _t2 = time.time()
                        safe_object_type = object_type.replace("_", "")
                        obj_name = "{}_{}_group".format(
                            safe_object_type, safe_material_name
                        )
                        blender_obj = bpy.data.objects.new(obj_name, mesh)

                        blender_obj.location = obj.position
                        blender_obj.rotation_euler = obj.rotation
                        blender_obj.scale = obj.scale

                        for vert_group in vertex_groups:
                            vg = blender_obj.vertex_groups.get(vert_group)
                            if vg is None:
                                vg = blender_obj.vertex_groups.new(name=vert_group)
                            vg.add(list(vertex_groups[vert_group]), 1.0, "ADD")
                        _t_obj_create += time.time() - _t2

                        _t3 = time.time()
                        for prop_name, prop_value in obj.custom_parameters.items():
                            blender_obj[prop_name] = prop_value

                        #blender_obj["material_name"] = material_name
                        blender_obj["surface_id"] = -1
                        #blender_obj["texture_id"] = -1
                        blender_obj["surface_index"] = -1
                        #blender_obj["material_group_size"] = len(face_ids)
                        #blender_obj["object_type"] = object_type
                        #blender_obj["original_object"] = obj.name
                        _t_props += time.time() - _t3

                        if getattr(import_settings, "cleanup_brush_meshes", False):
                            cleanup_bsp_mesh(mesh)

                        _t4 = time.time()
                        group_collection = material_collections[material_name]
                        group_collection.objects.link(blender_obj)
                        _t_link += time.time() - _t4

                        individual_objects.append(blender_obj)
                        processed_count += 1
        else:
            for surface_index, face_id in enumerate(face_ids):
                # Erstelle ein neues Model für diese einzelne Surface
                _t0 = time.time()
                face = bsp.lumps["surfaces"][face_id]
                surface_type = Surface_Type.bsp_value(face.type)

                # Erstelle ein Model nur für diese eine Surface
                model = MODEL("*{}_surface_{}".format(model_id, face_id))
                model.init_bsp_face_data_single(bsp, import_settings)

                if surface_type in (
                    Surface_Type.PLANAR,
                    Surface_Type.TRISOUP,
                    Surface_Type.FAKK_TERRAIN,
                ):
                    model.add_bsp_surface(bsp, face, import_settings)
                elif surface_type == Surface_Type.PATCH:
                    model.add_bsp_patch(bsp, face, import_settings)
                _t_model_build += time.time() - _t0

                if model.current_index > 0:
                    _t1 = time.time()
                    blender_meshes = create_meshes_from_models([model])
                    _t_mesh_create += time.time() - _t1

                    if blender_meshes:
                        for mesh_name, (mesh, vertex_groups) in blender_meshes.items():
                            if mesh is None:
                                continue

                            _t2 = time.time()
                            # Erstelle Unity-kompatiblen Objektnamen für FBX-Export
                            safe_object_type = object_type.replace("_", "")
                            safe_material_name = (
                                material_name.replace("/", "_")
                                .replace("\\", "_")
                                .replace(" ", "_")
                                .replace(".", "_")
                            )
                            obj_name = "{}_{}_surface_{}".format(
                                safe_object_type, safe_material_name, face_id
                            )
                            blender_obj = bpy.data.objects.new(obj_name, mesh)

                            # Setze Position, Rotation und Skalierung
                            blender_obj.location = obj.position
                            blender_obj.rotation_euler = obj.rotation
                            blender_obj.scale = obj.scale

                            # Füge Vertex Groups hinzu
                            for vert_group in vertex_groups:
                                vg = blender_obj.vertex_groups.get(vert_group)
                                if vg is None:
                                    vg = blender_obj.vertex_groups.new(name=vert_group)
                                vg.add(list(vertex_groups[vert_group]), 1.0, "ADD")
                            _t_obj_create += time.time() - _t2

                            _t3 = time.time()
                            # Kopiere Custom Properties vom ursprünglichen Objekt
                            for prop_name, prop_value in obj.custom_parameters.items():
                                blender_obj[prop_name] = prop_value

                            # Füge Metadaten hinzu
                            #blender_obj["material_name"] = material_name
                            blender_obj["surface_id"] = face_id
                            #blender_obj["texture_id"] = face.texture
                            blender_obj["surface_index"] = surface_index
                            #blender_obj["material_group_size"] = len(face_ids)
                            #blender_obj["object_type"] = object_type
                            #blender_obj["original_object"] = obj.name
                            _t_props += time.time() - _t3

                            if getattr(import_settings, "cleanup_brush_meshes", False):
                                cleanup_bsp_mesh(mesh)

                            # Verlinke Objekt zur entsprechenden Material-Gruppen-Collection
                            _t4 = time.time()
                            group_collection = material_collections[material_name]
                            group_collection.objects.link(blender_obj)
                            _t_link += time.time() - _t4

                            individual_objects.append(blender_obj)
                            processed_count += 1

        # Fortschritts-Logging nur alle 500 Items
        if processed_count % 500 == 0 and processed_count > 0:
            elapsed = time.time() - surface_start_time
            print(
                f"    Processed {processed_count} surfaces ({elapsed:.1f}s elapsed)..."
            )

    # PROFILING: Finale Zusammenfassung
    print(f"  ⏱ PROFILING TOTAL:")
    print(f"    model_build:  {_t_model_build:.2f}s")
    print(f"    mesh_create:  {_t_mesh_create:.2f}s")
    print(f"    obj_create:   {_t_obj_create:.2f}s")
    print(f"    props:        {_t_props:.2f}s")
    print(f"    link:         {_t_link:.2f}s")
    print(f"    collection:   {_t_collection:.2f}s")

    # Detaillierte Zusammenfassung für Surface-Splitting
    elapsed = time.time() - surface_start_time
    print(f"  ✅ Surface splitting completed in {elapsed:.2f}s")
    print(f"  📊 Surface Statistics:")
    print(f"    • Total surfaces: {n_faces}")
    print(f"    • Material groups: {len(surface_groups)}")
    print(f"    • Individual objects created: {len(individual_objects)}")
    if n_faces > 0:
        print(f"    • Average time per surface: {elapsed / n_faces * 1000:.1f}ms")
    else:
        print(f"    • No surfaces to process")

    return individual_objects


def split_worldspawn_into_individual_surfaces(
    worldspawn_obj, bsp, import_settings, worldspawn_collection
):
    """Teilt das Worldspawn-Objekt in einzelne Surfaces auf - verwendet die generische Funktion"""
    return split_object_into_individual_surfaces(
        worldspawn_obj, bsp, import_settings, worldspawn_collection, "worldspawn"
    )


def create_blender_objects(VFS, import_settings, objects, meshes, bsp):
    if len(objects) <= 0:
        return None
    object_list = []

    # Detailliertes Logging für die Hauptschleife
    print(f"🔄 Processing {len(objects)} objects...")
    processed_count = 0
    skipped_invalid = 0
    skipped_no_mesh = 0
    skipped_mesh_not_found = 0
    bsp_objects_processed = 0
    regular_objects_processed = 0

    start_time = time.time()

    for obj_name in objects:
        obj = objects[obj_name]

        if not is_object_valid_for_preset(obj, import_settings):
            skipped_invalid += 1
            continue

        # Spezielle Behandlung für alle BSP-Objekte mit * mesh_name (Worldspawn und andere)
        if obj.mesh_name and obj.mesh_name.startswith("*"):
            bsp_objects_processed += 1

            # Bestimme Objekttyp für Collection-Organisation
            object_type = "worldspawn"
            if obj_name != "worldspawn":
                classname = obj.custom_parameters.get("classname")
                if classname == "light":
                    object_type = "lights"
                elif classname == "info_player_start":
                    object_type = "spawns"
                elif classname == "trigger":
                    object_type = "triggers"
                elif classname == "func":
                    object_type = "func"
                elif classname == "misc":
                    object_type = "misc"
                else:
                    object_type = "entities"

            # Erstelle Collection für diesen Objekttyp
            if object_type == "worldspawn":
                parent_collection = create_custom_worldspawn_collection(
                    bpy.context.collection
                )
            else:
                parent_collection = create_object_type_collection(
                    bpy.context.collection, object_type
                )

            # Teile Objekt in einzelne Surfaces auf
            individual_objects = split_object_into_individual_surfaces(
                obj, bsp, import_settings, parent_collection, object_type
            )

            # Objekte sind bereits in den entsprechenden Collections organisiert
            object_list.extend(individual_objects)
            processed_count += 1

            continue

        if obj.mesh_name is None:
            classname = obj.custom_parameters.get("classname")
            if classname is not None and classname == "light":
                if import_settings.import_lights:
                    if import_settings.preset == "UNITY":
                        marker = create_light_marker(
                            import_settings, obj, objects)
                        object_list.append(marker)
                    else:
                        create_blender_light(import_settings, obj, objects)
                processed_count += 1
                continue
            else:
                class_dict = {}
                if classname in import_settings.entity_dict:
                    class_dict = import_settings.entity_dict[classname]
                obj.mesh_name = "box"
                if "Model" in class_dict:
                    obj.mesh_name = class_dict["Model"]

        mesh_z_name = obj.mesh_name
        # TODO: Get rid of this stupid zoffset BS
        if mesh_z_name.endswith(".md3"):
            mesh_z_name = mesh_z_name[: -len(".md3")]
        if mesh_z_name.endswith(".tik"):
            mesh_z_name = mesh_z_name[: -len(".tik")]
        if obj.zoffset != 0:
            mesh_z_name = mesh_z_name + ".z{}".format(obj.zoffset)

        if meshes is None:
            skipped_no_mesh += 1
            continue

        vertex_groups = {}
        if mesh_z_name not in meshes:
            if bsp is None and obj.custom_parameters.get("surfaces") is not None:
                blender_mesh, vertex_groups = load_map_entity_surfaces(
                    VFS, obj, import_settings
                )
            else:
                blender_mesh, vertex_groups = load_mesh(
                    VFS, obj.mesh_name, obj.zoffset, bsp
                )
            if blender_mesh is not None:
                blender_mesh.name = mesh_z_name
                meshes[mesh_z_name] = blender_mesh
            elif obj.model2 != "":
                blender_mesh, vertex_groups = load_mesh(VFS, "box", 0, bsp)
        else:
            blender_mesh = meshes[mesh_z_name]

        if blender_mesh is None:
            skipped_mesh_not_found += 1
            continue

        blender_obj = bpy.data.objects.new(obj.name, blender_mesh)
        blender_obj.location = obj.position

        if not blender_mesh.name.startswith("*"):
            blender_obj.rotation_euler = obj.rotation
            if "Tiki_Scale" in blender_mesh:
                new_scale = (
                    blender_mesh["Tiki_Scale"],
                    blender_mesh["Tiki_Scale"],
                    blender_mesh["Tiki_Scale"],
                )
                blender_obj.scale = new_scale
            else:
                blender_obj.scale = obj.scale

        bpy.context.collection.objects.link(blender_obj)
        object_list.append(blender_obj)

        for vert_group in vertex_groups:
            vg = blender_obj.vertex_groups.get(vert_group)
            if vg is None:
                vg = blender_obj.vertex_groups.new(name=vert_group)
            vg.add(list(vertex_groups[vert_group]), 1.0, "ADD")

        set_custom_properties(import_settings, blender_obj, obj)
        regular_objects_processed += 1

        if "model2" not in blender_obj:
            processed_count += 1
            continue
        blender_mesh, vertex_groups = load_mesh(
            VFS, blender_obj["model2"], obj.zoffset, None
        )
        if blender_mesh is None:
            processed_count += 1
            continue
        m2_obj = bpy.data.objects.new(obj.name + "_model2", blender_mesh)
        bpy.context.collection.objects.link(m2_obj)
        object_list.append(m2_obj)
        m2_obj.parent = blender_obj
        m2_obj.hide_select = True
        if blender_obj.data.name == "box":
            blender_obj.hide_render = True

        processed_count += 1

        # Fortschritts-Logging nur alle 100 Items
        if processed_count % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Processed {processed_count}/{len(objects)} objects ({elapsed:.1f}s elapsed)..."
            )

    # Detaillierte Zusammenfassung
    elapsed = time.time() - start_time
    print(f"✅ Object processing completed in {elapsed:.2f}s")
    print(f"📊 Processing Statistics:")
    print(f"  • Total objects: {len(objects)}")
    print(f"  • Processed: {processed_count}")
    print(f"  • BSP objects (split into surfaces): {bsp_objects_processed}")
    print(f"  • Regular objects: {regular_objects_processed}")
    print(f"  • Skipped (invalid preset): {skipped_invalid}")
    print(f"  • Skipped (no mesh): {skipped_no_mesh}")
    print(f"  • Skipped (mesh not found): {skipped_mesh_not_found}")

    return object_list


def get_bsp_file(VFS, import_settings):
    return BSP(VFS, import_settings)


def set_blender_clip_spaces(clip_start, clip_end):
    for ws in bpy.data.workspaces:
        for screen in ws.screens:
            for a in screen.areas:
                if a.type == "VIEW_3D":
                    for s in a.spaces:
                        if s.type == "VIEW_3D":
                            s.clip_start = clip_start
                            s.clip_end = clip_end


def get_main_collection(bsp_file):
    map_name = bsp_file.map_name
    if map_name.startswith("maps/"):
        map_name = map_name[len("maps/") :]
    main_collection = bpy.data.collections.get(map_name)
    if main_collection is None:
        main_collection = bpy.data.collections.new(name=map_name)
        bpy.context.scene.collection.children.link(main_collection)
    layer_collection = bpy.context.view_layer.layer_collection.children.get(
        main_collection.name
    )
    if layer_collection:
        bpy.context.view_layer.active_layer_collection = layer_collection
    return main_collection


def sort_bsp_objects_into_collections(main_collection):
    main_layer = bpy.context.view_layer.layer_collection.children.get(
        main_collection.name
    )
    if not main_layer:
        print("Could not find the map collection in the scene collection")
        return
    for obj in main_collection.objects:
        if obj.name.startswith("worldspawn"):
            continue
        split_name = obj.name.split("_", 1)
        if len(split_name) == 1:
            continue
        etype, rest = split_name
        col_name = "{}_{}".format(main_collection.name, etype)
        ent_collection = bpy.data.collections.get(col_name)
        if ent_collection is None:
            ent_collection = bpy.data.collections.new(name=col_name)
            main_collection.children.link(ent_collection)
        for other_col in obj.users_collection:
            other_col.objects.unlink(obj)
        if obj.name not in ent_collection.objects:
            ent_collection.objects.link(obj)


def import_bsp_file(import_settings):
    logger = ImportLogger()

    logger.start_task("Initialize Virtual File System")
    VFS = Q3VFS()
    for base_path in import_settings.base_paths:
        VFS.add_base(base_path)
    VFS.build_index()
    logger.end_task()

    logger.start_task("Read BSP Data")
    bsp_file = BSP(VFS, import_settings)
    logger.end_task()

    logger.start_task("Setup Main Collection")
    main_collection = get_main_collection(bsp_file)
    logger.end_task()

    # create blender objects
    blender_objects = []
    bsp_objects = None
    BRUSH_IMPORTS = ["BRUSHES", "SHADOW_BRUSHES"]
    if import_settings.preset in BRUSH_IMPORTS:
        logger.start_task(f"Import {import_settings.preset}")

        col_name = "{}_{}".format(main_collection.name, import_settings.preset)
        brushes_collection = bpy.data.collections.get(col_name)
        if brushes_collection is None:
            brushes_collection = bpy.data.collections.new(name=col_name)
            main_collection.children.link(brushes_collection)
        layer_collection = bpy.context.view_layer.layer_collection.children.get(
            main_collection.name
        )
        if layer_collection:
            brush_layer = layer_collection.children.get(brushes_collection.name)
            if brush_layer:
                bpy.context.view_layer.active_layer_collection = brush_layer

        bsp_models = bsp_file.get_bsp_models()
        blender_meshes = create_meshes_from_models(bsp_models)

        logger.log_loop_start("Create Brush Objects", len(blender_meshes))
        for mesh_name in blender_meshes:
            mesh, vertex_groups = blender_meshes[mesh_name]
            if mesh is None:
                mesh = bpy.data.meshes.new(mesh_name)
            ob = bpy.data.objects.new(name=mesh_name, object_data=mesh)
            for vert_group in vertex_groups:
                vg = ob.vertex_groups.get(vert_group)
                if vg is None:
                    vg = ob.vertex_groups.new(name=vert_group)
                vg.add(list(vertex_groups[vert_group]), 1.0, "ADD")

            if import_settings.preset == "SHADOW_BRUSHES":
                modifier = ob.modifiers.new("Displace", type="DISPLACE")
                modifier.strength = -4.0
                ob.data.name = "SB{}".format(ob.data.name)
            blender_objects.append(ob)
            bpy.context.collection.objects.link(ob)
        logger.log_loop_end("Create Brush Objects", len(blender_meshes))

        bsp_node = bpy.data.node_groups.get("BspInfo")
        if bsp_node is None:
            bsp_file.get_bsp_images()
            QuakeShader.init_shader_system(bsp_file)

        QuakeShader.build_quake_shaders(VFS, import_settings, blender_objects)
        QuakeShader.prepare_textures_for_unity(VFS, import_settings, blender_objects)

        logger.end_task()
        logger.log_summary()
        return

    logger.start_task("Prepare Lightmap Packing")
    bsp_file.lightmap_size = bsp_file.compute_packed_lightmap_size()
    logger.end_task()

    logger.start_task("Get BSP Entity Objects")
    bsp_objects = bsp_file.get_bsp_entity_objects()
    logger.end_task()

    logger.start_task("Create Blender Objects")
    blender_objects = create_blender_objects(
        VFS,
        import_settings,
        bsp_objects,
        {},  # blender_meshes,
        bsp_file,
    )
    logger.end_task()

    logger.start_task("Organize Objects into Collections")
    sort_bsp_objects_into_collections(main_collection)
    logger.end_task()

    # Unity preset: import brushes as invisible collision geometry
    logger.start_task("Handle Fog Volumes")
    bsp_fogs = bsp_file.get_bsp_fogs()
    fog_meshes = create_meshes_from_models(bsp_fogs)
    if fog_meshes is not None:
        logger.log_loop_start("Create Fog Objects", len(fog_meshes))
        for mesh_name in fog_meshes:
            mesh, vertex_groups = fog_meshes[mesh_name]
            if mesh is None:
                mesh = bpy.data.meshes.new(mesh_name)
            ob = bpy.data.objects.new(name=mesh_name, object_data=mesh)
            # Give the volume a slight push so cycles doesnt z-fight...
            modifier = ob.modifiers.new("Displace", type="DISPLACE")
            blender_objects.append(ob)
            bpy.context.collection.objects.link(ob)
        logger.log_loop_end("Create Fog Objects", len(fog_meshes))
    logger.end_task()

    logger.start_task("Get Clip Data and Grid Size")
    clip_end = 40000
    if bsp_objects is not None and "worldspawn" in bsp_objects:
        worldspawn_object = bsp_objects["worldspawn"]
        custom_parameters = worldspawn_object.custom_parameters
        # if ("distancecull" in custom_parameters and
        #   import_settings.preset == "PREVIEW"):
        #    clip_end = float(custom_parameters["distancecull"])
        if "gridsize" in custom_parameters:
            grid_size = custom_parameters["gridsize"]
            bsp_file.lightgrid_size = grid_size
            bsp_file.lightgrid_inverse_size = [
                1.0 / float(grid_size[0]),
                1.0 / float(grid_size[1]),
                1.0 / float(grid_size[2]),
            ]
    logger.end_task()

    logger.start_task("Apply Clip Data")
    set_blender_clip_spaces(4.0, clip_end)
    logger.end_task()

    """
    logger.start_task("Get BSP Images")
    bsp_images = bsp_file.get_bsp_images()
    logger.log_loop_start("Process BSP Images", len(bsp_images))
    for image in bsp_images:
        old_image = bpy.data.images.get(image.name)
        if old_image != None:
            old_image.name = image.name + "_prev.000"
        try:
            new_image = bpy.data.images.new(
                image.name,
                width=image.width,
                height=image.height,
                alpha=image.num_components == 4,
            )
            new_image.pixels = image.get_rgba()
            new_image.alpha_mode = "CHANNEL_PACKED"
            new_image.use_fake_user = True
            new_image.pack()
        except Exception:
            print("Couldn't retreve image from bsp:", image.name)
    logger.log_loop_end("Process BSP Images", len(bsp_images))
    logger.end_task()
    """

    logger.start_task("Handle External Lightmaps")
    if bsp_file.num_internal_lm_ids >= 0 and bsp_file.external_lm_files:
        external_lm_lump = []
        width, height = None, None

        logger.log_loop_start(
            "Load External Lightmap Files", len(bsp_file.external_lm_files)
        )
        for file_name in bsp_file.external_lm_files:
            tmp_image = BlenderImage.load_file(file_name, VFS)
            if tmp_image is None:
                print("Could not load:", file_name)
                continue
            if not width:
                width = tmp_image.size[0]
            if not height:
                height = tmp_image.size[1]

            if width != tmp_image.size[0] or height != tmp_image.size[1]:
                print("External lightmaps all need to be the same size")
                break

            external_lm_lump.append(list(tmp_image.pixels[:]))
        logger.log_loop_end("Load External Lightmap Files", len(external_lm_lump))

        bsp_file.internal_lightmap_size = (width, height)
        bsp_file.lightmap_size = bsp_file.compute_packed_lightmap_size()

        atlas_pixels = bsp_file.pack_lightmap(
            external_lm_lump, bsp_file.deluxemapping, False, False, 4
        )

        new_image = bpy.data.images.new(
            "$lightmap",
            width=bsp_file.lightmap_size[0],
            height=bsp_file.lightmap_size[1],
        )
        new_image.pixels = atlas_pixels
        new_image.alpha_mode = "CHANNEL_PACKED"

        if bsp_file.deluxemapping:
            atlas_pixels = bsp_file.pack_lightmap(
                external_lm_lump, bsp_file.deluxemapping, True, False, 4
            )

            new_image = bpy.data.images.new(
                "$deluxemap",
                width=bsp_file.lightmap_size[0],
                height=bsp_file.lightmap_size[1],
            )
            new_image.pixels = atlas_pixels
            new_image.alpha_mode = "CHANNEL_PACKED"
    logger.end_task()

    logger.start_task("Setup Lightmap Properties")
    bpy.context.scene.id_tech_3_lightmaps_per_row = int(
        bsp_file.lightmap_size[0] / bsp_file.internal_lightmap_size[0]
    )
    bpy.context.scene.id_tech_3_lightmaps_per_column = int(
        bsp_file.lightmap_size[1] / bsp_file.internal_lightmap_size[1]
    )
    logger.end_task()

    logger.start_task("Initialize Shader System")
    QuakeShader.init_shader_system(bsp_file)
    logger.end_task()

    logger.start_task("Build Quake Shaders")
    QuakeShader.build_quake_shaders(VFS, import_settings, blender_objects)
    logger.end_task()

    logger.start_task("Prepare Textures for Unity")
    QuakeShader.prepare_textures_for_unity(VFS, import_settings, blender_objects)
    logger.end_task()

    # Unity preset: create collision geometry from compiled BSP faces
    if import_settings.preset == "UNITY":
        logger.start_task("Import Collision Geometry")
        col_name = "{}_collision".format(main_collection.name)
        collision_collection = bpy.data.collections.get(col_name)
        if collision_collection is None:
            collision_collection = bpy.data.collections.new(name=col_name)
            main_collection.children.link(collision_collection)

        # === HYBRID COLLISION: compiled faces + invisible raw brushes ===
        #
        # Compiled faces: walls, terrain, arches (correct openings), all models
        #   → skip models/ shaders (tree quads)
        # Raw brushes (*0 only): ONLY invisible clip/nodraw_solid brushes
        #   → these never appear in compiled faces
        #
        # No overlap, no double collision.

        SURF_NODRAW = 0x00200000
        CONTENTS_SOLID      = 0x00000001
        CONTENTS_PLAYERCLIP = 0x00000010
        CONTENTS_SHOTCLIP   = 0x00000080
        SOLID_MASK = CONTENTS_SOLID | CONTENTS_PLAYERCLIP | CONTENTS_SHOTCLIP

        # 1) Compiled faces for all models (*0, *1, *2, ...)
        bsp_surface_types = (
            Surface_Type.PLANAR,
            Surface_Type.TRISOUP,
            Surface_Type.FAKK_TERRAIN,
        )
        bsp_models = []
        for model_id in range(len(bsp_file.lumps["models"])):
            model = MODEL("*{}".format(model_id))
            model.init_bsp_face_data(bsp_file, import_settings)
            bsp_model = bsp_file.lumps["models"][model_id]
            first_face = bsp_model.face

            for i in range(bsp_model.n_faces):
                face_id = first_face + i
                face = bsp_file.lumps["surfaces"][face_id]

                shader_name = bsp_file.lumps["shaders"][face.texture].name.decode("latin-1")
                if shader_name.startswith("models/"):
                    continue

                surface_type = Surface_Type.bsp_value(face.type)
                if not bool(surface_type & import_settings.surface_types):
                    continue

                if surface_type in bsp_surface_types:
                    model.add_bsp_surface(bsp_file, face, import_settings)
                elif surface_type == Surface_Type.PATCH:
                    model.add_bsp_patch(bsp_file, face, import_settings)

            if model.current_index > 0:
                bsp_models.append(model)

        # 2) Raw brushes from *0: ONLY invisible collision (nodraw + solid)
        #    These are playerclip, shotclip, nodraw_solid — never in compiled faces.
        nodraw_solid_shaders = set()
        for shader in bsp_file.lumps["shaders"]:
            if (shader.flags & SURF_NODRAW) and (shader.contents & SOLID_MASK):
                nodraw_solid_shaders.add(shader.name.decode("latin-1"))

        if nodraw_solid_shaders:
            brush_model = MODEL("*0_clip")
            brush_model.add_bsp_model_brushes(bsp_file, 0, import_settings)
            if brush_model.current_index > 0:
                bsp_models.append(brush_model)

        blender_meshes = create_meshes_from_models(bsp_models)

        for mesh_name in blender_meshes:
            mesh, vertex_groups = blender_meshes[mesh_name]
            if mesh is None:
                mesh = bpy.data.meshes.new(mesh_name)

            # For *0_clip: keep ONLY nodraw+solid materials (clips, nodraw_solid)
            if mesh_name == "*0_clip" and len(mesh.materials) > 0:
                keep_indices = set()
                for i, mat in enumerate(mesh.materials):
                    if mat is not None:
                        base = mat.name[:-6] if mat.name.endswith(".brush") else mat.name
                        if base in nodraw_solid_shaders or mat.name in nodraw_solid_shaders:
                            keep_indices.add(i)
                remove_indices = {i for i in range(len(mesh.materials))} - keep_indices
                if remove_indices:
                    bm = bmesh.new()
                    bm.from_mesh(mesh)
                    bm.faces.ensure_lookup_table()
                    faces_to_delete = [f for f in bm.faces if f.material_index in remove_indices]
                    if faces_to_delete:
                        bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
                    bm.to_mesh(mesh)
                    bm.free()
                    mesh.update()

            if len(mesh.polygons) == 0:
                bpy.data.meshes.remove(mesh)
                continue

            # Clean up: merge duplicate verts, remove degenerates
            cleanup_bsp_mesh(mesh)

            # Triangulate for Unity (prevents self-intersecting N-gon warnings)
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()

            col_obj_name = "COL_{}".format(mesh_name)
            ob = bpy.data.objects.new(name=col_obj_name, object_data=mesh)
            mesh.materials.clear()
            ob.hide_render = True
            ob.display_type = 'WIRE'
            collision_collection.objects.link(ob)

        # Hide collision collection in viewport by default
        layer_collection = bpy.context.view_layer.layer_collection.children.get(
            main_collection.name
        )
        if layer_collection:
            col_layer = layer_collection.children.get(collision_collection.name)
            if col_layer:
                col_layer.exclude = False
                col_layer.hide_viewport = True
        logger.end_task()

    # Unity preset: bake scale directly into mesh vertices (inches to meters)
    if import_settings.preset == "UNITY":
        logger.start_task("Apply Unity Scale (0.0254)")
        UNITY_SCALE = 0.0254
        from mathutils import Matrix
        scale_matrix = Matrix.Scale(UNITY_SCALE, 4)
        scaled_meshes = set()
        for ob in main_collection.all_objects:
            # Bake scale into mesh data so FBX export is correct regardless of settings
            if ob.data and hasattr(ob.data, 'transform') and ob.data.name not in scaled_meshes:
                ob.data.transform(scale_matrix)
                ob.data.update()
                scaled_meshes.add(ob.data.name)
            # Scale location (origin offset)
            ob.location = (
                ob.location[0] * UNITY_SCALE,
                ob.location[1] * UNITY_SCALE,
                ob.location[2] * UNITY_SCALE,
            )
        logger.end_task()

    logger.log_summary()


def import_map_file(import_settings):
    # initialize virtual file system
    VFS = Q3VFS()
    for base_path in import_settings.base_paths:
        VFS.add_base(base_path)
    VFS.build_index()

    byte_array = VFS.get(import_settings.file)

    entities = MAP.read_map_file(byte_array, import_settings)
    objects = create_blender_objects(VFS, import_settings, entities, {}, None)

    set_blender_clip_spaces(4.0, 40000.0)

    QuakeShader.init_shader_system(None)
    QuakeShader.build_quake_shaders(VFS, import_settings, objects)
