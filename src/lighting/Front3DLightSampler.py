import sys
from enum import Enum
import numpy as np
from src.main.Module import Module
from src.utility.LightUtility import Light
from src.utility.BlenderUtility import get_all_blender_mesh_objects, get_bounds

import bpy


def deg_to_rad(deg):
    """
    Converts an angle in degrees to an angle in radians.
    """
    return deg / 180.0 * np.pi


# https://www.pveducation.org/pvcdrom/properties-of-sunlight/elevation-angle
phi = deg_to_rad(48)  # approximately latitude of Munich
# https://en.wikipedia.org/wiki/Position_of_the_Sun#Declination_of_the_Sun_as_seen_from_Earth
delta = deg_to_rad(23.44)  # summer solstice


def compute_elevation_angle(local_solar_time):
    """
    Converts the local solar time to the elevation angle of the sun.
    For more information see: https://www.pveducation.org/pvcdrom/properties-of-sunlight/elevation-angle
    :param local_solar_time: The local solar time.
    :return: The elevation angle of the sun.
    """
    # Explanation of hour angle: https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time#HRA
    hour_angle = deg_to_rad(15) * (local_solar_time - 12)
    alpha = np.arcsin(np.sin(delta) * np.sin(phi) + np.cos(delta) * np.cos(phi) * np.cos(hour_angle))
    return alpha


def compute_local_solar_time_pair(elevation_angle):
    """
    Computes the the local solar time for a given sun elevation angle.
    There are two times per day (AM and PM) with equal elevation angles, which are returned as a tuple.
    For more information see: https://www.pveducation.org/pvcdrom/properties-of-sunlight/elevation-angle
    :param elevation_angle: The elevation angle of the sun.
    :return: The pair of solar times with the specified elevation angle.
    """
    hour_angle = np.arccos((np.sin(elevation_angle) - np.sin(phi) * np.sin(delta)) / (np.cos(phi) * np.cos(delta)))
    t0 = 12 - 1 / deg_to_rad(15) * hour_angle
    t1 = 12 + 1 / deg_to_rad(15) * hour_angle
    return t0, t1


# Time pair for alpha = 6° and alpha = 8° (i.e., the sun elevation interval when red sky phenomenon most often occurs).
t_alpha6 = compute_local_solar_time_pair(deg_to_rad(6))
t_alpha8 = compute_local_solar_time_pair(deg_to_rad(8))

# Solar time events.
midnight = 0
sunrise = 12 - 1 / deg_to_rad(15) * np.arccos(-np.tan(phi) * np.tan(delta))
dawn_start = t_alpha6[0]
dawn_end = t_alpha8[0]
noon = 12
dusk_start = t_alpha8[1]
dusk_end = t_alpha6[1]
sunset = 12 + 1 / deg_to_rad(15) * np.arccos(-np.tan(phi) * np.tan(delta))


def color_lerp(x, y, a):
    """
    Interpolates linearly between the color tuples x and y using the factor a.
    :return: x * (1 - a) + y * a
    """
    return tuple(xi * (1 - a) + yi * a for xi, yi in zip(x, y))


class Weather(Enum):
    """
    The weather (usually randomly sampled) in the scene.
    """
    CLEAR = 0
    CLOUDY = 1


class Front3DLightSampler(Module):
    """
    Samples a set of light configurations for...
    - A global directional light source simulating the effect of the sun.
    - Environment light.
    - All ceilings in the scene (Front3DRefinedLoader assigns emissive materials to ceilings).
    - Lamps.
    """
    def __init__(self, config):
        Module.__init__(self, config)
        self.weather = Weather(0)
        self.time = 12.0  # local solar time
        self.sun_elevation_angle = compute_elevation_angle(self.time)
        self.is_day = True
        self.red_sky = False  # red sky phenomenon at dusk and dawn
        self.sky_interp_factor = 0  # use one factor for environment light and sun.
        self.number_of_samples = config.get_int('number_of_samples', 1)
        self.high_variance_mode = config.get_int('high_variance_mode', False)
        if self.high_variance_mode:
            # In high variance mode, four lighting settings are used: Day+clear, day+cloudy, night, red sky.
            self.number_of_samples = 4

    def run(self):
        bpy.context.scene.frame_end = max(bpy.context.scene.frame_end, self.number_of_samples)

        all_objects = get_all_blender_mesh_objects()
        front_3d_objs = [obj for obj in all_objects if "is_3D_future" in obj and obj["is_3D_future"]]
        ceiling_objs = [obj for obj in front_3d_objs if "ceiling" in obj.name.lower()]
        lamp_objs = [obj for obj in front_3d_objs if "lamp" in obj.name.lower()]

        sun = Light()
        sun.set_type('SUN')
        sun.set_location((0, 0, 10))

        for frame in range(self.number_of_samples):
            if self.high_variance_mode and frame < 4:
                if frame == 0:
                    # Day scene (clear, morning).
                    self.weather = Weather.CLEAR
                    self.time = np.random.uniform(low=dawn_end + sys.float_info.epsilon, high=dawn_end + 4)
                elif frame == 1:
                    # Day scene (cloudy, noon).
                    self.weather = Weather.CLOUDY
                    self.time = np.random.uniform(low=noon - 1, high=noon + 1)
                elif frame == 2:
                    # Night scene.
                    self.weather = Weather.CLEAR
                    self.time = np.random.uniform(low=midnight, high=sunrise)
                elif frame == 3:
                    # Evening scene.
                    self.weather = Weather.CLEAR
                    self.time = np.random.uniform(low=dusk_start, high=dusk_end)
            else:
                self.weather = Weather(np.random.binomial(n=1, p=0.3))
                self.time = np.random.uniform(low=0, high=24)
            self.sun_elevation_angle = compute_elevation_angle(self.time)
            self.is_day = sunrise <= self.time <= sunset
            self.red_sky = (dawn_start <= self.time <= dawn_end) or (dusk_start <= self.time <= dusk_end)
            self.sky_interp_factor = np.random.uniform(0, 1)

            for ceiling in ceiling_objs:
                strength, color = self.sample_params_ceiling_light()
                ceiling.active_material.node_tree.nodes["Emission"].inputs[0].default_value = color
                ceiling.active_material.node_tree.nodes["Emission"].inputs[0].keyframe_insert(
                    "default_value", frame=frame)
                ceiling.active_material.node_tree.nodes["Emission"].inputs[1].default_value = strength
                ceiling.active_material.node_tree.nodes["Emission"].inputs[1].keyframe_insert(
                    "default_value", frame=frame)

            for lamp in lamp_objs:
                strength, color = self.sample_params_lamp()
                for node in lamp.active_material.node_tree.nodes:
                    if "Emission" in node.name:
                        node.inputs[0].default_value = color
                        node.inputs[0].keyframe_insert("default_value", frame=frame)
                        node.inputs[1].default_value = strength
                        node.inputs[1].keyframe_insert("default_value", frame=frame)

            sun_strength, sun_color, rot_angle, angle = self.sample_params_sun()
            sun.set_rotation_euler(rot_angle, frame=frame)
            sun.set_energy(sun_strength, frame=frame)
            sun.set_color(sun_color, frame=frame)
            sun.set_angle(angle, frame=frame)

            env_light_strength, env_light_color = self.sample_params_environment_light()
            bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = env_light_color
            bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = env_light_strength
            bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].keyframe_insert(
                "default_value", frame=frame)
            bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].keyframe_insert(
                "default_value", frame=frame)

    def sample_params_lamp(self):
        if self.is_day:
            if self.weather == Weather.CLEAR:
                probability_on = 0.1
            else:
                probability_on = 0.4
        else:
            probability_on = 0.8

        is_on = np.random.binomial(n=1, p=probability_on)
        if is_on:
            strength = np.random.uniform(5.0, 25.0)
            # White to orange-ish.
            color = color_lerp((1.0, 1.0, 1.0, 1.0), (1.0, 0.73, 0.49, 1.0), np.random.uniform(0, 1))
        else:
            strength = 0.0
            color = (0, 0, 0, 1.0)

        return strength, color

    def sample_params_ceiling_light(self):
        if self.is_day:
            if self.weather == Weather.CLEAR:
                probability_on = 0.15
            else:
                probability_on = 0.5
        else:
            probability_on = 0.85

        is_on = np.random.binomial(n=1, p=probability_on)
        if is_on:
            strength = np.random.uniform(0.6, 1.0)
        else:
            strength = 0.1

        # White to orange-ish.
        color = color_lerp((1.0, 1.0, 1.0, 1.0), (1.0, 0.73, 0.49, 1.0), np.random.uniform(0, 1))

        return strength, color

    def sample_params_sun(self):
        strength = max(np.sin(self.sun_elevation_angle), 0.5)  # == cos of solar zenith angle

        if self.is_day:
            if self.weather == Weather.CLOUDY:
                strength *= np.random.uniform(0.5, 2.0)
            else:
                strength *= np.random.uniform(6.0, 12.0)

        if self.red_sky:
            # Either red-ish or orange-ish sky during dawn and dusk.
            color = color_lerp((1.0, 0.15, 0.15), (1.0, 0.47, 0.25), self.sky_interp_factor)
            strength = np.random.uniform(4.0, 10.0)
        elif self.is_day:
            # Daylight.
            color = (1.0, 1.0, 1.0)
        else:
            # Night.
            color = (0.09, 0.14, 0.22)

        rot_angle = (np.pi / 2.0 - self.sun_elevation_angle, 0.0, np.random.uniform(0.0, 2.0 * np.pi))
        angular_diameter_angle = np.random.uniform(deg_to_rad(0.1), deg_to_rad(5.0))

        return strength, color, rot_angle, angular_diameter_angle

    def sample_params_environment_light(self):
        strength = max(np.sin(self.sun_elevation_angle), 0.5)  # == cos of solar zenith angle

        if self.is_day:
            if self.weather == Weather.CLOUDY:
                strength *= np.random.uniform(1.5, 3.0)
            else:
                strength *= np.random.uniform(2.0, 4.0)
        else:
            strength *= np.random.uniform(0.1, 0.6)

        if self.red_sky:
            # Either red-ish or orange-ish sky during dawn and dusk.
            color = color_lerp((1.0, 0.15, 0.17, 1.0), (1.0, 0.6, 0.35, 1.0), self.sky_interp_factor)
        elif self.is_day:
            # Daylight.
            color = color_lerp((1.0, 1.0, 1.0, 1.0), (0.42, 0.75, 1.0, 1.0), self.sky_interp_factor)
        else:
            # Night.
            color = (0.09, 0.14, 0.22, 1.0)

        return strength, color
