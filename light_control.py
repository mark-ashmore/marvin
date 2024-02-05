import time
from python_hue_v2 import Hue, BridgeFinder

hue = Hue('0017887e2e47.local.', 'YCMWZq3gYGb7sM8f4-m3POpVnHvneg8VPh96EzYV')  # create Hue instance

def turn_on_light(name: str, brightness: int = 100.00) -> None:
    """Turn on a Hue light by its name."""
    lights = hue.lights
    for light in lights:
        metadata = light.metadata
        if metadata['name'] == name:
            light.on = True
            light.brightness = brightness

def turn_off_light(name: str) -> None:
    """Turn off a Hue light by its name."""
    lights = hue.lights
    for light in lights:
        metadata = light.metadata
        if metadata['name'] == name:
            light.on = False

def change_light_brightness(name: str, brightness: int) -> None:
    """Change a Hue light brightness by its name."""
    lights = hue.lights
    for light in lights:
        metadata = light.metadata
        if metadata['name'] == name:
            light.brightness = brightness

turn_on_light('Lounge lamp', 100.00)
turn_off_light('Living room 1')
time.sleep(10)
change_light_brightness('Lounge lamp', 50.00)
