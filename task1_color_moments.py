
from services.color_moments import ColorMomentsGenerator

color_moment_generator = ColorMomentsGenerator('gfg_dummy_pic.png')

color_moment_features = color_moment_generator.get_color_moments()


print("color moment Features: {}".format(color_moment_features))
