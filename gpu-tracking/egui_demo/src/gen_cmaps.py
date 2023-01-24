from matplotlib import cm
from matplotlib import colors
import numpy as np

LEVELS = 30
NESTED = False

def cmap_to_str(cmap, levels = 30, indent_level = 0, nested = NESTED):
	out = "["
	arr = cmap(np.linspace(0, 1, levels))
	for row in arr:
		out += "\n" + "\t"*(indent_level + 1)
		if nested:
			out += "["
		str_row = ""
		for val in row:
			str_row += f"{val}, "
		if nested:
			out += str_row[:-2]
			out += "],"
		else:
			out += str_row
	out += "\n" + "\t" * indent_level + "]"
	return out


all_maps = {map.name: map for ele in dir(cm) if isinstance((map:=getattr(cm, ele)), colors.Colormap) and not ele.endswith("_r")}
with open("src/colormaps.rs", "w") as file:
	file.write("use phf::phf_map;\n\n")
	file.write(f"pub const LEVEL_COUNT: usize = {LEVELS};\n\n")
	if NESTED:
		shape = "[[f32; 4]; LEVEL_COUNT]"
	else:
		shape = f"[f32; {int(LEVELS*4)}]"
	file.write(f"pub const MAPS: phf::Map<&'static str, {shape}> = " + "phf_map! {")
	for name, map in all_maps.items():
		file.write(f"\n\t\"{name}\" => {cmap_to_str(map, indent_level = 1)},")
		file.write("\n")
	file.write("\n};")


