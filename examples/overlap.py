
import openmc

surface1 = openmc.Sphere(r=50, boundary_type="vacuum")
surface2 = openmc.Sphere(r=30)
cell1 = openmc.Cell(region=-surface1)
cell2 = openmc.Cell(region=-surface2)
geometry = openmc.Geometry([cell1, cell2])
geometry.export_to_xml("geometry_with_overlaps.xml")