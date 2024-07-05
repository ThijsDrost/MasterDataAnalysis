from General.simulation.CrossSections import CrossSectionCollection

loc = r"C:\Users\20222772\Downloads\Argon.txt"
cross_sections = CrossSectionCollection.read_txt(loc)

out_loc = r"C:\Users\20222772\Downloads\Argon_Comsol.txt"
cross_sections.write_comsol(out_loc)
