from General.Data_handling.Bolsig_plus import BolsigRuns
from General.Data_handling.ComsolAnalyzer import Comsol1DAnalyzer

bolsig_data = BolsigRuns.read_file(r"C:\Users\20222772\Downloads\output.dat")
bolsig_data.plot_cross_section_with_energy(plot_kwargs={'xlim': (1, 100)})
