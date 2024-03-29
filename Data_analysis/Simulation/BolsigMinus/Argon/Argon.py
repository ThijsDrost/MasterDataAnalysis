from General.Simulation._BolsigMinus import BolsigMinus

loc = r"E:\OneDrive - TU Eindhoven\Master thesis\Bolsig+\bolsigminus.exe"
cross_sections = r"E:\OneDrive - TU Eindhoven\Master thesis\Cross sections"

# %%
# bolsig = BolsigMinus(loc, {rf"'{cross_sections}\Argon2.txt'": ['Ar', 'Ar*']}, [0.9, 0.1], n_grid=300, ion_degree=1e-8)
# out_loc = r"E:\OneDrive - TU Eindhoven\Master thesis\Bolsig+\data\Argon\test.txt"
# bolsig.run_series(out_loc, [5e3, 1e-2], print_stdout=True, run_variable=1, num=10)

# %%
bolsig = BolsigMinus(loc, {rf"'{cross_sections}\Argon2.txt'": ['Ar', 'Ar*']}, [1, 0], n_grid=300)
out_loc = r"E:\OneDrive - TU Eindhoven\Master thesis\Bolsig+\data\Argon"
for exponent in range(-6, -1, 1):
    bolsig.gas_fractions = (1-10**exponent, 10 ** exponent)
    bolsig.run_2D(rf"{out_loc}\var_En_nIon_with_Ars\Ars_exp({exponent}).txt", [1e-2, 5e3], ((1e-8, 1e-4), 25, 3),
                  '_ion_degree', print_stdout=False, run_variable=1, num=25, pre_print=f'Exponent {exponent}: ')
    # print(f'\r {exponent} done', end='')
