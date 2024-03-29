pressure = 1  # atm
temperature = 300  # K
voltage = 10_000  # V
distance = 0.2  # cm

particle_density = pressure / (temperature * 1.38064852e-23)  # m^-3
electric_field = voltage / distance  # V/m
print(particle_density, electric_field)
print(electric_field / particle_density)
