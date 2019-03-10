matrix_conc = [[0 for i in range(500)] for i in range(45)]
concentration = []
density = 200
while True:
    concentration.append(density)
    if density == 780:
        break
    if density < 600:
        density += 50
    else:
        density += 5
print(concentration)
print(len(concentration))
