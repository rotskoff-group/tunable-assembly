def generate_rigid_positions(L, spacing):
    rigid_positions = []
    num = int(L * L)
    offset = 0
    if L // 2 % 2 == 0:
        offset = 0.5
    for j in range(L):
        for i in range(L):
            particle_position = [
                -L / 2 + j * spacing + offset,
                -L / 2 + i * spacing + offset,
                0,
            ]
            rigid_positions.append(particle_position)
    return rigid_positions, num
