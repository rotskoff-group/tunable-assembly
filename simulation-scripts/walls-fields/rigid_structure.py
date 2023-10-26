def generate_rigid_positions(L, spacing):
    rigid_positions = []
    num = int(L / spacing + 1)
    for i in range(num):
        particle_position = [0, -L / 2 + i * spacing, 0]
        rigid_positions.append(particle_position)
    return rigid_positions, num
