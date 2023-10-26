from matplotlib.cm import register_cmap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

cmap = LinearSegmentedColormap.from_list(
    "mycmap", ["#1B346C", "#01ABE9", "#F1F8F1", "#F54B1A"]
)

register_cmap(name="mycmap", cmap=cmap)


# Evaluate the concentration profile in the odd diffusion system
def evaluate_solution(
    D_iso=10.0, D_odd=10.0, flux=10, flux_ratio=0.1, position=4, scheme=0
):
    # Evaluate solution to the odd diffusion system using finite differences
    # Will construct matrix L and right hand side f
    # and then solve the system for concentrations u
    # Depending on the scheme, will use different discretizations
    # Scheme 0: 2nd order parallel to plate, 4th order perpendicular to plate
    # Scheme 1: 4th order parallel to plate, 2nd order perpendicular to plate
    # Construct mesh
    # Convention on arrays are
    # (X_{ij}, Y_{ij}) = (x_i, y_j), 0 <= i,j <= n
    n = 800
    length = 80
    h = length / n
    x = np.linspace(-length / 2, length / 2, n + 1)
    y = np.linspace(-length / 2, length / 2, n + 1)
    X, Y = np.meshgrid(x, y)

    # Placement of the plates
    # Have some length and width I'll need to address
    length_plate = 10
    width_plate = 2
    # width_plate = 3
    length_plate_disc = int(length_plate / h)
    width_plate_disc = int(width_plate / h)
    # Plate 1
    x1 = -position
    y1 = 0
    x1_disc = int((x1 + length / 2) / h)
    y1_disc = int((y1 + length / 2) / h)
    # Plate 2
    x2 = position
    y2 = 0
    x2_disc = int((x2 + length / 2) / h)
    y2_disc = int((y2 + length / 2) / h)

    # Specify plate currents
    J_x_0 = -flux
    J_x_1 = flux
    J_y_0 = -flux
    J_y_1 = flux

    # Construct right hand side
    f = np.zeros((n + 1, n + 1))
    # Enforce Dirichlet boundary conditions far from plate
    # Dirichlet boundary conditions
    f[0, :] = 0.2  # Bottom
    f[n, :] = 0.2  # Top
    f[:, 0] = 0.2  # Left
    f[:, n] = 0.2  # Right
    # Will set equal to zero inside the plates
    f[
        (y1_disc - length_plate_disc // 2) : (
            y1_disc + length_plate_disc // 2
        ),
        (x1_disc - width_plate_disc // 2) : (x1_disc + width_plate_disc // 2),
    ] = 0
    f[
        (y2_disc - length_plate_disc // 2) : (
            y2_disc + length_plate_disc // 2
        ),
        (x2_disc - width_plate_disc // 2) : (x2_disc + width_plate_disc // 2),
    ] = 0
    # Fluxes introduce nonzero f terms on the plate boundaries
    # Based on scheme, will use different coefficients for variables
    if scheme == 0:
        flux_coeff = -8 * h * D_iso
    elif scheme == 1:
        flux_coeff = -6 * h * D_iso

    # Plate 1
    # Left
    for j in [x1_disc - width_plate_disc // 2 - 1]:
        for i in range(
            y1_disc - length_plate_disc // 2, y1_disc + length_plate_disc // 2
        ):
            f[i, j] = flux_coeff * J_y_0
    # Right
    for j in [x1_disc + width_plate_disc // 2]:
        for i in range(
            y1_disc - length_plate_disc // 2, y1_disc + length_plate_disc // 2
        ):
            f[i, j] = flux_coeff * J_y_1 * flux_ratio
    # Top
    for j in range(
        x1_disc - width_plate_disc // 2, x1_disc + width_plate_disc // 2
    ):
        for i in [y1_disc + length_plate_disc // 2]:
            f[i, j] = flux_coeff * J_x_1
    # Bottom
    for j in range(
        x1_disc - width_plate_disc // 2, x1_disc + width_plate_disc // 2
    ):
        for i in [y1_disc - length_plate_disc // 2 - 1]:
            f[i, j] = flux_coeff * J_x_0
    # Plate 2
    # Left
    for j in [x2_disc - width_plate_disc // 2 - 1]:
        for i in range(
            y2_disc - length_plate_disc // 2, y2_disc + length_plate_disc // 2
        ):
            f[i, j] = flux_coeff * J_y_0 * flux_ratio
    # Right
    for j in [x2_disc + width_plate_disc // 2]:
        for i in range(
            y2_disc - length_plate_disc // 2, y2_disc + length_plate_disc // 2
        ):
            f[i, j] = flux_coeff * J_y_1
    # Top
    for j in range(
        x2_disc - width_plate_disc // 2, x2_disc + width_plate_disc // 2
    ):
        for i in [y2_disc + length_plate_disc // 2]:
            f[i, j] = flux_coeff * J_x_1
    # Bottom
    for j in range(
        x2_disc - width_plate_disc // 2, x2_disc + width_plate_disc // 2
    ):
        for i in [y2_disc - length_plate_disc // 2 - 1]:
            f[i, j] = flux_coeff * J_x_0
    # Reshape to a vector
    f = f.reshape((n + 1) ** 2)

    # Construct matrix
    # Use coo_array to help initialize, and then convert to csr
    # Track the row and column indices, and then pass them in
    # Per convention, same row entries are adjacent (y-direction)
    # Same column entries are separated by n+1 (x-direction)
    row_index = []
    col_index = []
    data = []
    # Dirichlet boundary conditions first
    # Left
    for i in range(0, n + 1):
        ind = i * (n + 1)
        row_index.append(ind)
        col_index.append(ind)
        data.append(1)
    # Right
    for i in range(0, n + 1):
        ind = i * (n + 1) + n
        row_index.append(ind)
        col_index.append(ind)
        data.append(1)
    # Top
    for i in range(1, n):
        ind = i
        row_index.append(ind)
        col_index.append(ind)
        data.append(1)
    # Bottom
    for i in range(1, n):
        ind = i + n * (n + 1)
        row_index.append(ind)
        col_index.append(ind)
        data.append(1)
    # Now the plates
    # Plate 1
    for j in range(
        (x1_disc - width_plate_disc // 2), (x1_disc + width_plate_disc // 2)
    ):
        for i in range(
            (y1_disc - length_plate_disc // 2),
            (y1_disc + length_plate_disc // 2),
        ):
            ind = i * (n + 1) + j
            row_index.append(ind)
            col_index.append(ind)
            data.append(1)
    # Plate 2
    for j in range(
        (x2_disc - width_plate_disc // 2), (x2_disc + width_plate_disc // 2)
    ):
        for i in range(
            (y2_disc - length_plate_disc // 2),
            (y2_disc + length_plate_disc // 2),
        ):
            ind = i * (n + 1) + j
            row_index.append(ind)
            col_index.append(ind)
            data.append(1)
    # Now the interior
    # Standard five point stencil
    for i in range(1, n):
        for j in range(1, n):
            # Only implement if not adjacent to plate
            statement_x1_left = j < (x1_disc - width_plate_disc // 2 - 1)
            statement_x1_right = j > (x1_disc + width_plate_disc // 2)
            statement_y1_bottom = i < (y1_disc - length_plate_disc // 2 - 1)
            statement_y1_top = i > (y1_disc + length_plate_disc // 2)
            statement_x2_left = j < (x2_disc - width_plate_disc // 2 - 1)
            statement_x2_right = j > (x2_disc + width_plate_disc // 2)
            statement_y2_bottom = i < (y2_disc - length_plate_disc // 2 - 1)
            statement_y2_top = i > (y2_disc + length_plate_disc // 2)
            # If a plate node, all of these statements
            # for that plate should be false
            if (
                statement_x1_left
                or statement_x1_right
                or statement_y1_bottom
                or statement_y1_top
            ) and (
                statement_x2_left
                or statement_x2_right
                or statement_y2_bottom
                or statement_y2_top
            ):
                governing_append(i, j, D_iso, h, n, row_index, col_index, data)
            # Now check if a "corner" node
            # Eight potential sites
            if (j == x1_disc - width_plate_disc // 2 - 1) and (
                i == y1_disc - length_plate_disc // 2 - 1
            ):
                governing_append(i, j, D_iso, h, n, row_index, col_index, data)
            if (j == x1_disc - width_plate_disc // 2 - 1) and (
                i == y1_disc + length_plate_disc // 2
            ):
                governing_append(i, j, D_iso, h, n, row_index, col_index, data)
            if (j == x1_disc + width_plate_disc // 2) and (
                i == y1_disc - length_plate_disc // 2 - 1
            ):
                governing_append(i, j, D_iso, h, n, row_index, col_index, data)
            if (j == x1_disc + width_plate_disc // 2) and (
                i == y1_disc + length_plate_disc // 2
            ):
                governing_append(i, j, D_iso, h, n, row_index, col_index, data)
            if (j == x2_disc - width_plate_disc // 2 - 1) and (
                i == y2_disc - length_plate_disc // 2 - 1
            ):
                governing_append(i, j, D_iso, h, n, row_index, col_index, data)
            if (j == x2_disc - width_plate_disc // 2 - 1) and (
                i == y2_disc + length_plate_disc // 2
            ):
                governing_append(i, j, D_iso, h, n, row_index, col_index, data)
            if (j == x2_disc + width_plate_disc // 2) and (
                i == y2_disc - length_plate_disc // 2 - 1
            ):
                governing_append(i, j, D_iso, h, n, row_index, col_index, data)
            if (j == x2_disc + width_plate_disc // 2) and (
                i == y2_disc + length_plate_disc // 2
            ):
                governing_append(i, j, D_iso, h, n, row_index, col_index, data)
    # Now to add in the plate nodes
    # Plate 1
    # Left
    for j in [x1_disc - width_plate_disc // 2 - 1]:
        for i in range(
            y1_disc - length_plate_disc // 2, y1_disc + length_plate_disc // 2
        ):
            left_plate_append(
                i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
            )
    # Right
    for j in [x1_disc + width_plate_disc // 2]:
        for i in range(
            y1_disc - length_plate_disc // 2, y1_disc + length_plate_disc // 2
        ):
            right_plate_append(
                i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
            )
    # Bottom
    for j in range(
        x1_disc - width_plate_disc // 2, x1_disc + width_plate_disc // 2
    ):
        for i in [y1_disc - length_plate_disc // 2 - 1]:
            bottom_plate_append(
                i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
            )
    # Top
    for j in range(
        x1_disc - width_plate_disc // 2, x1_disc + width_plate_disc // 2
    ):
        for i in [y1_disc + length_plate_disc // 2]:
            top_plate_append(
                i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
            )
    # Plate 2
    # Left
    for j in [x2_disc - width_plate_disc // 2 - 1]:
        for i in range(
            y2_disc - length_plate_disc // 2, y2_disc + length_plate_disc // 2
        ):
            left_plate_append(
                i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
            )
    # Right
    for j in [x2_disc + width_plate_disc // 2]:
        for i in range(
            y2_disc - length_plate_disc // 2, y2_disc + length_plate_disc // 2
        ):
            right_plate_append(
                i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
            )
    # Bottom
    for j in range(
        x2_disc - width_plate_disc // 2, x2_disc + width_plate_disc // 2
    ):
        for i in [y2_disc - length_plate_disc // 2 - 1]:
            bottom_plate_append(
                i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
            )
    # Top
    for j in range(
        x2_disc - width_plate_disc // 2, x2_disc + width_plate_disc // 2
    ):
        for i in [y2_disc + length_plate_disc // 2]:
            top_plate_append(
                i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
            )

    # Construct coo matrix
    coo = scipy.sparse.coo_matrix(
        (data, (row_index, col_index)), shape=((n + 1) ** 2, (n + 1) ** 2)
    )
    Lh = coo.tocsr()

    # Solve
    u = scipy.sparse.linalg.spsolve(Lh, f)
    # Save u, f, Lh to file
    np.savez("u_f_Lh.npz", u=u, f=f, Lh=Lh)
    u = np.reshape(u, (n + 1, n + 1))

    # Plot
    fmt = lambda x, pos: "$ {:.4g} $".format(x)

    fig = plt.figure(figsize=(7.0, 7.0))
    gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])

    # Mask to get only non-plate nodes
    mask = np.ones_like(u, dtype=bool)
    mask[
        (y1_disc - length_plate_disc // 2) : (
            y1_disc + length_plate_disc // 2
        ),
        (x1_disc - width_plate_disc // 2) : (x1_disc + width_plate_disc // 2),
    ] = False
    mask[
        (y2_disc - length_plate_disc // 2) : (
            y2_disc + length_plate_disc // 2
        ),
        (x2_disc - width_plate_disc // 2) : (x2_disc + width_plate_disc // 2),
    ] = False

    u_max = u[mask].max()
    u_min = u[mask].min()
    CS = ax0.contourf(
        X, Y, u, alpha=1, cmap=cmap, levels=np.linspace(u_min, u_max, 15)
    )
    # Rasterize
    for c in CS.collections:
        c.set_rasterized(True)
    ax0.set_facecolor("black")
    # Find max, min values of u away from the plates
    cbar = fig.colorbar(
        CS, ax=ax0, location="bottom", pad=0.2, format=FuncFormatter(fmt)
    )
    cbar.ax.tick_params(axis="y", which="major", labelsize=10)
    # ax0.set_xlim(left=-left_right_zoom,right=left_right_zoom)
    # ax0.set_ylim(bottom=-bottom_top_zoom,top=bottom_top_zoom)
    ax0.tick_params(axis="both", which="major", labelsize=10)
    ax0.tick_params(axis="both", which="minor", labelsize=10)
    ax0.text(
        0.03,
        0.64,
        r"$ j = {:.3g} $".format(flux),
        transform=ax0.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
    )
    ax0.text(
        0.03,
        0.74,
        r"$ D_{{\mathrm{{odd}}}} = {:.3g} $".format(D_odd),
        transform=ax0.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
    )
    ax0.text(
        0.03,
        0.84,
        r"$ D_{{\mathrm{{iso}}}} = {:.3g} $".format(D_iso),
        transform=ax0.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
    )
    ax0.text(
        0.03,
        0.94,
        r"$ r = {:.3g} $".format(2 * position),
        transform=ax0.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
    )
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    # Save figure
    plt.savefig("fd_two_plates_wc.png", dpi=300, bbox_inches="tight")
    plt.close()
    # Save data in compressed format with X,Y,u,f
    np.savez_compressed("fd_two_plates_wc.npz", X=X, Y=Y, u=u, f=f)


def governing_append(i, j, D_iso, h, n, row_index, col_index, data):
    # Append in the governing equations entries
    # Middle
    ind_middle = i * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind_middle)
    data.append(-4 * D_iso / h**2)
    # Left
    ind = i * (n + 1) + j - 1
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(D_iso / h**2)
    # Right
    ind = i * (n + 1) + j + 1
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(D_iso / h**2)
    # Top
    ind = (i + 1) * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(D_iso / h**2)
    # Bottom
    ind = (i - 1) * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(D_iso / h**2)


def left_plate_append(
    i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
):
    # Append stencil for entries left of a plate
    # Middle
    ind_middle = i * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind_middle)
    data.append(27 * D_iso * D_odd)
    # Left
    ind = i * (n + 1) + j - 1
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(-16 * D_iso * D_odd)
    # 2 Left
    ind = i * (n + 1) + j - 2
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(D_iso * D_odd)
    # Bottom
    ind = (i - 1) * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind)
    if scheme == 0:
        data.append(-4 * D_iso**2 - 6 * D_iso * D_odd + 3 * D_odd**2)
    elif scheme == 1:
        data.append(-3 * D_iso**2 - 6 * D_iso * D_odd + 4 * D_odd**2)
    # Top
    ind = (i + 1) * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind)
    if scheme == 0:
        data.append(4 * D_iso**2 - 6 * D_iso * D_odd - 3 * D_odd**2)
    elif scheme == 1:
        data.append(3 * D_iso**2 - 6 * D_iso * D_odd - 4 * D_odd**2)


def right_plate_append(
    i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
):
    # Append stencil for entries right of a plate
    # Middle
    ind_middle = i * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind_middle)
    data.append(-27 * D_iso * D_odd)
    # Right
    ind = i * (n + 1) + j + 1
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(16 * D_iso * D_odd)
    # 2 Right
    ind = i * (n + 1) + j + 2
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(-D_iso * D_odd)
    # Bottom
    ind = (i - 1) * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind)
    if scheme == 0:
        data.append(-4 * D_iso**2 + 6 * D_iso * D_odd + 3 * D_odd**2)
    elif scheme == 1:
        data.append(-3 * D_iso**2 + 6 * D_iso * D_odd + 4 * D_odd**2)
    # Top
    ind = (i + 1) * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind)
    if scheme == 0:
        data.append(4 * D_iso**2 + 6 * D_iso * D_odd - 3 * D_odd**2)
    elif scheme == 1:
        data.append(3 * D_iso**2 + 6 * D_iso * D_odd - 4 * D_odd**2)


def bottom_plate_append(
    i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
):
    # Append stencil for entries bottom of a plate
    # Middle
    ind_middle = i * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind_middle)
    data.append(-27 * D_iso * D_odd)
    # Bottom
    ind = (i - 1) * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(16 * D_iso * D_odd)
    # 2 Bottom
    ind = (i - 2) * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(-D_iso * D_odd)
    # Right
    ind = i * (n + 1) + j + 1
    row_index.append(ind_middle)
    col_index.append(ind)
    if scheme == 0:
        data.append(4 * D_iso**2 + 6 * D_iso * D_odd - 3 * D_odd**2)
    elif scheme == 1:
        data.append(3 * D_iso**2 + 6 * D_iso * D_odd - 4 * D_odd**2)
    # Left
    ind = i * (n + 1) + j - 1
    row_index.append(ind_middle)
    col_index.append(ind)
    if scheme == 0:
        data.append(-4 * D_iso**2 + 6 * D_iso * D_odd + 3 * D_odd**2)
    elif scheme == 1:
        data.append(-3 * D_iso**2 + 6 * D_iso * D_odd + 4 * D_odd**2)


def top_plate_append(
    i, j, D_iso, D_odd, h, n, row_index, col_index, data, scheme
):
    # Append stencil for entries top of a plate
    # Middle
    ind_middle = i * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind_middle)
    data.append(27 * D_iso * D_odd)
    # Top
    ind = (i + 1) * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(-16 * D_iso * D_odd)
    # 2 Top
    ind = (i + 2) * (n + 1) + j
    row_index.append(ind_middle)
    col_index.append(ind)
    data.append(D_iso * D_odd)
    # Left
    ind = i * (n + 1) + j - 1
    row_index.append(ind_middle)
    col_index.append(ind)
    if scheme == 0:
        data.append(-4 * D_iso**2 - 6 * D_iso * D_odd + 3 * D_odd**2)
    elif scheme == 1:
        data.append(-3 * D_iso**2 - 6 * D_iso * D_odd + 4 * D_odd**2)
    # Right
    ind = i * (n + 1) + j + 1
    row_index.append(ind_middle)
    col_index.append(ind)
    if scheme == 0:
        data.append(4 * D_iso**2 - 6 * D_iso * D_odd - 3 * D_odd**2)
    elif scheme == 1:
        data.append(3 * D_iso**2 - 6 * D_iso * D_odd - 4 * D_odd**2)
