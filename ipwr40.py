import math
from pathlib import Path

import numpy as np
import openmc


# ============================================================
# Aegis-40 / 40 MWe-class iPWR preliminary 2D OpenMC model
# OpenMC 0.15+ compatible
# ------------------------------------------------------------
# Geometry basis:
#   - 2D XY model with infinite axial height (Z-infinite)
#   - 61 assemblies in a hexagonal lattice (CAREM-like assembly count)
#   - Each assembly is a 17x17 square lattice (PWR-like)
#   - UO2 fuel with Gd2O3-bearing fuel pins, no soluble boron
#   - Reflective outer boundary -> no leakage approximation
# ============================================================


# -----------------------------
# Global design parameters
# -----------------------------
FUEL_ENRICHMENT = 4.0          # wt% U-235 in standard UO2 pins
GD_FUEL_ENRICHMENT = 3.6       # wt% U-235 in gadolinia pins
GD2O3_WT_FRAC = 0.04           # 4 wt% Gd2O3-bearing fuel pins

PIN_PITCH = 1.26              # cm
FUEL_RADIUS = 0.4096          # cm
GAP_RADIUS = 0.4178           # cm
CLAD_OUTER_RADIUS = 0.4750    # cm

GUIDE_INNER_RADIUS = 0.561    # cm
GUIDE_OUTER_RADIUS = 0.602    # cm
INSTR_INNER_RADIUS = 0.561    # cm
INSTR_OUTER_RADIUS = 0.602    # cm

ASSEMBLY_SIZE = 17
ASSEMBLY_PITCH = ASSEMBLY_SIZE * PIN_PITCH   # across flats approximation for lattice spacing
CORE_RINGS = 5                                # 61 assemblies total = 1+6+12+18+24

# mesh parameters for flux tally
MESH_NX = 300
MESH_NY = 300

# explicit finite box used for source/mesh/plot extents in the 2D infinite-height model
CORE_BOUNDARY_EDGE = (2.0 * CORE_RINGS * ASSEMBLY_PITCH) / math.sqrt(3.0)
PLOT_EXTENT_XY = 2.25 * CORE_BOUNDARY_EDGE
SOURCE_HALF_HEIGHT = 1.0

openmc.config["cross_sections"] = "/mnt/d/openmc_data/endfb-viii.0-hdf5/cross_sections.xml"

# -----------------------------
# Materials
# -----------------------------
def build_materials():
    """Build materials for the 2D Aegis-40 preliminary model.

    OpenMC 0.15+ supports enrichment through Material.add_element(...,
    enrichment=..., enrichment_type='wo').
    """

    # Standard UO2 fuel
    # UO2 fuel
    uo2 = openmc.Material(name='uo2')
    uo2.set_density('g/cm3', 10.5)
    uo2.add_nuclide('U235', 0.04, 'ao')
    uo2.add_nuclide('U238', 0.96, 'ao')
    uo2.add_nuclide('O16', 2.0, 'ao')

    # Gd2O3-bearing UO2 fuel
    # OpenMC does not have a single "Gd2O3" compound shortcut here, so we build
    # the mixture explicitly. For a preliminary model, we treat the composition as
    # 96 wt% UO2 matrix + 4 wt% Gd2O3 additive.
    gd_fuel = openmc.Material(name='gd_fuel')
    gd_fuel.set_density('g/cm3', 10.2)
    gd_fuel.add_nuclide('U235', 0.036, 'ao')
    gd_fuel.add_nuclide('U238', 0.924, 'ao')
    gd_fuel.add_nuclide('Gd155', 0.02, 'ao')
    gd_fuel.add_nuclide('Gd157', 0.02, 'ao')
    gd_fuel.add_nuclide('O16', 2.0, 'ao')

    # Helium gap
    helium = openmc.Material(name='Helium')
    helium.set_density('g/cm3', 0.001598)
    helium.add_element('He', 1.0)

    # Zircaloy-4 cladding
    zircaloy = openmc.Material(name='Zircaloy-4')
    zircaloy.set_density('g/cm3', 6.56)
    zircaloy.add_element('Zr', 0.9825, percent_type='wo')
    zircaloy.add_element('Sn', 0.0145, percent_type='wo')
    zircaloy.add_element('Fe', 0.0021, percent_type='wo')
    zircaloy.add_element('Cr', 0.0009, percent_type='wo')

    # Light water, no boron
    water = openmc.Material(name='water')
    water.set_density('g/cm3', 0.743)
    water.add_nuclide('H1', 2.0, 'ao')
    water.add_nuclide('O16', 1.0, 'ao')
    water.add_s_alpha_beta('c_H_in_H2O')

    # Stainless steel for simplified structural regions if needed later
    steel = openmc.Material(name='SS304')
    steel.set_density('g/cm3', 8.0)
    steel.add_element('Fe', 0.68, percent_type='wo')
    steel.add_element('Cr', 0.19, percent_type='wo')
    steel.add_element('Ni', 0.10, percent_type='wo')
    steel.add_element('Mn', 0.02, percent_type='wo')
    steel.add_element('Si', 0.01, percent_type='wo')

    mats = openmc.Materials([uo2, gd_fuel, helium, zircaloy, water, steel])
    return mats


# -----------------------------
# Pin cell universes
# -----------------------------
def build_fuel_pin(fuel_mat, gap_mat, clad_mat, water_mat, name='fuel pin'):
    """Create a cylindrical fuel pin universe."""
    fuel_or = openmc.ZCylinder(r=FUEL_RADIUS)
    gap_or = openmc.ZCylinder(r=GAP_RADIUS)
    clad_or = openmc.ZCylinder(r=CLAD_OUTER_RADIUS)

    fuel = openmc.Cell(fill=fuel_mat, region=-fuel_or)
    gap = openmc.Cell(fill=gap_mat, region=+fuel_or & -gap_or)
    clad = openmc.Cell(fill=clad_mat, region=+gap_or & -clad_or)
    moderator = openmc.Cell(fill=water_mat, region=+clad_or)

    return openmc.Universe(name=name, cells=[fuel, gap, clad, moderator])


def build_guide_tube(clad_mat, water_mat, name='guide tube'):
    gt_inner = openmc.ZCylinder(r=GUIDE_INNER_RADIUS)
    gt_outer = openmc.ZCylinder(r=GUIDE_OUTER_RADIUS)

    water_in = openmc.Cell(fill=water_mat, region=-gt_inner)
    clad = openmc.Cell(fill=clad_mat, region=+gt_inner & -gt_outer)
    water_out = openmc.Cell(fill=water_mat, region=+gt_outer)

    return openmc.Universe(name=name, cells=[water_in, clad, water_out])


# -----------------------------
# Assembly pattern helpers
# -----------------------------
def guide_tube_positions_17x17():
    """Return a conventional 17x17 PWR-like guide/instrument map.

    24 guide tubes + 1 central instrument tube.
    """
    return {
        (2, 5), (2, 8), (2, 11),
        (3, 3), (3, 13),
        (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
        (8, 2), (8, 5), (8, 11), (8, 14),
        (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
        (13, 3), (13, 13),
        (14, 5), (14, 8), (14, 11),
    }


def instrument_position_17x17():
    return (8, 8)


def gad_positions_12():
    """A symmetric 12-pin gadolinia pattern for preliminary workflow checks."""
    return {
        (4, 4), (4, 8), (4, 12),
        (8, 4),          (8, 12),
        (12, 4), (12, 8), (12, 12),
        (6, 6), (6, 10), (10, 6), (10, 10),
    }


# -----------------------------
# Assembly universe
# -----------------------------
def build_fuel_assembly(materials):
    mats = {m.name: m for m in materials}

    fuel_u = build_fuel_pin(
        mats['uo2'], mats['Helium'], mats['Zircaloy-4'], mats['water'],
        name='standard fuel pin'
    )
    gd_u = build_fuel_pin(
        mats['gd_fuel'], mats['Helium'], mats['Zircaloy-4'], mats['water'],
        name='gad fuel pin'
    )
    guide_u = build_guide_tube(mats['Zircaloy-4'], mats['water'], name='guide tube')
    inst_u = build_guide_tube(mats['Zircaloy-4'], mats['water'], name='instrument tube')
    water_u = openmc.Universe(name='assembly outer water')
    water_u.add_cell(openmc.Cell(fill=mats['water']))

    guide_positions = guide_tube_positions_17x17()
    inst_position = instrument_position_17x17()
    gad_pos = gad_positions_12()

    lattice = openmc.RectLattice(name='17x17 assembly lattice')
    lattice.pitch = (PIN_PITCH, PIN_PITCH)
    lattice.lower_left = (-ASSEMBLY_PITCH / 2.0, -ASSEMBLY_PITCH / 2.0)
    lattice.outer = water_u

    universes = []
    for j in range(ASSEMBLY_SIZE):
        row = []
        for i in range(ASSEMBLY_SIZE):
            pos = (i, j)
            if pos == inst_position:
                row.append(inst_u)
            elif pos in guide_positions:
                row.append(guide_u)
            elif pos in gad_pos:
                row.append(gd_u)
            else:
                row.append(fuel_u)
        universes.append(row)

    # RectLattice expects first row = highest y, so flip vertically
    lattice.universes = universes[::-1]

    prism = openmc.model.RectangularPrism(
        width=ASSEMBLY_PITCH,
        height=ASSEMBLY_PITCH,
        axis='z',
        origin=(0.0, 0.0)
    )
    assembly_cell = openmc.Cell(fill=lattice, region=-prism)
    return openmc.Universe(name='fuel assembly', cells=[assembly_cell])


# -----------------------------
# Core geometry
# -----------------------------
def build_core_geometry(materials):
    mats = {m.name: m for m in materials}

    assembly_u = build_fuel_assembly(materials)
    outer_water_u = openmc.Universe(name='core outer water')
    outer_water_u.add_cell(openmc.Cell(fill=mats['water']))

    hex_lat = openmc.HexLattice(name='61-assembly hex core')
    hex_lat.center = (0.0, 0.0)
    hex_lat.pitch = [ASSEMBLY_PITCH]
    hex_lat.outer = outer_water_u

    # OpenMC expects rings from outermost to innermost.
    ring4 = [assembly_u] * 24
    ring3 = [assembly_u] * 18
    ring2 = [assembly_u] * 12
    ring1 = [assembly_u] * 6
    ring0 = [assembly_u]
    hex_lat.universes = [ring4, ring3, ring2, ring1, ring0]

    # Bounding hexagonal prism with reflective BC => no leakage approximation.
    # across_flats estimate sized to fully contain 5-ring lattice.
    core_boundary = openmc.model.HexagonalPrism(
        edge_length=CORE_BOUNDARY_EDGE,
        orientation='y',
        boundary_type='reflective'
    )

    root_cell = openmc.Cell(fill=hex_lat, region=-core_boundary)
    root_universe = openmc.Universe(name='root universe', cells=[root_cell])
    return openmc.Geometry(root_universe)


# -----------------------------
# Settings
# -----------------------------
def build_settings(geometry):
    settings = openmc.Settings()
    settings.run_mode = 'eigenvalue'
    settings.batches = 180
    settings.inactive = 40
    settings.particles = 50000
    settings.temperature = {'method': 'interpolation'}

    source_box = openmc.stats.Box(
        [-PLOT_EXTENT_XY / 2.0, -PLOT_EXTENT_XY / 2.0, -SOURCE_HALF_HEIGHT],
        [ PLOT_EXTENT_XY / 2.0,  PLOT_EXTENT_XY / 2.0,  SOURCE_HALF_HEIGHT],
    )
    settings.source = openmc.IndependentSource(
        space=source_box,
        constraints={'fissionable': True, 'rejection_strategy': 'resample'}
    )

    return settings


# -----------------------------
# Tallies
# -----------------------------
def build_tallies(geometry):
    mesh = openmc.RegularMesh(name='core flux mesh')
    mesh.lower_left = (-PLOT_EXTENT_XY / 2.0, -PLOT_EXTENT_XY / 2.0, -SOURCE_HALF_HEIGHT)
    mesh.upper_right = (PLOT_EXTENT_XY / 2.0, PLOT_EXTENT_XY / 2.0, SOURCE_HALF_HEIGHT)
    mesh.dimension = (MESH_NX, MESH_NY, 1)

    flux_tally = openmc.Tally(name='flux mesh tally')
    flux_tally.filters = [openmc.MeshFilter(mesh)]
    flux_tally.scores = ['flux', 'fission']

    return openmc.Tallies([flux_tally])


# -----------------------------
# Plots
# -----------------------------
def build_plots(geometry):
    plots = openmc.Plots()

    p1 = openmc.Plot(name='full_core_xy')
    p1.basis = 'xy'
    p1.origin = (0.0, 0.0, 0.0)
    p1.width = (PLOT_EXTENT_XY, PLOT_EXTENT_XY)
    p1.pixels = (1500, 1500)
    p1.color_by = 'material'

    p2 = openmc.Plot(name='assembly_xy')
    p2.basis = 'xy'
    p2.origin = (0.0, 0.0, 0.0)
    p2.width = (ASSEMBLY_PITCH, ASSEMBLY_PITCH)
    p2.pixels = (1200, 1200)
    p2.color_by = 'material'

    plots += [p1, p2]
    return plots


# -----------------------------
# Model container
# -----------------------------
def build_model():
    materials = build_materials()
    geometry = build_core_geometry(materials)
    settings = build_settings(geometry)
    tallies = build_tallies(geometry)
    plots = build_plots(geometry)

    model = openmc.model.Model(
        geometry=geometry,
        materials=materials,
        settings=settings,
        tallies=tallies,
        plots=plots,
    )
    return model


def export_model(output_dir='.'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()
    model.export_to_xml(directory=output_dir)
    print(f'OpenMC XML exported to: {output_dir.resolve()}')
    return model


if __name__ == '__main__':
    export_model()
