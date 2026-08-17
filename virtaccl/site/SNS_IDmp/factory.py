import math

from orbit.core.bunch import Bunch
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import WaterBagDist3D
from orbit.py_linac.lattice.LinacAccLatticeLib import LinacAccLattice
from orbit.py_linac.lattice.LinacAccLatticeLib import Sequence
from orbit.py_linac.lattice.LinacAccNodes import Quad
from orbit.py_linac.lattice.LinacAccNodes import Drift
from orbit.py_linac.lattice.LinacAccNodes import DCorrectorH
from orbit.py_linac.lattice.LinacAccNodes import DCorrectorV
from orbit.utils.consts import speed_of_light

from virtaccl.PyORBIT_Model.pyorbit_va_nodes import BPMclass
from virtaccl.PyORBIT_Model.pyorbit_va_nodes import WSclass
from virtaccl.PyORBIT_Model.pyorbit_va_nodes import ScreenClass
from virtaccl.PyORBIT_Model.bunch_generator import BunchGenerator


def make_lattice(debug: bool = False) -> LinacAccLattice:
    # Field strength and length of the quadrupoles
    quad_field = 0.5
    dch_field = 0.01
    dcv_field = -0.02
    mag_len = 0.673
    bpm_frequency = 402.5e6

    nodes = []
    WS00 = WSclass("WS00")
    nodes.append(WS00)

    D0 = Drift("Drift0")
    D0.setLength(6.48 - 5.31)
    nodes.append(D0)

    BPM00 = BPMclass("BPM00", frequency=bpm_frequency)
    nodes.append(BPM00)

    D1 = Drift("Drift1")
    D1.setLength((10.029 - mag_len / 2) - 6.48)
    nodes.append(D1)

    Q1 = Quad("Quad1")
    Q1.setLength(mag_len)
    Q1.setField(-quad_field)

    DCH1 = DCorrectorH("HCorrector1")
    DCH1.setParam("effLength", mag_len)
    DCH1.setField(dch_field)
    Q1.addChildNode(DCH1, Q1.EXIT)

    DCV1 = DCorrectorV("VCorrector1")
    DCV1.setParam("effLength", mag_len)
    DCV1.setField(dcv_field)
    Q1.addChildNode(DCV1, Q1.EXIT)

    BPM01 = BPMclass("BPM01", frequency=bpm_frequency)
    Q1.addChildNode(BPM01, Q1.EXIT)
    nodes.append(Q1)

    D2 = Drift("Drift2")
    D2.setLength((13.599 - mag_len / 2) - (10.029 + mag_len / 2))
    nodes.append(D2)

    Q2 = Quad("Quad2")
    Q2.setLength(mag_len)
    Q2.setField(quad_field)

    DCH2 = DCorrectorH("HCorrector2")
    DCH2.setParam("effLength", mag_len)
    DCH2.setField(-dch_field)
    Q2.addChildNode(DCH2, Q2.EXIT)

    DCV2 = DCorrectorV("VCorrector2")
    DCV2.setParam("effLength", mag_len)
    DCV2.setField(-dcv_field)
    Q2.addChildNode(DCV2, Q2.EXIT)

    BPM02 = BPMclass("BPM02", frequency=bpm_frequency)
    Q2.addChildNode(BPM02, Q2.EXIT)

    nodes.append(Q2)

    D3 = Drift("Drift3")
    D3.setLength(16.612 - (13.59872 + mag_len / 2))
    nodes.append(D3)

    WS01 = WSclass("WS01")
    nodes.append(WS01)

    D4 = Drift("Drift4")
    D4.setLength(17.380 - 16.612)
    nodes.append(D4)

    BPM03 = BPMclass("BPM03", frequency=bpm_frequency)
    nodes.append(BPM03)

    D5 = Drift("Drift5")
    D5.setLength(12.998)
    nodes.append(D5)

    Screen = ScreenClass("Screen")
    nodes.append(Screen)

    idmp = Sequence("IDmp")
    idmp.setNodes(nodes)

    lattice = LinacAccLattice("My Lattice")
    lattice.setNodes(nodes)
    lattice.initialize()
    if debug:
        print("Total length=", lattice.getLength())

    return lattice


def make_bunch(
    nparts: int = 1000,
    x_off: float = 0.0,
    xp_off: float = 0.0,
    y_off: float = 0.0,
    yp_off: float = 0.0,
    alpha_x: float = 0.3777,
    alpha_y: float = -0.7225,
    alpha_z: float = +17.0460,
    beta_x: float = 7.5421,
    beta_y: float = 9.1459,
    beta_z: float = 179.6212,
    eps_x: float = 0.4249,
    eps_y: float = 0.3691,
    eps_z: float = 1.1498,
    dist: str = "gauss",
    debug: bool = False,
) -> Bunch:
    # Twiss parameters at MEBT entrance.
    # Transverse emittances are unnormalized and in [pi * mm * mrad].
    # Longitudinal emittance is in [pi * eV * sec].
    e_kin_ini = 1.349648024  # [GeV]
    mass = 0.939294  # [GeV]
    gamma = (mass + e_kin_ini) / mass
    beta = math.sqrt(gamma * gamma - 1.0) / gamma
    frequency = 402.5e6

    # Emittances are normalized --- transverse by beta * gamma and long. by beta * gamma^3.
    alpha_x, beta_x, emitt_x = (alpha_x, beta_x, eps_x)
    alpha_y, beta_y, emitt_y = (alpha_y, beta_y, eps_y)
    alpha_z, beta_z, emitt_z = (alpha_z, beta_z, eps_z)

    # Unnormalize emittances [m * rad].
    emitt_x = 1.0e-6 * emitt_x / (gamma * beta)
    emitt_y = 1.0e-6 * emitt_y / (gamma * beta)
    emitt_z = 1.0e-6 * emitt_z / (gamma**3 * beta)

    # Transform longitudinal coordinates to pyORBIT units [GeV * m].
    emitt_z = emitt_z * gamma**3 * beta**2 * mass
    beta_z = beta_z / (gamma**3 * beta**2 * mass)

    twiss_x = TwissContainer(alpha_x, beta_x, emitt_x)
    twiss_y = TwissContainer(alpha_y, beta_y, emitt_y)
    twiss_z = TwissContainer(alpha_z, beta_z, emitt_z)

    bunch_gen = BunchGenerator(twiss_x, twiss_y, twiss_z)
    bunch_gen.setKinEnergy(e_kin_ini)  # [GeV]
    bunch_gen.setBeamCurrent(38.0)  # [mA]

    bunch = None
    if dist == "gauss":
        bunch = bunch_gen.getBunch(nParticles=nparts, distributorClass=GaussDist3D)
    elif dist == "waterbag":
        bunch = bunch_gen.getBunch(nParticles=nparts, distributorClass=WaterBagDist3D)
    else:
        raise ValueError(f"Invalid distribution '{dist}'")

    bunch.charge(1.0)

    for i in range(nparts):
        x = bunch.x(i)
        y = bunch.y(i)
        xp = bunch.xp(i)
        yp = bunch.yp(i)
        bunch.x(i, x + x_off)
        bunch.y(i, y + y_off)
        bunch.xp(i, xp + xp_off)
        bunch.yp(i, yp + yp_off)

    return bunch
