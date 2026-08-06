import json
import sys
from pathlib import Path

from virtaccl.PyORBIT_Model.pyorbit_va_nodes import BPMclass
from virtaccl.PyORBIT_Model.pyorbit_va_nodes import WSclass
from virtaccl.PyORBIT_Model.pyorbit_va_nodes import ScreenClass
from virtaccl.PyORBIT_Model.pyorbit_virtual_accelerator import PyorbitVirtualAcceleratorBuilder
from virtaccl.PyORBIT_Model.pyorbit_virtual_accelerator import add_pyorbit_arguments

from virtaccl.site.SNS_Linac.virtual_devices import Quadrupole
from virtaccl.site.SNS_Linac.virtual_devices import Corrector
from virtaccl.site.SNS_Linac.virtual_devices import Quadrupole_Power_Supply
from virtaccl.site.SNS_Linac.virtual_devices import Corrector_Power_Supply
from virtaccl.site.SNS_Linac.virtual_devices import WireScanner
from virtaccl.site.SNS_Linac.virtual_devices import BPM
from virtaccl.site.SNS_Linac.virtual_devices import Screen
from virtaccl.site.SNS_Linac.virtual_devices_SNS import SNS_Dummy_BCM
from virtaccl.site.SNS_Linac.virtual_devices_SNS import SNS_Dummy_ICS

from virtaccl.PyORBIT_Model.pyorbit_lattice_controller import OrbitModel
from virtaccl.beam_line import BeamLine
from virtaccl.EPICS_Server.ca_server import EPICS_Server
from virtaccl.EPICS_Server.ca_server import add_epics_arguments
from virtaccl.virtual_accelerator import VA_Parser

from .maker import get_lattice
from .maker import get_bunch


def parse_args():
    loc = Path(__file__).parent
    va_parser = VA_Parser()
    va_parser.set_description("Run the SNS Injection Dump PyORBIT virtual accelerator server.")
    va_parser = add_pyorbit_arguments(va_parser)

    # Set the defaults for the PyORBIT model.
    va_parser.remove_argument("--lattice")
    va_parser.remove_argument("--start")
    va_parser.remove_argument("--drift_length")
    va_parser.remove_argument("end")
    va_parser.remove_argument("--bunch")

    va_parser = add_epics_arguments(va_parser)

    # Json file that contains a dictionary connecting EPICS name of devices with their associated element model names.
    va_parser.add_argument(
        "--config_file",
        "-f",
        default=loc / "va_config.json",
        type=str,
        help="Pathname of config json file.",
    )

    # Json file that contains a dictionary connecting EPICS name of devices with their phase offset.
    va_parser.add_argument(
        "--phase_offset", default=None, type=str, help="Pathname of phase offset file."
    )

    va_args = va_parser.initialize_arguments()
    return va_args


def build(**kwargs):
    kwargs = idmp_arguments() | kwargs

    debug = kwargs["debug"]

    config_file = Path(kwargs["config_file"])
    with open(config_file, "r") as json_file:
        devices_dict = json.load(json_file)

    part_num = kwargs["particle_number"]

    latttice = get_lattice(debug=debug)
    bunch = get_bunch(part_num, x_off=0.002, xp_off=0.0003, debug=debug)

    model = OrbitModel(input_bunch=bunch, debug=debug)
    model.define_custom_node(BPMclass.node_type, BPMclass.parameter_list, diagnostic=True)
    model.define_custom_node(WSclass.node_type, WSclass.parameter_list, diagnostic=True)
    model.define_custom_node(ScreenClass.node_type, ScreenClass.parameter_list, diagnostic=True)
    model.set_beam_current(0.038)
    model.initialize_lattice(lattice)
    element_list = model.get_element_list()

    beam_line = BeamLine()

    refresh_rate = kwargs['refresh_rate']
    update_frequency = kwargs['device_frequency']

    offset_file = kwargs['phase_offset']
    if offset_file is not None:
        with open(offset_file, "r") as json_file:
            offset_dict = json.load(json_file)

    quad_ps = devices_dict["Quadrupole_Power_Supply"]
    quads = devices_dict["Quadrupole"]
    for name, device_dict in quads.items():
        ele_name = device_dict["PyORBIT_Name"]
        polarity = device_dict["Polarity"]
        if ele_name in element_list:
            initial_field = abs(model.get_element_parameters(ele_name)["dB/dr"])
            if "Power_Supply" in device_dict and device_dict["Power_Supply"] in quad_ps:
                ps_name = device_dict["Power_Supply"]
                ps_device = Quadrupole_Power_Supply(ps_name, initial_field)
                beam_line.add_device(ps_device)
                corrector_device = Quadrupole(
                    name, ele_name, power_supply=ps_device, polarity=polarity
                )
                beam_line.add_device(corrector_device)

    corr_ps = devices_dict["Corrector_Power_Supply"]
    correctors = devices_dict["Corrector"]
    for name, device_dict in correctors.items():
        ele_name = device_dict["PyORBIT_Name"]
        polarity = device_dict["Polarity"]
        if ele_name in element_list:
            initial_field = model.get_element_parameters(ele_name)["B"]
            if "Power_Supply" in device_dict and device_dict["Power_Supply"] in corr_ps:
                ps_name = device_dict["Power_Supply"]
                ps_device = Corrector_Power_Supply(ps_name, initial_field)
                beam_line.add_device(ps_device)
                corrector_device = Corrector(
                    name, ele_name, power_supply=ps_device, polarity=polarity
                )
                beam_line.add_device(corrector_device)

    wire_scanners = devices_dict["Wire_Scanner"]
    for name, model_name in wire_scanners.items():
        if model_name in element_list:
            # Passing refresh rate to the WireScanner device for velocity calculations
            ws_device = WireScanner(name, model_name, {
                'refresh_rate': refresh_rate,
                'update_frequency': update_frequency})
            beam_line.add_device(ws_device)

    bpms = devices_dict["BPM"]
    for name, model_name in bpms.items():
        if model_name in element_list:
            phase_offset = 0
            if offset_file is not None:
                phase_offset = offset_dict[name]
            bpm_device = BPM(name, model_name, phase_offset=phase_offset)
            beam_line.add_device(bpm_device)

    screen = devices_dict["Screen"]
    for name, model_name in screen.items():
        if model_name in element_list:
            screen_device = Screen(name, model_name)
            beam_line.add_device(screen_device)

    dummy_device = SNS_Dummy_BCM("Ring_Diag:BCM_D09", "HEBT_Diag:BPM11")
    beam_line.add_device(dummy_device)
    dummy_device = SNS_Dummy_ICS("ICS_Tim")
    beam_line.add_device(dummy_device)

    delay = kwargs["ca_proc"]
    server = EPICS_Server(process_delay=delay)

    idmp_virac = PyorbitVirtualAcceleratorBuilder(model, beam_line, server, **kwargs)
    return idmp_virac


def main():
    args = parse_args()
    va = build(**args).build()
    va.start_server()


if __name__ == "__main__":
    main()
