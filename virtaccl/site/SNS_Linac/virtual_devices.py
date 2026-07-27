import sys
import time
import math
from random import randint, random
from typing import Dict, Any, Union, Literal
from enum import IntEnum

import numpy as np
from scipy.interpolate import interp2d

from virtaccl.beam_line import Device, AbsNoise, LinearT, PhaseT, PhaseTInv, LinearTInv, PosNoise, NormalizePeak


# Here are the device definitions that take the information from PyORBIT and translates/packages it into information for
# the server.
#
# All the devices need a name that will determine the name of the device in the EPICS server. If the corresponding
# element in PyORBIT has a different name, then it will need to be specified in the declaration as the model_name. If
# the device has settings (values changed by the user), they can be given initial values to match the model using
# initial_dict. These initial settings need to be in a dictionary using the appropriate keys defined by PyORBIT to
# differentiate different parameters. And finally, if the device has a phase, both setting and measurement, they can be
# given an offset using phase_offset.
#
# The strings denoted with a "_pv" are labels for EPICS. Changing these will alter the EPICS labels for values on the
# server. The strings denoted with a "_key" are the keys for parameters in PyORBIT for that device. These need to match
# the keys PyORBIT uses in the paramsDict for that devices corresponding PyORBIT element.

class Quadrupole(Device):
    # EPICS PV names
    field_readback_pv = 'B'  # [T/m]
    field_noise = 1e-6  # [T/m]

    # PyORBIT parameter keys
    field_key = 'dB/dr'  # [T/m]

    def __init__(self, name: str, model_name: str, power_supply: 'Quadrupole_Power_Supply',
                 power_shunt: 'Quadrupole_Power_Shunt' = None, polarity: Literal[-1, 1] = None):

        self.model_name = model_name
        self.power_supply = power_supply
        self.power_shunt = power_shunt

        if power_shunt is not None:
            connected_devices = [power_supply, power_shunt]
        else:
            connected_devices = power_supply

        super().__init__(name, self.model_name, connected_devices)

        self.pol_transform = LinearTInv(scaler=polarity)

        field_noise = AbsNoise(noise=Quadrupole.field_noise)

        # Registers the device's PVs with the server
        self.register_readback(Quadrupole.field_readback_pv, transform=self.pol_transform, noise=field_noise)

    # Return the setting value of the PV name for the device as a dictionary using the model key and it's value. This is
    # where the PV names are associated with their model keys.

    def get_field_from_PS(self):
        new_field = self.power_supply.get_parameter_value(Quadrupole_Power_Supply.field_set_pv)
        new_field = self.pol_transform.real(new_field)

        if self.power_shunt:
            shunt_field = self.power_shunt.get_parameter_value(Quadrupole_Power_Shunt.field_set_pv)
            shunt_field = self.pol_transform.real(shunt_field)
            new_field += shunt_field

        return new_field

    def get_model_optics(self) -> Dict[str, Dict[str, Any]]:
        new_field = self.get_field_from_PS()
        params_dict = {Quadrupole.field_key: new_field}
        model_dict = {self.model_name: params_dict}
        return model_dict

    def update_readbacks(self):
        rb_field = abs(self.get_field_from_PS())
        self.update_readback(Quadrupole.field_readback_pv, rb_field)


class Corrector(Device):
    # EPICS PV names
    field_readback_pv = 'B'  # [T]
    field_noise = 1e-6  # [T/m]

    # PyORBIT parameter keys
    field_key = 'B'  # [T]

    def __init__(self, name: str, model_name: str, power_supply: 'Corrector_Power_Supply',
                 polarity: Literal[-1, 1] = None):
        self.model_name = model_name
        self.power_supply = power_supply

        super().__init__(name, self.model_name, self.power_supply)

        self.pol_transform = LinearTInv(scaler=polarity)

        field_noise = AbsNoise(noise=Corrector.field_noise)

        # Registers the device's PVs with the server
        self.register_readback(Corrector.field_readback_pv, transform=self.pol_transform, noise=field_noise)

    # Return the setting value of the PV name for the device as a dictionary using the model key and it's value. This is
    # where the setting PV names are associated with their model keys.
    # These settings have been limited by field_limits, meaning that if the server has a value outside that range, the
    # model will receive the max or min limit defined above.

    def get_field_from_PS(self):
        new_field = self.power_supply.get_parameter_value(Corrector_Power_Supply.field_set_pv)

        field_limit_high = self.power_supply.get_parameter_value(Corrector_Power_Supply.field_high_limit_pv)
        field_limit_low = self.power_supply.get_parameter_value(Corrector_Power_Supply.field_low_limit_pv)
        if new_field > field_limit_high:
            new_field = field_limit_high
        elif new_field < field_limit_low:
            new_field = field_limit_low

        new_field = self.pol_transform.real(new_field)
        return new_field

    def get_model_optics(self) -> Dict[str, Dict[str, Any]]:
        new_field = self.get_field_from_PS()
        params_dict = {Corrector.field_key: new_field}
        model_dict = {self.model_name: params_dict}
        return model_dict

    def update_readbacks(self):
        rb_field = self.get_field_from_PS()
        self.update_readback(Corrector.field_readback_pv, rb_field)


class Bend(Device):
    # EPICS PV names
    field_readback_pv = 'B'  # [T]
    field_noise = 1e-6  # [T/m]

    # PyORBIT parameter keys
    field_key = 'B'  # [T]

    def __init__(self, name: str, model_name: str, power_supply: 'Bend_Power_Supply'):
        self.model_name = model_name
        self.power_supply = power_supply

        super().__init__(name, self.model_name, self.power_supply)

        field_noise = AbsNoise(noise=Bend.field_noise)

        # Registers the device's PVs with the server
        self.register_readback(Bend.field_readback_pv, noise=field_noise)

    def update_readbacks(self):
        rb_field = self.power_supply.get_parameter_value(Bend_Power_Supply.field_set_pv)
        self.update_readback(Bend.field_readback_pv, rb_field)


class Cavity(Device):
    # EPICS PV names
    phase_pv = 'CtlPhaseSet'  # [degrees (-180 - 180)]
    amp_pv = 'CtlAmpSet'  # [arb. units]
    amp_goal_pv = 'cavAmpGoal'  # [arb. units]
    blank_pv = 'BlnkBeam'  # [0 or 1]
    phase_rbk_pv = "cavPhaseAvg"
    amp_rbk_pv = "cavAmpAvg"

    # PyORBIT parameter keys
    phase_key = 'phase'  # [radians]
    amp_key = 'amp'  # [arb. units]

    # Device Defaults
    default_initial_phase = 0  # [radians]
    default_initial_amp = 1.0  # [arb. units]

    def __init__(self, name: str, model_name: str = None, initial_dict: Dict[str, Any] = None, phase_offset=0,
                 design_amp=15):
        if model_name is None:
            self.model_name = name
        else:
            self.model_name = model_name
        super().__init__(name, self.model_name)
        initial_dict = {} if initial_dict is None else initial_dict

        # Sets initial values for parameters.
        if Cavity.phase_key in initial_dict:
            initial_phase = initial_dict[Cavity.phase_key]
        else:
            initial_phase = Cavity.default_initial_phase
        if Cavity.amp_key in initial_dict:
            initial_amp = initial_dict[Cavity.amp_key]
        else:
            initial_amp = Cavity.default_initial_amp

        self.design_amp = design_amp  # [MV]

        # Create old amp variable for ramping
        self.old_amp = initial_amp

        # Adds a phase offset. Default is 0 offset.
        offset_transform = PhaseTInv(offset=phase_offset, scaler=180 / math.pi)
        amp_transform = LinearTInv(scaler=design_amp)

        # Registers the device's PVs with the server
        self.register_setting(Cavity.phase_pv, default=initial_phase, transform=offset_transform)
        self.register_setting(Cavity.amp_pv, default=initial_amp, transform=amp_transform)
        self.register_setting(Cavity.amp_goal_pv, default=initial_amp, transform=amp_transform)
        self.register_setting(Cavity.blank_pv, default=0.0)

        self.register_readback(Cavity.phase_rbk_pv,Cavity.phase_pv,transform=offset_transform)
        self.register_readback(Cavity.amp_rbk_pv,Cavity.amp_pv,transform=amp_transform)

    # Return the setting value of the PV name for the device as a dictionary using the model key and it's value. This is
    # where the setting PV names are associated with their model keys.
    def get_model_optics(self) -> Dict[str, Dict[str, Any]]:
        phase = self.get_parameter_value(Cavity.phase_pv)
        params_dict = {Cavity.phase_key: phase}

        goal_value = self.get_parameter_value(Cavity.amp_goal_pv)
        set_value = self.get_parameter_value(Cavity.amp_pv)
        if goal_value != self.old_amp:
            model_value = goal_value
            self.server_setting_override(Cavity.amp_pv, goal_value)
        elif set_value != self.old_amp:
            model_value = set_value
            self.server_setting_override(Cavity.amp_goal_pv, set_value)
        else:
            model_value = self.old_amp
        self.old_amp = model_value

        # If the cavity is blanked, turn off acceleration.
        blank_value = self.get_parameter_value(Cavity.blank_pv)
        if blank_value == 0:
            params_dict = params_dict | {Cavity.amp_key: model_value}
        else:
            params_dict = params_dict | {Cavity.amp_key: 0.0}

        model_dict = {self.model_name: params_dict}
        return model_dict


class BPM(Device):
    # EPICS PV names
    x_pv = 'xAvg'  # [mm]
    y_pv = 'yAvg'  # [mm]
    xy_noise = 1e-8  # [mm]
    phase_pv = 'phaseAvg'  # [degrees]
    phase_noise = 1e-4  # [degrees]
    amp_pv = 'amplitudeAvg'  # [mA]
    amp_noise = 1e-6  # mA
    oeda_pv = 'OEDA'  # Off Energy Delay Adjustment. Should be 0 during production.

    # PyORBIT parameter keys
    x_key = 'x_avg'  # [m]
    y_key = 'y_avg'  # [m]
    phase_key = 'phi_avg'  # [radians]
    amp_key = 'amp_avg'  # [A]

    def __init__(self, name: str, model_name: str = None, phase_offset=0):
        if model_name is None:
            self.model_name = name
        else:
            self.model_name = model_name
        super().__init__(name, self.model_name)

        # Changes the units from meters to millimeters for associated PVs.
        milli_units = LinearTInv(scaler=1e3)

        # Creates flat noise for associated PVs.
        xy_noise = AbsNoise(noise=BPM.xy_noise)
        phase_noise = AbsNoise(noise=BPM.phase_noise)
        amp_noise = PosNoise(noise=BPM.amp_noise)

        # Adds a phase offset. Default is 0 offset.
        offset_transform = PhaseTInv(offset=phase_offset, scaler=180 / math.pi)

        # Registers the device's PVs with the server.
        self.register_measurement(BPM.x_pv, noise=xy_noise, transform=milli_units)
        self.register_measurement(BPM.y_pv, noise=xy_noise, transform=milli_units)
        self.register_measurement(BPM.phase_pv, noise=phase_noise, transform=offset_transform)
        self.register_measurement(BPM.amp_pv, noise=amp_noise, transform=milli_units)

        self.register_setting(BPM.oeda_pv, default=0)

    # Updates the measurement values on the server. Needs the model key associated with its value and the new value.
    # This is where the measurement PV name is associated with its model key.
    def update_measurements(self, new_params: Dict[str, Dict[str, Any]] = None):
        bpm_params = new_params[self.model_name]
        amp = bpm_params[BPM.amp_key]
        self.update_measurement(BPM.amp_pv, amp)

        if amp < 1e-8:
            x_avg = (random() * 2 - 1) * 0.1
            y_avg = (random() * 2 - 1) * 0.1
            phase_avg = (random() * 2 - 1) * math.pi
        else:
            x_avg = bpm_params[BPM.x_key]
            y_avg = bpm_params[BPM.y_key]
            phase_avg = bpm_params[BPM.phase_key]

        self.update_measurement(BPM.x_pv, x_avg)
        self.update_measurement(BPM.y_pv, y_avg)
        self.update_measurement(BPM.phase_pv, phase_avg)


class WireScanner(Device):
    #  _
    # |\    Direction of motion
    #   \
    #    \
    #    _\|
    #   __
    #  /  \.
    # /     \.
    # \___y___\
    #  \     /|\
    #   \   D | \
    #    \/   |  \
    #     \   X   \
    #      \  |    \
    #       \ |     \
    #        \|      \
    #             .|.
    #            --*--    <-- Beam
    #             '|'
    # EPICS PV names
    # Integrated charge at a specific point
    x_charge_pv = 'Hor_Cont'  # [arb. units]
    y_charge_pv = 'Ver_Cont'  # [arb. units]
    d_charge_pv = 'Diag_Cont' # [arb. units]
    # General Moving
    position_pv = 'MoveToPos'  # [mm]
    position_readback_pv = 'Position'  # [mm]
    speed_pv = 'Speed'  # [mm/s]
    # These define "hard" limits between the different wires scan ranges
    stop_1_pv = 'Stop1' # [mm]
    stop_2_pv = 'Stop2' # [mm]
    stop_3_pv = 'Stop3' # [mm]
    # Horizontal wire scan range and step size
    x_start_pv = 'Hor_Start' # [mm]
    x_stop_pv = 'Hor_Stop' # [mm]
    x_dx_pv = "Hor_dX" # [mm]
    # Vertical wire scan range and step size
    y_dx_pv = "Ver_dX" # [mm]
    y_start_pv = "Ver_Start" # [mm]
    y_stop_pv = "Ver_Stop" # [mm]
    # Diagonal wire scan range and step size
    d_dx_pv = "Diag_dX" # [mm]
    d_start_pv = "Diag_Start" # [mm]
    d_stop_pv = "Diag_Stop" # [mm]
    # Fitting PVs
    x_avg_pv = 'Hor_Mean_gs'  # [mm]
    y_avg_pv = 'Ver_Mean_gs'  # [mm]
    d_avg_pv = 'Diag_Mean_gs' # [mm]
    x_sigma_pv = 'Hor_Sigma_gs'  # [mm]
    y_sigma_pv = 'Ver_Sigma_gs'  # [mm]
    d_sigma_pv = 'Diag_Sigma_gs' # [mm]
    # Resulting profile PVs
    x_profile_pv = 'Hor_live_sig'  # [arb. units]
    x_axis_pv = 'Hor_live_pos'  # [mm]
    y_profile_pv = 'Ver_live_sig'  # [arb. units]
    y_axis_pv = 'Ver_live_pos'  # [mm]
    d_profile_pv = 'Diag_live_sig' # [arb. units]
    d_axis_pv = 'Diag_live_pos'  # [mm]
    # Pulse trace PVs
    x_trace_pv = "Hor_trace_raw"
    y_trace_pv = "Ver_trace_raw"
    d_trace_pv = "Diag_trace_raw"
    trace_time_pv = "trace_x"
    refresh_rate_pv = 'BeamRepRate' # [Hz]
    # Command PV is what client uses to request wire scanner to do things
    command_pv = 'Command' # int
    commands_dict = {
        "Move": 7,
        "Scan": 21,
        "Park": 5,
        "Halt": 8,
        "Abort": 2
    }
    # PyORBIT parameter keys
    x_hist_key = 'x_histogram'  # [m, arb. units]
    y_hist_key = 'y_histogram'  # [m, arb. units]
    # d_hist_key = 'd_histogram'  # [m, arb. units]
    x_avg_key = 'x_avg'  # [m]
    y_avg_key = 'y_avg'  # [m]
    # d_avg_key = 'd_avg'  # [m]
    x_sigma_key = 'x_sigma'  # [m]
    y_sigma_key = 'y_sigma'  # [m]
    # d_sigma_key = 'd_sigma'  # [m]
    bin_number_key = 'bin_number'  # [number]

    # Device keys
    position_key = 'wire_position'  # [m]
    speed_key = 'wire_speed'  # [m/s]
    refresh_rate_key = 'refresh_rate' # [Hz]
    dx_key = 'dx' # [m]

    # Device Constants
    max_speed = 0.002 # [mm/s]
    park_position = -.035 # [m]
    x_offset = -0.01  # [m]
    y_offset = 0.01  # [m]
    # diagonal wire will be in the middle
    d_offset = 0.0 # [m]
    wire_coeff = 1 / math.sqrt(2)
    # Parameters for waveform trace
    trace_bin_number = 1024
    pulse_width = 0.01  # [ms]
    trace_time = 0.1  # [ms]
    reduction_factor = 20 # [arb]

    # Device Defaults. These will be overridden by initial_dict if keys match
    # Every item here will be accessible as self.<key>
    initial_defaults = {
        "x_offset": -0.015, # [m]
        "y_offset": 0.015,# [m]
        "x_start": .005, # [m]
        "x_stop": .025, # [m]
        "y_start": -.025, # [m]
        "y_stop": -.005, # [m]
        "d_start": -.005, # [m]
        "d_stop": .005, # [m]
        "refresh_rate": 100,
        "position": -0.03,# [m]
        "wire_count": 2,
        "bin_number": 50,
        "d_dx": 0.001, # [m]
        "x_dx": 0.001, # [m]
        "y_dx": 0.001, # [m]
        "stop_1": -0.030, # [m]
        "stop_2": 0.000, # [m]
        "stop_3": 0.030 # [m]
    }

    def __init__(self, name: str, model_name: str = None, initial_dict: Dict[str, Any] = None):
        if model_name is None:
            self.model_name = name
        else:
            self.model_name = model_name
        super().__init__(name, self.model_name)
        initial_dict = {} if initial_dict is None else initial_dict

        # Apply each key in the passed dict to self
        for key in initial_dict:
            setattr(self, key, initial_dict[key])
        # Apply defaults that are not already set by the passed dict
        for key in self.initial_defaults:
            if key not in initial_dict:
                setattr(self, key, self.initial_defaults[key])

        # Defines internal parameters to keep track of the wire position.
        self.last_wire_pos = self.position
        self.last_wire_time = time.time()
        self.setpoint = self.park_position
        self.last_direction = 1
        self.speed = 0
        # Flags to describe the wire scanner's state
        self.scanning = False
        self.scan_init = False
        self.moving = False
        # Changes the units from meters to millimeters for associated PVs.
        self.milli_units = LinearTInv(scaler=1e3)
        # Threshold at which below the scanner is set to the set point
        self.tolerance = 1e-3
        # plan to later have wire count as a parameter in va_config
        self.wire_count = self.initial_defaults["wire_count"]
        # Creates flat noise for associated PVs.
        xy_noise = AbsNoise(noise=1e-9)
        pos_noise = AbsNoise(noise=1e-6)
        # Registers the device's PVs with the server. Diagonal wire not supported yet.
        self.register_measurement(WireScanner.x_charge_pv, noise=xy_noise)
        self.register_measurement(WireScanner.y_charge_pv, noise=xy_noise)
        # self.register_measurement(WireScanner.d_charge_pv, noise=xy_noise)
        self.register_measurement(WireScanner.x_avg_pv, noise=xy_noise, transform=self.milli_units)
        self.register_measurement(WireScanner.y_avg_pv, noise=xy_noise, transform=self.milli_units)
        # self.register_measurement(WireScanner.d_avg_pv, noise=xy_noise, transform=self.milli_units)
        self.register_measurement(WireScanner.x_sigma_pv, noise=xy_noise, transform=self.milli_units)
        self.register_measurement(WireScanner.y_sigma_pv, noise=xy_noise, transform=self.milli_units)
        # self.register_measurement(WireScanner.d_sigma_pv, noise=xy_noise, transform=self.milli_units)
        self.register_measurement(WireScanner.x_profile_pv, definition={'count': self.trace_bin_number})
        self.register_measurement(WireScanner.x_axis_pv, transform=self.milli_units, definition={'count': self.trace_bin_number})
        self.register_measurement(WireScanner.y_profile_pv, definition={'count': self.trace_bin_number})
        self.register_measurement(WireScanner.y_axis_pv, transform=self.milli_units, definition={'count': self.trace_bin_number})
        self.register_measurement(WireScanner.x_trace_pv, definition={'count': self.trace_bin_number})
        self.register_measurement(WireScanner.y_trace_pv, definition={'count': self.trace_bin_number})
        self.register_measurement(WireScanner.trace_time_pv, definition={'count': self.trace_bin_number})
        times = np.linspace(0, self.trace_time, self.trace_bin_number)
        self.update_measurement(WireScanner.trace_time_pv, times)
        # self.register_measurement(WireScanner.d_profile_pv, definition={'count': bin_number})
        # self.register_measurement(WireScanner.d_axis_pv, transform=self.milli_units, definition={'count': bin_number})
        # PVs for client I/O
        self.register_readback(WireScanner.speed_pv, transform=self.milli_units)
        self.register_setting(WireScanner.d_dx_pv, default=self.d_dx, transform=self.milli_units)
        self.register_setting(WireScanner.x_dx_pv, default=self.x_dx, transform=self.milli_units)
        self.register_setting(WireScanner.y_dx_pv, default=self.y_dx, transform=self.milli_units)
        self.register_setting(WireScanner.position_pv, default=self.position, transform=self.milli_units)
        self.register_setting(WireScanner.x_start_pv, default=self.x_start, transform=self.milli_units)
        self.register_setting(WireScanner.x_stop_pv, default=self.x_stop, transform=self.milli_units)
        self.register_setting(WireScanner.y_start_pv, default=self.y_start, transform=self.milli_units)
        self.register_setting(WireScanner.y_stop_pv, default=self.y_stop, transform=self.milli_units)
        self.register_setting(WireScanner.d_start_pv, default=self.d_start, transform=self.milli_units)
        self.register_setting(WireScanner.d_stop_pv, default=self.d_stop, transform=self.milli_units)
        self.register_setting(WireScanner.stop_1_pv, default=self.stop_1, transform=self.milli_units)
        self.register_setting(WireScanner.stop_2_pv, default=self.stop_2, transform=self.milli_units)
        self.register_setting(WireScanner.stop_3_pv, default=self.stop_3, transform=self.milli_units)
        self.register_setting(WireScanner.command_pv, default=0)
        self.register_readback(WireScanner.position_readback_pv, WireScanner.position_pv, transform=self.milli_units,
                               noise=pos_noise)
        self.register_parameter(WireScanner.refresh_rate_pv,default=self.refresh_rate)
        # Initialize the individual wires.
        self.wires = [self.Wire(self,axis) for axis in ["Hor", "Ver"]]
        if self.wire_count == 3:
            self.wires.append(self.Wire(self,"Diag"))
    # Sub class for the individual wires. Helps make the amount of wires modular
    class Wire():
        def __init__(self, ws, axis: str):
            self.axis = axis
            self.ws = ws
            self.position = 0
            # This isn't actually used in any calculations
            self.wire_coeff = 1 / math.sqrt(2)
            # initialize scan result arrays.
            self.pos_array = np.zeros(self.ws.trace_bin_number, dtype=float)
            self.charge_array = np.zeros(self.ws.trace_bin_number, dtype=float)
            self.scan_point = 0
            self.config = {
                "Hor": {"offset": ws.x_offset, "coeff": self.wire_coeff, "key_prefix":"x_"},
                "Ver": {"offset": ws.y_offset, "coeff": self.wire_coeff, "key_prefix":"y_"},
                "Diag": {"offset": 0, "coeff": 1.0, "key_prefix": "d_"},
            }
            # Gives easy access to PV and pyorbit keys
            self.prefix = self.config[self.axis]["key_prefix"]
            # These two functions are called on relevant CA events
            self.update_scan_limits()
            self.update_speed()

        # Gets updated scan limits from client
        def update_scan_limits(self):
            self.start = self.ws.get_parameter_value(getattr(self.ws,f"{self.prefix}start_pv"))
            self.stop = self.ws.get_parameter_value(getattr(self.ws,f"{self.prefix}stop_pv"))

        # Gets updated dX spacing from client, then determines new speed for
        # when this wire is actively being scanned. If the wire isn't being
        # scanned, the speed is set to the maximum speed.
        def update_speed(self,value = None):
            if value is None:
                dX = self.ws.get_parameter_value(getattr(self.ws,f"{self.prefix}dx_pv"))
            else:
                dX = value
            new_speed = self.ws.refresh_rate * dX
            if new_speed > self.ws.max_speed:
                new_speed = self.ws.max_speed
            self.speed = new_speed

        # Wire coefficient not used at the moment. Previous code had position of wire linearly scaling with absolute position
        # of scanner, Meaning the wires spread out as they moved away from the beam
        def set_wire_position(self, position: float):
            wire_factors = self.config[self.axis]
            self.position = position + wire_factors["offset"]

        # Makes an artificial square pulse dependent on the charge observed. For now, this just gives us a trace to look at
        # with the client and doesn't provide any real information about the beam.
        def generate_trace(self,amp):
            waveform = np.random.normal(0,1,self.ws.trace_bin_number)
            pulse_start_index = int((1 - self.ws.pulse_width/self.ws.trace_time) * self.ws.trace_bin_number / 2)
            pulse_stop_index = int((1 + self.ws.pulse_width/self.ws.trace_time) * self.ws.trace_bin_number / 2)
            waveform[pulse_start_index:+pulse_stop_index] += amp
            waveform /= self.ws.reduction_factor
            return waveform
    # Enum storing the possible commands we accept from the client
    class Command(IntEnum):
        Move = 7
        Scan = 21
        Park = 5
        Halt = 8
        Abort = 2
        Idle = 0

    # Function to find the position of the virtual wire using time of flight from the previous position and the speed of
    # the wire. (Thomas Bailey:) I moved much of this code into update_readbacks alongside the addition of the event
    # generating loop in the server. This function is still called in slit_va.py so it is left, with the same return
    # value as before
    def get_wire_position(self):
        return self.last_wire_pos

    # Called on every server update.
    def update_readbacks(self):
        # No direct control of speed
        self.update_readback("Speed", self.speed)
        # Determine if we need to move, which direction, and the step size
        current_position = self.last_wire_pos
        setpoint = self.setpoint
        delta_s = setpoint-current_position
        now = time.time()
        delta_t = now - self.last_wire_time
        direction = np.sign(delta_s)
        new_position = current_position + direction * self.speed * delta_t
        # If we are close enough to the setpoint, stop moving and jump to the setpoint
        if abs(delta_s) < self.tolerance:
            new_position = setpoint
            # If we're only moving, and not scanning
            if self.moving:
                self.moving = False
                self.handle_command(self.Command.Idle)
            # If we are moving to the end to start a full scan
            elif self.scan_init and now > self.scan_start:
                self.scan_init = False
                self.scanning = True
                # Then we need to move to the other end
                if setpoint == self.stop_1:
                    self.setpoint =  self.stop_3
                else:
                    self.setpoint =self.stop_1
            # If we finished the scan
            elif self.scanning and now > self.scan_start + 1:
                self.scanning = False
                self.handle_command(self.Command.Idle)
                self.server_setting_override(self.command_pv, self.Command.Idle)
            self.update_readback(self.position_readback_pv,new_position)
        else:
            self.update_readback(self.position_readback_pv,new_position)
        # set values for next step
        self.last_wire_pos = new_position
        self.last_wire_time = time.time()
        # Finally, update position of individual wires
        for wire in self.wires:
            wire.set_wire_position(new_position)
            wire.update_scan_limits()

    # Called on beam events and determines how much charge each wire sees
    def update_measurements(self, new_params: Dict[str, Dict[str, Any]] = None):
        # load model
        ws_params = new_params[self.model_name]
        # Flag to determine if any wire is being scanned at the current position
        _wire_scanned = False
        # Loop through each wire and update its charge and traces regardless of
        # whether or not it is scanning
        for wire in self.wires:
            position = self.last_wire_pos
            config = wire.config[wire.axis]
            hist = ws_params[getattr(self, f"{config["key_prefix"]}hist_key")]
            axis = hist[:, 0]
            profile = hist[:, 1]
            wire_pos = wire.position
            key = config["key_prefix"]
            value = np.interp(wire_pos, axis, profile, left=0, right=0)
            self.update_measurement(getattr(self, f"{key}charge_pv"), value)
            self.update_measurement(getattr(self, f"{key}trace_pv"),
                                    wire.generate_trace(value))
            # If we are scanning we will write to the result PVs too
            if self.scanning:
                if position > wire.start and position < wire.stop:
                    _wire_scanned = True
                    self.speed = wire.speed
                    self.update_readback("Speed", wire.speed)
                    wire.pos_array[wire.scan_point] = position
                    wire.charge_array[wire.scan_point] = value
                    self.update_measurement(getattr(self,f"{key}profile_pv"), wire.charge_array)
                    self.update_measurement(getattr(self,f"{key}axis_pv"), wire.pos_array)
                    self.update_measurement(getattr(self,f"{key}avg_pv"), ws_params[getattr(self,f"{key}avg_key")])
                    self.update_measurement(getattr(self,f"{key}sigma_pv"), ws_params[getattr(self,f"{key}sigma_key")])
                    wire.scan_point += 1
                # If we aren't scanning we will reset the result arrays so they're ready next time
                else:
                    wire.pos_array = np.zeros(self.trace_bin_number)
                    wire.charge_array = np.zeros(self.trace_bin_number)
                    wire.scan_point = 0
                # If we didn't scan a wire at all, we can speed up a little bit
                if not _wire_scanned:
                    self.speed = self.max_speed
    # Called whenever a client updates a PV
    def handle_ca_event(self,attr = None, value = None):
        if attr is None:
            return
        if attr == self.position_pv:
            self.setpoint = value/1000
        if attr == "Command":
            value = int(value)
            self.handle_command(value)
        if "dX" in attr:
            axis = attr.split('_')[0]
            for wire in self.wires:
                if wire.axis == axis:
                    wire.update_speed(value/1000)
        else:
            return
    # distributes recieved commands to their respective functions
    def handle_command(self, value:int) -> None:
        cmd = self.Command(value)
        match cmd:
            case self.Command.Move:
                self.do_move()
            case self.Command.Scan:
                self.do_scan()
            case self.Command.Park:
                self.do_park()
            case self.Command.Halt:
                self.do_halt()
            case self.Command.Idle:
                self.do_idle()
    # Only getting the wire from A-->B, no scanning involved
    def do_move(self):
        self.moving = True
        self.scan_init = False
        self.scanning = False
        self.speed = self.max_speed
    # Performing a scan
    def do_scan(self):
        # Don't interrupt an active scan
        if self.scanning:
            return
        # Figure out which endpoint is closer and move to it
        self.speed = self.max_speed
        self.scan_init = True
        for wire in self.wires:
            wire.pos_array = np.zeros(self.trace_bin_number)
            wire.charge_array = np.zeros(self.trace_bin_number)
            wire.scan_point = 0
        lo = self.stop_1
        hi = self.stop_3
        midpoint = (lo+hi)/2
        scan_init = time.time()
        current_position = self.get_parameter_value(self.position_readback_pv)
        self.update_readback(self.speed_pv, self.max_speed)
        # Once we start towards one end, determine how long it will take to get there
        if current_position > midpoint:
            self.setpoint = hi
        else:
            self.setpoint = lo
        self.server_setting_override(self.position_pv, self.setpoint)
        self.update_setting(self.position_pv, self.setpoint)
        self.scan_start = scan_init + abs(current_position - self.setpoint) / self.max_speed + 0.25

    # Stop immediately and move to the park position
    def do_park(self):
        self.moving = True
        self.scanning = False
        self.speed = self.max_speed
        self.setpoint = self.park_position
    # stop immediately. In the future this could have other actions performed too
    def do_halt(self):
        self.do_idle()
    def do_idle(self):
        self.moving = False
        self.scanning = False
        self.scan_init = False
        self.speed = 0


class Screen(Device):
    # EPICS PV names
    x_profile_pv = 'resultsHorProf'  # [au]
    y_profile_pv = 'resultsVerProf'  # [au]
    image_pv = 'resultsImg'  # [au]
    image_noise = 5  # [au]
    x_axis_pv = 'resultsHorProfX'  # [mm]
    y_axis_pv = 'resultsVerProfX'  # [mm]

    # PyORBIT parameter keys
    hist_key = 'xy_histogram'  # [au]
    x_axis_key = 'x_axis'  # [m]
    y_axis_key = 'y_axis'  # [m]
    x_key = 'x_avg'  # [m]
    y_key = 'y_avg'  # [m]

    def __init__(self, name: str, model_name: str = None, x_pixels: int = 600, y_pixels: int = 960,
                 x_scale=100, y_scale=100):
        if model_name is None:
            self.model_name = name
        else:
            self.model_name = model_name
        super().__init__(name, self.model_name)

        x_pixels = x_pixels
        y_pixels = y_pixels
        x_scale = x_scale
        y_scale = y_scale

        # Define new grid for higher resolution
        self.x_axis_new = np.linspace(-x_scale / 2, x_scale / 2, x_pixels)
        self.y_axis_new = np.linspace(-y_scale / 2, y_scale / 2, y_pixels)

        # Creates flat noise for associated PVs.
        self.image_noise = PosNoise(noise=Screen.image_noise, count=(y_pixels, x_pixels))

        signal_max = 254
        self.signal_normalize = NormalizePeak(max_value=signal_max)

        # Registers the device's PVs with the server.
        self.register_measurement(Screen.x_profile_pv, definition={'count': x_pixels})
        self.register_measurement(Screen.y_profile_pv, definition={'count': y_pixels})
        self.register_measurement(Screen.image_pv, definition={'type': 'char', 'count': x_pixels * y_pixels})

        self.register_readback(Screen.x_axis_pv, definition={'count': x_pixels})
        self.register_readback(Screen.y_axis_pv, definition={'count': y_pixels})

    # Updates the measurement values on the server. Needs the model key associated with its value and the new value.
    # This is where the measurement PV name is associated with it's model key.
    def update_measurements(self, new_params: Dict[str, Dict[str, Any]] = None):
        screen_params = new_params[self.model_name]
        xy_hist = screen_params[Screen.hist_key]
        x_axis = screen_params[Screen.x_axis_key] * 1000
        y_axis = screen_params[Screen.y_axis_key] * 1000

        # Calculate bin centers
        x_centers = (x_axis[:-1] + x_axis[1:]) / 2
        y_centers = (y_axis[:-1] + y_axis[1:]) / 2

        # Create linearly interpolated function
        interp_func = interp2d(x_centers, y_centers, xy_hist, kind='linear', fill_value=False)

        # Interpolate histogram to higher resolution
        xy_hist_new = interp_func(self.x_axis_new, self.y_axis_new)
        xy_hist_new = self.image_noise.add_noise(xy_hist_new)
        xy_hist_new = self.signal_normalize.raw(xy_hist_new)

        x_profile = np.sum(xy_hist_new, axis=0)
        y_profile = np.sum(xy_hist_new, axis=1)
        image_list = xy_hist_new.flatten()

        self.update_measurement(Screen.image_pv, image_list)
        self.update_measurement(Screen.x_profile_pv, x_profile)
        self.update_measurement(Screen.y_profile_pv, y_profile)

    def update_readbacks(self):
        self.update_readback(Screen.x_axis_pv, self.x_axis_new)
        self.update_readback(Screen.y_axis_pv, self.y_axis_new)


class Quadrupole_Power_Supply(Device):
    # EPICS PV names
    field_set_pv = 'B_Set'  # [T/m]
    field_readback_pv = 'B'  # [T/m]
    field_noise = 1e-6  # [T/m]

    book_pv = 'B_Book'

    def __init__(self, name: str, init_field=None):
        super().__init__(name)

        field_noise = AbsNoise(noise=1e-6)

        # Registers the device's PVs with the server.
        self.register_setting(Quadrupole_Power_Supply.field_set_pv, default=init_field)
        self.register_readback(Quadrupole_Power_Supply.field_readback_pv, Quadrupole_Power_Supply.field_set_pv,
                               noise=field_noise)

        self.register_readback(Quadrupole_Power_Supply.book_pv, Quadrupole_Power_Supply.field_set_pv)


class Quadrupole_Power_Shunt(Device):
    # EPICS PV names
    field_set_pv = 'B_Set'  # [T/m]
    field_readback_pv = 'B'  # [T/m]
    field_noise = 1e-6  # [T/m]

    book_pv = 'B_Book'

    def __init__(self, name: str, init_field=None):
        super().__init__(name)

        field_noise = AbsNoise(noise=1e-6)

        # Registers the device's PVs with the server.
        self.register_setting(Quadrupole_Power_Shunt.field_set_pv, default=init_field)
        self.register_readback(Quadrupole_Power_Shunt.field_readback_pv, Quadrupole_Power_Shunt.field_set_pv,
                               noise=field_noise)

        self.register_readback(Quadrupole_Power_Shunt.book_pv, Quadrupole_Power_Shunt.field_set_pv)


class Bend_Power_Supply(Device):
    # EPICS PV names
    field_set_pv = 'B_Set'  # [T/m]
    field_readback_pv = 'B'  # [T/m]
    field_noise = 1e-6  # [T/m]

    book_pv = 'B_Book'

    def __init__(self, name: str, init_field=None):
        super().__init__(name)

        field_noise = AbsNoise(noise=1e-6)

        # Registers the device's PVs with the server.
        self.register_setting(Bend_Power_Supply.field_set_pv, default=init_field)
        self.register_readback(Bend_Power_Supply.field_readback_pv, Bend_Power_Supply.field_set_pv, noise=field_noise)

        self.register_readback(Bend_Power_Supply.book_pv, Bend_Power_Supply.field_set_pv)


class Corrector_Power_Supply(Device):
    # EPICS PV names
    field_set_pv = 'B_Set'  # [T/m]
    field_readback_pv = 'B'  # [T/m]
    field_noise = 1e-6  # [T/m]

    # Initial field limits
    field_high_limit_pv = 'B_Set.HOPR'
    field_low_limit_pv = 'B_Set.LOPR'
    field_limits = [-0.1, 0.1]  # [T]

    book_pv = 'B_Book'

    def __init__(self, name: str, init_field=None):
        super().__init__(name)

        field_noise = AbsNoise(noise=1e-6)

        # Registers the device's PVs with the server.
        self.register_setting(Corrector_Power_Supply.field_set_pv, default=init_field)
        self.register_readback(Corrector_Power_Supply.field_readback_pv, Corrector_Power_Supply.field_set_pv,
                               noise=field_noise)

        self.register_setting(Corrector_Power_Supply.field_high_limit_pv,
                              default=Corrector_Power_Supply.field_limits[1])
        self.register_setting(Corrector_Power_Supply.field_low_limit_pv, default=Corrector_Power_Supply.field_limits[0])

        self.register_readback(Corrector_Power_Supply.book_pv, Corrector_Power_Supply.field_set_pv)
