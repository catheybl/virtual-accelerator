import math
from numpy.random import random_sample
from typing import Optional, Union, List, Dict, Any

from .ca_server import Server


class Transform:

    def real(self, x):
        return x

    def raw(self, x):
        return x


class LinearT(Transform):
    def __init__(self, offset=0.0, scaler=1.0, reason_rb=None):
        self._offset = offset
        self._scaler = scaler
        self._reason_rb = reason_rb

    def real(self, x):
        return x / self._scaler - self._offset

    def raw(self, x):
        return (x + self._offset) * self._scaler

    def calculate_rb(self, x):
        return self.raw(x)


class PhaseT(LinearT):
    def __init__(self, noise=0.0, **kw):
        super().__init__(**kw)
        self.noise = noise

    @staticmethod
    def wrap_phase(deg):
        x = deg % 360
        return x - 360 if x > 180 else x

    def raw(self, x):
        return self.wrap_phase(LinearT.raw(self, x))

    def real(self, x):
        return self.wrap_phase(LinearT.real(self, x))


class LinearTInv(Transform):
    def __init__(self, offset=0.0, scaler=1.0, reason_rb=None):
        self._offset = offset
        self._scaler = scaler
        self._reason_rb = reason_rb

    def real(self, x):
        return (x - self._offset) / self._scaler

    def raw(self, x):
        return (x * self._scaler) + self._offset

    def calculate_rb(self, x):
        return self.raw(x)


class PhaseTInv(LinearTInv):
    def __init__(self, noise=0.0, **kw):
        super().__init__(**kw)
        self.noise = noise

    @staticmethod
    def wrap_phase_deg(deg):
        x = deg % 360
        return x - 360 if x > 180 else x

    @staticmethod
    def wrap_phase_rad(rad):
        x = rad % (2 * math.pi)
        return x - 2 * math.pi if x > math.pi else x

    def raw(self, x):
        return self.wrap_phase_deg(LinearTInv.raw(self, x))

    def real(self, x):
        return self.wrap_phase_rad(LinearTInv.real(self, x))


class Noise:
    def add_noise(self, x):
        return x


class AbsNoise(Noise):
    def __init__(self, noise=0.0, **kw):
        super().__init__(**kw)
        self.noise = noise

    def add_noise(self, x):
        return x + self.noise * (random_sample() - 0.5)


class Parameter:
    def __init__(self, reason: str, definition=None, default=0, transform=None, noise=None, name_override: str = None):
        self.reason = reason
        self.definition = definition
        self.default = default
        self.transform, self.noise = self._default(transform, noise)
        self.name_or = name_override

        self.device: Device = None
        self.server: Server = None

        self.setting_param = None

    @classmethod
    def _default(cls, transform, noise):
        # default transformation is identity
        transform = transform if transform else Transform()
        # default noise is zero
        noise = noise if noise else Noise()
        return transform, noise

    def get_pv(self):
        if self.name_or is None:
            return self.device.name + ':' + self.reason
        else:
            return self.name_or + ':' + self.reason

    def get_definition(self):
        return self.definition

    def get_default(self):
        return self.default

    def is_name_override(self):
        if self.name_or:
            return True
        else:
            return False

    def get_param(self):
        if self.server:
            val = self.server.getParam(self.get_pv())
            return self.transform.real(val)

    def set_param(self, value=None, timestamp=None):
        if self.server:
            if value is None:
                value = self.get_param()
            param = self.noise.add_noise(self.transform.raw(value))
            self.server.setParam(self.get_pv(), param, timestamp)

    def get_value(self):
        if self.server:
            return self.server.getParam(self.get_pv())

    def set_value(self, value, timestamp=None):
        if self.server:
            self.server.setParam(self.get_pv(), value, timestamp)


class Device:

    def __init__(self, pv_name: str, model_name: Optional[Union[str, List[str]]] = None):
        self.name = pv_name
        if model_name is None:
            self.model_names = [pv_name]
        elif not isinstance(model_name, list):
            self.model_names = [model_name]
        else:
            self.model_names = model_name

        self.server = None
        self.__db_dictionary__ = {}

        # dictionary stores (definition, default, transform, noise)
        self.settings: {str: Parameter} = {}

        # dictionary stores (definition, transform, noise)
        self.measurements: {str: Parameter} = {}

        # dictionary stores (definition, transform, noise)
        self.readbacks: {str: Parameter} = {}

    def register_measurement(self, reason, definition=None, transform=None, noise=None, name_override=None):
        if definition is None:
            definition = {}
        param = Parameter(reason, definition, transform=transform, noise=noise, name_override=name_override)
        param.device = self
        self.measurements[reason] = param
        return param

    def register_setting(self, reason: str, definition=None, default=0, transform=None, noise=None, name_override=None):
        if definition is None:
            definition = {}
        param = Parameter(reason, definition, default=default, transform=transform, noise=noise,
                          name_override=name_override)
        param.device = self
        self.settings[reason] = param
        return param

    def register_readback(self, reason: str, setting: Parameter, transform=None, noise=None, name_override=None):
        definition = setting.get_definition()
        param = Parameter(reason, definition, transform=transform, noise=noise, name_override=name_override)
        param.device = self
        param.setting_param = setting
        self.readbacks[reason] = param
        return param

    def get_setting(self, reason):
        return self.settings[reason].get_param()

    def get_settings(self) -> Dict[str, Dict[str, Any]]:
        params_dict = {}
        for model_name in self.model_names:
            param_input_dict = {}
            for setting, param in self.settings.items():
                param_value = param.get_param()
                param_input_dict = param_input_dict | {setting: param_value}
            if param_input_dict:
                params_dict = params_dict | {model_name: param_input_dict}
        return params_dict

    def update_measurement(self, reason: str, value=None):
        param = self.measurements[reason]
        param.set_param(value)

    def update_measurements(self, new_measurements: Dict[str, Dict[str, Any]] = None) -> None:
        new_dict = {}
        for model_name, model_dict in new_measurements.items():
            if model_name in self.model_names:
                for param_name, new_value in model_dict.items():
                    if param_name in self.measurements:
                        reason = model_name + ':' + param_name
                        new_dict[reason] = new_value

        for reason, param in self.measurements.items():
            if param.get_pv() in new_dict:
                self.update_measurement(reason, new_dict[param.get_pv()])
            else:
                self.update_measurement(reason)

    def update_readback(self, reason, value=None):
        rb_param = self.readbacks[reason]
        if value is None:
            value = rb_param.setting_param.get_param()
        rb_param.set_param(value)

    def update_readbacks(self):
        for reason, param in self.readbacks.items():
            self.update_readback(reason)

    def get_readback(self, reason):
        return self.readbacks[reason].get_param()

    def reset(self):
        for k, v in self.settings.items():
            v.set_value(v.get_default())

    def build_db(self):
        setting_pvs = {v.get_pv(): v for k, v in self.settings.items()}
        readback_pvs = {v.get_pv(): v for k, v in self.readbacks.items()}
        measurement_pvs = {v.get_pv(): v for k, v in self.measurements.items()}
        full_db = setting_pvs | readback_pvs | measurement_pvs
        return full_db