import sys
import threading
import time
import argparse
import queue
from datetime import datetime
from dataclasses import dataclass
from importlib.metadata import version
from typing import Dict, Any, List, TypeVar, Generic, Optional

from virtaccl.server import Server, not_ctrlc
from virtaccl.beam_line import BeamLine
from virtaccl.model import Model

EPICS_EPOCH_OFFSET = 631152000.0

@dataclass
class Event:
    type: str
    device: Optional[str] = None
    attr: Optional[str] = None
    value: Optional[float] = None
    time: Optional[datetime] = None

class VA_Parser:
    def __init__(self):
        self._va_arguments_: Dict[str, Dict[str, Any]] = {}
        self.model_arguments: Dict[str, Dict[str, Any]] = {}
        self.server_arguments: Dict[str, Dict[str, Any]] = {}
        self.custom_arguments: Dict[str, Dict[str, Any]] = {}
        self.__all_arguments__ = {'va': self._va_arguments_, 'model': self.model_arguments,
                                  'server': self.server_arguments, 'custom': self.custom_arguments}
        self.__all_argument_keys__ = set()

        self.version = version('virtaccl')
        self.description = 'Run the Virac virtual accelerator server.'

        add_va_arguments(self)

    def __find_argument_dict__(self, name) -> Dict[str, Dict[str, Any]]:
        for argument_group, arguments in self.__all_arguments__.items():
            if name in arguments:
                return arguments

    def set_description(self, new_description: str):
        self.description = new_description

    def add_argument(self, *args, **kwargs):
        arg_key = args[0]
        if arg_key in self.__all_argument_keys__:
            print(f'Warning: Argument name "{arg_key}" already exists. Argument not added.')
        else:
            self.custom_arguments[arg_key] = {'positional': args, 'optional': kwargs}
            self.__all_argument_keys__.add(arg_key)

    def add_va_argument(self, *args, **kwargs):
        arg_key = args[0]
        if arg_key in self.__all_argument_keys__:
            print(f'Warning: Argument name "{arg_key}" already exists. Argument not added.')
        else:
            self._va_arguments_[arg_key] = {'positional': args, 'optional': kwargs}
            self.__all_argument_keys__.add(arg_key)

    def add_model_argument(self, *args, **kwargs):
        arg_key = args[0]
        if arg_key in self.__all_argument_keys__:
            print(f'Warning: Argument name "{arg_key}" already exists. Argument not added.')
        else:
            self.custom_arguments[arg_key] = {'positional': args, 'optional': kwargs}
            self.__all_argument_keys__.add(arg_key)

    def add_server_argument(self, *args, **kwargs):
        arg_key = args[0]
        if arg_key in self.__all_argument_keys__:
            print(f'Warning: Argument name "{arg_key}" already exists. Argument not added.')
        else:
            self.custom_arguments[arg_key] = {'positional': args, 'optional': kwargs}
            self.__all_argument_keys__.add(arg_key)

    def remove_argument(self, name: str):
        if name not in self.__all_argument_keys__:
            print(f'Warning: Argument name "{name}" was not found.')
        else:
            arguments = self.__find_argument_dict__(name)
            del arguments[name]
            self.__all_argument_keys__.remove(name)

    def edit_argument(self, name: str, new_options: Dict[str, Any]):
        if name not in self.__all_argument_keys__:
            print(f'Warning: Argument name "{name}" was not found.')
        else:
            arguments = self.__find_argument_dict__(name)
            for option_key, new_value in new_options.items():
                arguments[name]['optional'][option_key] = new_value

    def change_argument_default(self, name: str, new_value: Any):
        arguments = self.__find_argument_dict__(name)
        arguments[name]['optional']['default'] = new_value

    def change_argument_help(self, name: str, new_help: Any):
        arguments = self.__find_argument_dict__(name)
        arguments[name]['optional']['help'] = new_help

    def initialize_arguments(self) -> Dict[str, Any]:
        va_parser = argparse.ArgumentParser(
            description=self.description + ' Version ' + self.version,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        for group_key, argument_group in self.__all_arguments__.items():
            for argument_name, argument_dict in argument_group.items():
                va_parser.add_argument(*argument_dict['positional'], **argument_dict['optional'])
        return vars(va_parser.parse_args())


def add_va_arguments(va_parser: VA_Parser) -> VA_Parser:
    # Number (in Hz) determining the update rate for the virtual accelerator.
    va_parser.add_va_argument('--refresh_rate', default=1.0, type=float,
                              help='Rate (in Hz) at which the virtual accelerator updates.')
    va_parser.add_va_argument('--device_frequency', default=10.0, type=float,
                              help='Rate (in Hz) at which the server updates.')
    va_parser.add_va_argument('--sync_time', dest='sync_time', action='store_true',
                              help="Synchronize timestamps for server parameters.")

    # Desired amount of output.
    va_parser.add_va_argument('--debug', dest='debug', action='store_true',
                              help="Some debug info will be printed.")
    va_parser.add_va_argument('--production', dest='debug', action='store_false',
                              help="DEFAULT: No additional info printed.")

    va_parser.add_server_argument('--print_server_keys', action='store_true',
                                  help="Will print all server keys for the server. Will NOT run the virtual "
                                       "accelerator.")
    va_parser.add_server_argument('--print_settings', action='store_true',
                                  help="Will only print setting keys for the server. Will NOT run the virtual "
                                       "accelerator.")

    return va_parser


# Define a TypeVar constrained to Model
ModelType = TypeVar('ModelType', bound='Model')
ServerType = TypeVar('ServerType', bound='Server')


class VirtualAcceleratorBuilder(Generic[ModelType, ServerType]):
    def __init__(self, model: ModelType, beam_line: BeamLine, server: ServerType, **kwargs):
        self.model = model
        self.beam_line = beam_line
        self.server = server
        self.options = kwargs

    def get_model(self) -> ModelType:
        return self.model

    def get_beamline(self) -> BeamLine:
        return self.beam_line

    def get_server(self) -> ServerType:
        return self.server

    def build(self) -> 'VirtualAccelerator[ModelType, ServerType]':
        return VirtualAccelerator(self.model, self.beam_line, self.server, **self.options)


class VirtualAccelerator(Generic[ModelType, ServerType]):
    def __init__(self, model: ModelType, beam_line: BeamLine, server: ServerType, **kwargs):
        if not kwargs:
            kwargs = VA_Parser().initialize_arguments()

        if kwargs['print_settings']:
            for key in beam_line.get_setting_keys():
                print(key)
            sys.exit()

        if kwargs['print_server_keys']:
            for key in beam_line.get_all_keys():
                print(key)
            sys.exit()

        self.sync_time = kwargs['sync_time']
        self.update_period = 1 / kwargs['refresh_rate']
        self.server_period = 1 / kwargs['device_frequency']

        self.model = model
        self.beam_line = beam_line
        self.server = server

        server_parameters = beam_line.get_server_parameter_definitions()
        server.add_parameters(server_parameters)
        server.add_parameter("VIRAC:beam_time", {
            "value": 0.0,
            "count": 1
        })
        server.add_parameter("VIRAC:beam_state",{
            "type": "enum",
            "value": 1,
            "enums": ["OFF", "ON"]
        })
        beam_line.reset_devices()

        if kwargs['debug']:
            print(server)

        self.track()

    def get_model(self) -> ModelType:
        return self.model

    def get_beamline(self) -> BeamLine:
        return self.beam_line

    def get_server(self) -> ServerType:
        return self.server

    def set_value(self, server_key: str, new_value):
        self.server.set_parameter(server_key, new_value)
        self.track()

    def set_values(self, new_settings: Dict[str, Any]):
        self.server.set_parameters(new_settings)
        self.track()

    def get_value(self, *server_key: str):
        if len(server_key) == 1:
            return self.server.get_parameter(server_key[0])
        else:
            return tuple(self.server.get_parameter(key) for key in server_key)

    def get_values(self, value_keys: List[str] = None) -> Dict[str, Any]:
        if value_keys is not None:
            return_dict = {}
            for key in value_keys:
                return_dict |= {key: self.server.get_parameter(key)}
        else:
            return_dict = self.server.get_parameters()
        return return_dict

    def track(self, timestamp: datetime = None):
        server_parameters = self.server.get_parameters()
        self.beam_line.update_settings_from_server(server_parameters)
        new_optics = self.beam_line.get_model_optics()

        self.model.update_optics(new_optics)
        self.model.track()
        new_measurements = self.model.get_measurements()

        self.beam_line.update_measurements_from_model(new_measurements)
        self.beam_line.update_readbacks()
        new_server_values = self.beam_line.get_parameters_for_server()
        self.server.set_parameters(new_server_values, timestamp=timestamp)
    # self.readback() is copied from self.track(), but removes lines associated
    # with calculating the model optics.
    def readback(self,  timestamp: datetime = None):
        server_parameters = self.server.get_parameters()
        self.beam_line.update_settings_from_server(server_parameters)
        new_measurements = self.model.get_measurements()

        self.beam_line.update_readbacks()
        new_server_values = self.beam_line.get_parameters_for_server()
        self.server.set_parameters(new_server_values, timestamp=timestamp)

    def start_server(self):
        # Event queue for the server to pull from
        self.q = queue.Queue()
        self.server.start(self.q)
        # Thread to handle event processing loop
        self.main_server_thread = threading.Thread(target=self.run_main_thread)
        self.main_server_thread.start()
        print(f"Server started.")
        # Initialize event times
        now = datetime.now()
        now_ts = now.timestamp() - EPICS_EPOCH_OFFSET
        next_beam_time = now_ts + self.update_period
        next_update_time = now_ts + self.server_period
        # Stop generating events once the keyboard interrupt stop event is set.
        # This loop just generates events, beam events are at --refresh-rate,
        # and server updates are at --update-frequency. Ideally server updates
        # happen more frequently than beam events.
        while not_ctrlc():
            now = datetime.now()
            now_ts = now.timestamp()- EPICS_EPOCH_OFFSET
            if now_ts > next_beam_time:
                next_beam_time += self.update_period
                if self.server.get_parameter("VIRAC:beam_state") == 0:
                    continue
                # Beam event generates a new pulse and updates measurements,
                # unchanged from prior virac loop
                beam_event = Event(
                    type="BEAM",
                    time=now
                )
                self.q.put(beam_event)
            # rbk event processes changes to server and device values, but
            # does not generate a new model for a beam pulse
            if now_ts > next_update_time:
                rbk_event = Event(
                    type="RBK",
                    time=now
                )
                self.q.put(rbk_event)
                next_update_time += self.server_period
            time.sleep(.01)

        print('Exiting. Thank you for using our virtual accelerator!')
    def run_main_thread(self):
        while not_ctrlc():
            # check the queue for events
            if self.q.empty():
                time.sleep(0.01)
                continue
            else:
                event = self.q.get(timeout=.2)
                if event.type == "BEAM":
                    self.handle_beam_event(event)
                elif event.type == "RBK":
                    self.handle_rbks_event()
                elif event.type == "CA":
                    self.handle_ca_events(event)
                else:
                    raise ValueError(f"Unknown event type: {event.type}")
                self.server.update()

    # main virac loop from prior version copied into here. Also added a pulse
    # PV that can be read as a callback on clients, with the pulse time. and
    # happens after the beam event is processed, avoiding issues of reading
    # PVs associated with different beam events.
    def handle_beam_event(self, event: Event):
        loop_start_time = time.time()
        now = event.time
        if self.sync_time:
            now = datetime.now()
        # main virac loop.
        self.track(timestamp=now)
        self.server.set_parameter("VIRAC:beam_time", now.timestamp() - EPICS_EPOCH_OFFSET)
        loop_time_taken = time.time() - loop_start_time
        if loop_time_taken > self.update_period:
            print("Warning: Beam event took longer than refresh rate")

    # Process device updates in between beam events.
    def handle_rbks_event(self):
        self.readback()

    # If there's any extra work do be done on a PV update, this is the place to do it.
    def handle_ca_events(self, event: Event):
        device_name = event.device
        attr = event.attr
        value = event.value
        if device_name == "VIRAC":
            if attr == "beam_state":
                self.server.set_parameter("VIRAC:beam_state", value)
            return
        self.beam_line.devices[device_name].handle_ca_event(attr, value)
