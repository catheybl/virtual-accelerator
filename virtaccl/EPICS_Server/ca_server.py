import os
import sys
from threading import Thread
from datetime import datetime
from time import sleep
from math import floor
from typing import Any, Dict

from virtaccl.server import Server
from virtaccl.virtual_accelerator import VA_Parser


def add_epics_arguments(va_parser: VA_Parser) -> VA_Parser:
    # Set the print_settings help description to specify 'PVs' instead of server keys.
    va_parser.edit_argument('--print_settings', {'help': "Will only print setting PVs. Will NOT run "
                                                         "the virtual accelerator."})

    # Number (in seconds) that determine some delay parameter in the server. Not exactly sure how it works, so use at
    # your own risk.
    va_parser.add_server_argument('--ca_proc', default=0.1, type=float,
                                  help='Number (in seconds) that determine some delay parameter in the server. Not '
                                       'exactly sure how it works, so use at your own risk.')

    va_parser.remove_argument('--print_server_keys')
    va_parser.add_va_argument('--print_pvs', dest='print_server_keys', action='store_true',
                              help="Will print all server PVs. Will NOT run the virtual accelerator.")
    return va_parser


class EPICS_Server(Server):
    def __init__(self, prefix='', process_delay=0.1):
        super().__init__()
        self.prefix = prefix
        self.driver = None
        self.process_delay = process_delay

    def _CA_events(self, server):
        while True:
            server.process(self.process_delay)

    def set_parameter(self, reason: str, value: Any, timestamp: datetime = None):
        if timestamp is not None:
            timestamp = self.driver.to_epics_timestamp(timestamp)
        self.driver.setParam(reason, value, timestamp)

    def get_parameter(self, reason: str) -> Any:
        return self.driver.getParam(reason)

    def update(self):
        if self.driver is not None:
            self.driver.updatePVs()

    def start(self):
        os.environ['EPICS_CA_MAX_ARRAY_BYTES'] = '10000000'
        server = SimpleServer()
        server.createPV(self.prefix, self.parameter_db)
        self.driver = TDriver()
        tid = Thread(target=self._CA_events, args=(server,))

        # So it will die after main thread is gone
        tid.setDaemon(True)
        tid.start()
        self.run()

    def stop(self):
        # it's unclear how to gracefully stop the server
        sleep(1)

    def __str__(self):
        return 'Following PVs are registered:\n' + '\n'.join([f'{self.prefix}{k}' for k in self.parameter_db.keys()])

    def run(self):
        pass


