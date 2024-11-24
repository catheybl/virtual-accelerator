import os

from pcaspy import Driver
from pcaspy.cas import epicsTimeStamp
from pcaspy import SimpleServer

from threading import Thread
from datetime import datetime
from time import sleep
from math import floor
from typing import Any

from virtaccl.server import Server


class TDriver(Driver):
    def __init__(self):
        Driver.__init__(self)

    def setParam(self, reason, value, timestamp=None):
        super().setParam(reason, value)
        if timestamp is not None:
            self.pvDB[reason].time = timestamp

    def to_epics_timestamp(self, t: datetime):
        if t is None:
            return None
        tst = epicsTimeStamp()
        epics_tst = t.timestamp() - 631152000.0
        tst.secPastEpoch = int(floor(epics_tst))
        tst.nsec = int((epics_tst % 1) * 1_000_000_000)
        return tst
