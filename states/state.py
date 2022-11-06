from datetime import datetime
from threading import Thread, Event
from utils.time import datetime_now, seconds_from_now
from utils.logger import getLogger
from firebase import TempDb
from firebase_admin.db import Event as dbEvent
from deepdiff import DeepDiff
from controller import ControllerServer
from alpr import ALPR


class State(object):

    def __init__(self, name: str, source='0', init_state: str = 'init'):
        # > Local variables
        self.name = name
        self._logger = getLogger(f'State.{self.name.capitalize()}')
        self.current_state = init_state
        self.prev_state = ''
        self.next_state = ''
        self.enter_timestamp = datetime.now()
        self.info = {}

        # > Controller
        self.controller = ControllerServer(self.name)

        # # > ALPR
        self.alpr = ALPR(self.name, source)

        # > Database
        self._connected_timestamp = datetime.now()
        self._status = {}
        self._command = ''
        self._db_ref = TempDb.reference(f'{self.name}/state')
        self._db_ref.child("status").listen(self._db_status_callback)
        self._db_ref.child("command").listen(self._db_command_callback)

        # > Thread
        self._thread = Thread(target=self._process, daemon=True)
        self._stop_event = Event()

    # > Database functions
    def _format_status_db(self):
        format = {
            'current_state': self.current_state,
            'prev_state': self.prev_state,
            'next_state': self.next_state,
            'enter_timestamp': self.enter_timestamp.strftime("%d/%m/%Y %H:%M:%S"),
        }
        if len(self.info) != 0:
            format.update({'info': self.info})
        return format

    def _is_status_difference(self):
        return len(DeepDiff(self._format_status_db(), self._status)) != 0

    def _db_status_callback(self, event: dbEvent):
        self._status = event.data

    def _db_command_callback(self, event: dbEvent):
        self._command = event.data if event.data else ''

    # > State utilities
    def seconds_from_now(self, seconds: int):
        return seconds_from_now(self.enter_timestamp, seconds)

    # > State functions
    def start(self):
        if self._thread.is_alive():
            return self._logger.warning("state thread is already running.")
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def is_running(self):
        self._thread.is_alive()
    # > State logic functions

    def _process(self):
        self._logger.info(f'{self.name.capitalize()} State has started.')

        # initialize value in the database.
        self._db_ref.child("status").set(self._format_status_db())
        new_datetime, new_datetime_string = datetime_now()
        self._connected_timestamp = new_datetime
        self._db_ref.child("connected_timestamp").set(new_datetime_string)
        self._db_ref.child("command").set(self._command)

        while not self._stop_event.isSet():
            self._update_state()
            self._process_state()
            self._command_exec()
        self._logger.info(f'{self.name.capitalize()} State has stopped.')

    def _update_state(self):
        if self.next_state != '':
            if self.next_state != self.current_state:
                init_method = "_init_" + self.next_state
                end_method = "_end_" + self.current_state

                if hasattr(self, end_method):
                    getattr(self, end_method)()

                self.prev_state = self.current_state
                self.current_state = self.next_state
                self.next_state = ''
                self.enter_timestamp = datetime.now()
                self._db_ref.child("status").set(self._format_status_db())

                if hasattr(self, init_method):
                    getattr(self, init_method)()

        if self._is_status_difference():
            self._db_ref.child("status").set(self._format_status_db())

        if seconds_from_now(self._connected_timestamp, 5):
            new_datetime, new_datetime_string = datetime_now()
            self._connected_timestamp = new_datetime
            self._db_ref.child("connected_timestamp").set(new_datetime_string)

    def _process_state(self):
        if hasattr(self, "_" + self.current_state):
            getattr(self, "_" + self.current_state)()

    def _command_exec(self):
        if self._command != '':
            input = self._command.split(':')
            if hasattr(self, f'_c_{input[0]}()'):
                if input[1]:
                    getattr(self, f'_c_{input[0]}()')(input[1])
                else:
                    getattr(self, f'_c_{input[0]}()')()
            self._db_ref.child('command').set('')
