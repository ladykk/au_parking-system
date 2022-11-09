from state import State
from transaction import Transaction


class EntranceState(State):

    def __init__(self):
        super().__init__('entrance', init_state='idle',
                         source="rtsp://admin:a1234567@10.0.0.100:554/Streaming/Channels/101/")
        self.alpr.start()

    # [S0]: Idle
    def _init_idle(self):  # > Entry
        # Clear keys on ALPR.
        self.alpr.clear()
        # Close barricade on Controller.
        self.controller.close_barricade()
        # Clear state info.
        self.info.clear()

    def _idle(self):  # > Logic
        # > Next state
        # 1.ALPR detected -> [S1:Detect]
        if self.alpr.is_detect():
            self.next_state = "detect"
            return

    # [S1]: Detect.
    def _detect(self):  # > Logic
        # Clear ALPR if button is pressed.
        if self.controller.k_button:
            self.alpr.clear()

        # > Next state
        # 1.Hand hovered on Controller -> [S2:Process]
        if self.controller.k_hover:
            self.info = {"license_number", self.alpr.candidate_key()}
            self.next_state = "process"
            return
        # 2.No action after 30 seconds -> [S0:Idle]
        if self.seconds_from_now(30):
            self.next_state = "idle"
            return

    # [S2]: Process.
    def _init_process(self):  # > Entry
        # Get license number.
        license_number = self._info.get("license_number", None)
        # 1.No license number -> [S4:Failed]
        if license_number is None:
            self.info = {"reason": "No license number to add."}
            self.next_state = "failed"
            return
        # Add transaction.
        success, tid = Transaction.add(self.alpr)
        # 2.Transaction added -> [S3:Success]
        if success:
            self.info = {"tid": tid}
            self.next_state = "success"
            return
        # 3.Has previous transaction -> [S4:Failed]
        if tid is not None:
            self.info = {
                "reason": "Previous transaction has an issue.", "tid": tid}
            self.next_state = "failed"
            return
        # 4.No previous transaction -> [S4: Failed]
        else:
            self.info = {"reason": "Cannot add transaction."}
            self.next_state = "failed"
            return

    # [S3]: Success.
    def _init_success(self):  # > Entry
        # Open barricade.
        self.controller.open_barricade()
        # Create is_car_pass.
        self.info.update({"is_car_pass": False})

    def _success(self):  # > Logic
        # Update is_car_pass when detected car at first time.
        if self.controller.p_has_car and self.info.get("is_car_pass", False):
            self.info.update({"is_car_pass": True})

        # > Next state
        # 1. Car has pass and not detected car.
        if self.info.get("is_car_pass", False) and not self.controller.p_has_car:
            self.next_state = "idle"
            return

    # [S4]: Failed.
    def _failed(self):  # > Logic
        # > Next state
        # 1.After 10 seconds and not have previous issue -> [S0:Idle]
        if self.seconds_from_now(10) and self.info.get("tid", None) is None:
            self.next_state = "idle"
            return
        # 2.After issue solve or timeout 120 seconds  -> [S0:Idle]
        if Transaction.get(self.info.get("tid", '')) is None or self.seconds_from_now(120):
            self.next_state = "idle"
            return
        # 3.Button pressed on Controller -> [S0:Idle]
        if self.controller.k_button:
            self.next_state = "idle"
            return


def main():
    entrance = EntranceState()
    entrance.start()


if __name__ == "__main__":
    main()
