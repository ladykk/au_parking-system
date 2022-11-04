from modules.state import State
from modules.transaction import Transaction


class ExitState(State):

    def __init__(self):
        super().__init__('exit', init_state='idle')
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

    # [S1]: Detect
    def _init_detect(self):  # > Entry
        # Create checked_license_numbers.
        self.info.update({"checked_license_numbers": list()})

    def _detect(self):  # > Logic
        # Check for license_number in transaction.
        for license_number in self.alpr.keys():
            # Continue if already checked.
            if license_number in self.info.get("checked_license_numbers"):
                continue
            # Check is license_number exit.
            is_exists, tid = Transaction.is_license_number_exists(
                license_number)
            if is_exists:  # Break if license_number exists.
                break
            else:  # Append to checked_license_numbers if not found.
                self.info["checked_license_numbers"].append(license_number)

        # > Next state
        # 1.Found tid -> [S2:Detect]
        if tid is not None:
            self.next_state = "get"
            self.info = {"tid": tid}
            return
        # 2.Hand hovered on Controller -> [S0:Idle]
        if self.controller.k_hover:
            self.next_state = "idle"
            return
        # 3.If not found all -> [S5:Failed]
        if len(self.alpr.keys()) == len(self.info.get("checked_license_numbers", [])):
            self.info = {"reason": "Not found license_number in the system."}
            self.next_state = "failed"
            return
        # 4.No action after 30 seconds -> [S0:Idle]
        if self.seconds_from_now(30):
            self.next_state = "idle"
            return

    # [S2]: Get transaction
    def _init_get(self):  # > Entry
        if self.info.get("tid", None) is None:  # No tid -> [S5: Failed]
            self.info = {"reason": "No transaction id."}
            self.next_state = "failed"
            return
        # Get transaction
        transaction = Transaction.get(self.info.get("transaction"))
        if transaction is None:  # No transaction -> [S5:Failed]
            self.info = {"reason": "Transaction not exists in the system."}
            self.next_state = "failed"
            return
        if transaction.is_paid():  # Transaction paid -> [S3: Success]
            self.next_state = "success"
            return
        else:
            self.next_state = "payment"
            return

    # [S3]: Success
    def _init_success(self):  # > Entry
        # Get transaction
        transaction = Transaction.get(self.info.get("transaction"))
        if transaction is None:  # No transaction -> [S5:Failed]
            self.info = {"reason": "Transaction not exists in the system."}
            self.next_state = "failed"
            return
        # Close transaction.
        transaction.closed()
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

    # [S4]: Payment
    def _payment(self):  # > Logic
        if self.info.get("tid", None) is None:  # No tid -> [S5: Failed]
            self.info = {"reason": "No transaction id."}
            self.next_state = "failed"
            return
        # Get transaction
        transaction = Transaction.get(self.info.get("transaction"))
        if transaction is None:  # No transaction -> [S5:Failed]
            self.info = {"reason": "Transaction not exists in the system."}
            self.next_state = "failed"
            return
        if transaction.is_paid():  # Transaction paid -> [S3: Success]
            self.next_state = "success"
            return
        # Transaction unpaid after 120 seconds -> [S4: Failed]
        if self.seconds_from_now(120):
            self.info = {"reason": "Payment timeout."}
            self.next_state = "failed"
            return

    # [S4]: Failed.
    def _failed(self):  # > Logic
        # > Next state
        # 1.After 10 seconds and not have previous issue -> [S0:Idle]
        if self.seconds_from_now(10):
            self.next_state = "idle"
            return
        # 2.Button pressed on Controller -> [S0:Idle]
        if self.controller.k_button:
            self.next_state = "idle"
            return


def main():
    exit = ExitState()
    exit.start()


if __name__ == "__main__":
    main()
