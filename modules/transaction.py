from datetime import datetime, timedelta
from modules.firebase import Db
from firebase_admin.firestore import firestore


class Transaction(object):

    list = dict()
    transactions_ref = Db.collection("transactions")

    def __init__(
        self,
        tid: str,
        license_number: str,
        timestamp_in: datetime,
        fee: float, status: str,
        paid: float,
        timestamp_out: datetime = None
    ):
        self.tid = tid
        self.license_number = license_number
        self.timestamp_in = timestamp_in
        self.fee = fee
        self.status = status
        self.paid = paid
        self.timestamp_out = timestamp_out

    @staticmethod
    def from_dict(data: dict) -> 'Transaction':
        return Transaction(
            data.get("tid", ""),
            data.get("license_number"),
            data.get("timestamp_in"),
            data.get("fee", 0),
            data.get("status", ""),
            data.get("paid", 0),
            data.get("timestamp_out", None)
        )

    @staticmethod
    def is_license_number_exists(license_number: str):
        for tid, transaction in Transaction.list.items():
            if transaction.license_number == license_number:
                if not transaction.is_out():
                    return True, tid
        return False, None

    @staticmethod
    def is_license_number_unpaid(license_number: str):
        for tid, transaction in Transaction.list.items():
            if transaction.license_number == license_number:
                if not transaction.is_paid():
                    return True, tid
        return False, None

    @staticmethod
    def add(license_number: str):
        is_exists, tid = Transaction.is_license_number_exists(license_number)
        if is_exists:
            return False, None
        is_unpaid, tid = Transaction.is_license_number_unpaid(license_number)
        if is_unpaid:
            return False, tid
        update_time, ref = Transaction.transactions_ref.add(
            {"license_number": license_number, "timestamp_in": firestore.SERVER_TIMESTAMP})
        return True, ref.id

    @staticmethod
    def get(tid: str):
        return Transaction.list.get(tid, None)

    def update(self, data: dict):
        self.tid = data.get("tid", ""),
        self.license_number = data.get("license_number"),
        self.timestamp_in = data.get("timestamp_in"),
        self.fee = data.get("fee", 0),
        self.status = data.get("status", ""),
        self.paid = data.get("paid", 0),
        self.timestamp_out = data.get("timestamp_out", None)

    def is_paid(self):
        return self.status == "Paid"

    def is_out(self):
        return self.timestamp_out is datetime

    def closed(self):
        # Update timestamp_out.
        update_time = Db.collection("transactions").document(self.tid).update(
            {"timestamp_out": firestore.SERVER_TIMESTAMP, "is_edit": True})


def on_transactions_snapshot(collection, changes, read_time):
    for change in changes:
        if change.type.name == "ADDED":
            Transaction.list.update(
                {change.document.id: Transaction.from_dict(change.document.to_dict())})
        elif change.type.name == "MODIFIED":
            transaction = Transaction.list.get(change.document.id, None)
            if transaction is None:
                Transaction.list.update(
                    {change.document.id: Transaction.from_dict(change.document.to_dict())})
            else:
                transaction.update(change.document.to_dict())
        elif change.type.name == "REMOVED":
            Transaction.list.pop(change.document.id, None)


Transaction.transactions_ref.where("timestamp_in", ">=", datetime.now(
) - timedelta(weeks=1)).on_snapshot(on_transactions_snapshot)
