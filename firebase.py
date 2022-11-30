import firebase_admin
from firebase_admin import credentials, db, firestore, storage, messaging

cred = credentials.Certificate('configs/serviceAccountKey.json')

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://au-parking-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': "au-parking.appspot.com",
})

TempDb = db
Db: firestore.firestore.Client = firestore.client()
Storage = storage.bucket()
Messaging = messaging


def messagePayload(title, body, type):
    return Messaging.Message(notification={
        "title": title,
        "body": body
    },
        data={
        "type": type
    },
        topic="staffs")
