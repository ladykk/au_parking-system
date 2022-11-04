import firebase_admin
from firebase_admin import credentials, db, firestore

cred = credentials.Certificate('configs/serviceAccountKey.json')

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://au-parking-default-rtdb.asia-southeast1.firebasedatabase.app/',
})

TempDb = db
Db: firestore.firestore.Client = firestore.client()
