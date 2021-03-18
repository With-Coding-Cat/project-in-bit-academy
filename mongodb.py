import pymongo
from pymongo import MongoClient

def Createmongodb(database_name, collection_name, collection_name2):
    client = MongoClient()
    
    if (client.database_name == None):              # 동일한 이름의 데이터베이스가 없을 때, 데이터베이스 생성 
        db = client.database_name
        db = client['{}'.format(database_name)]     
    else:                                           # 동일한 이름의 데이터베이스가 있을 때, 그 데이터베이스 불러오기
        db = client['{}'.format(database_name)]
        
        
    if (db.collection_name == None):                # 동일한 이름의 collection이 없을 때, collection 생성
        collection = db.collection_name
        collection = db['{}'.format(collection_name)]
    else:                                           # 동일한 이름의 collection이 있을 때, 그 collection 불러오기
        collection = db['{}'.format(collection_name)]
        
    if (db.collection_name2 == None):               # 동일한 이름의 collection이 없을 때, collection 생성
        collection = db.collection_name2
        collection = db['{}'.format(collection_name2)]
    else:                                           # 동일한 이름의 collection이 있을 때, 그 collection 불러오기
        collection = db['{}'.format(collection_name2)]


def MongoDB(video1, video2, image, time, plate1):
    db = client['car_database']             # 데이터베이스 불러오기
    collection = db["collection_1"]         # collection 불러오기
    
    collection.insert({                                         # collection에 데이터 넣기
                  "original video":"{}".format(video1),         # 원본 비디오 주소
                  "detected video":"{}".format(video2),         # detecting된 비디오 주소
                  "image":"{}".format(image),                   # 이미지 주소
                  "time":"{}".format(time),                     # 해당 이미지가 나타난 비디오의 시간
                  "car license plate":"{}".format(plate1)})     # 번호판

def TrackingDB(plate_input):
    db = client['car_database']             # 데이터베이스 불러오기
    collection = db['collection_2']         # collection 불러오기

    collection.insert({"car license plate":"{}".format(plate_input)})    # 찾는 번호판 db2에 입력
    
    collection = db['collection_1']         # collection 불러오기
    match_data = db.collection_1.find({"car license plate":"{}".format(plate_input)})    # 찾는 번호판 데이터를 db1에서 호출
    
    if match_data == None:    # 번호판 호출했을 때 없다면 False 있다면 True
        print('False')
    else:
        print("True")