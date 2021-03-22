import pymongo
from pymongo import MongoClient

def Createmongodb(): 
    client = pymongo.MongoClient("mongodb+srv://ted001:ted0014758@cluster0.stzaz.mongodb.net/Car_plate_DB?retryWrites=true&w=majority")
    db = client.Car_plate_DB
    
    if (client.Car_plate_DB == None):              # 동일한 이름의 데이터베이스가 없을 때, 데이터베이스 생성 
        db = client.Car_plate_DB
        db = client['{}'.format("Car_plate_DB")]     
    else:                                           # 동일한 이름의 데이터베이스가 있을 때, 그 데이터베이스 불러오기
        db = client['{}'.format("Car_plate_DB")]
        
    if (db.All_collection == None):                # 동일한 이름의 collection이 없을 때, collection 생성
        collection = db.All_collection
        collection = db['{}'.format("All_collection")]
    else:                                           # 동일한 이름의 collection이 있을 때, 그 collection 불러오기
        collection = db['{}'.format("All_collection")]

    if (db.Find_collection == None):               # 동일한 이름의 collection이 없을 때, collection 생성
        collection = db.Find_collection
        collection = db['{}'.format("Find_collection")]
    else:                                           # 동일한 이름의 collection이 있을 때, 그 collection 불러오기
        collection = db['{}'.format("Find_collection")]

def MongoDB(video1, video2, image, time, plate1):
    client = pymongo.MongoClient("mongodb+srv://ted001:ted0014758@cluster0.stzaz.mongodb.net/Car_plate_DB?retryWrites=true&w=majority")
    db = client["{}".format("Car_plate_DB")]               # 데이터베이스 불러오기
    collection = db["{}".format("All_collection")]         # collection 불러오기
    
    collection.insert({                                         # collection에 데이터 넣기
                  "original video":"{}".format(video1),         # 원본 비디오 주소
                  "detected video":"{}".format(video2),         # detecting된 비디오 주소
                  "image":"{}".format(image),                   # 이미지 주소
                  "time":"{}".format(time),                     # 해당 이미지가 나타난 비디오의 시간
                  "car license plate":"{}".format(plate1)})

def TrackingDB(plate_input):
    client = pymongo.MongoClient("mongodb+srv://ted001:ted0014758@cluster0.stzaz.mongodb.net/Car_plate_DB?retryWrites=true&w=majority")
    db = client["{}".format("Car_plate_DB")]                # 데이터베이스 불러오기
    collection = db["{}".format("Find_collection")]         # collection 불러오기

    collection.insert({"car license plate":"{}".format(plate_input)})          # 찾는 번호판 db2에 입력
    
    collection = db["{}".format("All_collection")]          # collection 불러오기
    match_data = db.All_collection.find({"car license plate":"{}".format(plate_input)})    # 찾는 번호판 데이터를 db1에서 호출
    
    if match_data == None:    # 번호판 호출했을 때 없다면 False 있다면 True
        return False
    else:
        return True