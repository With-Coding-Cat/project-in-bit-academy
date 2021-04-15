import pymongo
from pymongo import MongoClient

def Createmongodb(): 
    client = pymongo.MongoClient("mongodb+srv://ted001:ted0014758@cluster0.stzaz.mongodb.net/Car_plate_DB?retryWrites=true&w=majority")    
    if (client.Car_plate_DB == None):              # 동일한 이름의 데이터베이스가 없을 때, 데이터베이스 생성 
        db = client.Car_plate_DB
        db = client["Car_plate_DB"]     
    else:                                           # 동일한 이름의 데이터베이스가 있을 때, 그 데이터베이스 불러오기
        db = client['{}'.format("Car_plate_DB")]
        
    if (db.tables == None):                # 동일한 이름의 collection이 없을 때, collection 생성
        collection = db.tables
        collection = db["tables"]
    else:                                           # 동일한 이름의 collection이 있을 때, 그 collection 불러오기
        collection = db["tables"]

    if (db.Find_collection == None):               # 동일한 이름의 collection이 없을 때, collection 생성
        collection = db.Find_collection
        collection = db["Find_collection"]
    else:                                           # 동일한 이름의 collection이 있을 때, 그 collection 불러오기
        collection = db["Find_collection"]


def MongoDB(video1, raw_image, image, time, time_raw, plate1):
    client = pymongo.MongoClient("mongodb+srv://ted001:ted0014758@cluster0.stzaz.mongodb.net/Car_plate_DB?retryWrites=true&w=majority")
    db = client["Car_plate_DB"]               # 데이터베이스 불러오기
    collection = db["tables"]         # collection 불러오기
    id_num = 1
    collection.insert({                                         # collection에 데이터 넣기
                  "Id":"{}".format(id_num)                      # 데이터를 넣을 때 +=1 필요
                  "original video":"{}".format(video1),         # 원본 비디오 주소
                  "raw image":"{}".format(raw_image),         # detecting된 비디오 주소
                  "image":"{}".format(image),                   # 이미지 주소
                  "time":"{}".format(time),
                  "time_raw":int("{}".format(time_raw)),                     # 해당 이미지가 나타난 비디오의 시간
                  "car license plate":"{}".format(plate1)})


def DuplicateDB(duplicated_plate):
    client = pymongo.MongoClient("mongodb+srv://ted001:ted0014758@cluster0.stzaz.mongodb.net/Car_plate_DB?retryWrites=true&w=majority")
    db = client["Car_plate_DB"]                # 데이터베이스 불러오기
    collection = db["tables"]         # collection 불러오기

    duplicate_plate = db.tables.find_one({"car license plate":"{}".format(duplicated_plate)})    # 해당 번호판 정보가 데이터베이스에 있는지 조회
    if duplicate_plate == None:     # 중복값이 없다면 True
        return True
    else:
        return False
        
        


def Time_DuplicateDB(time_duplicated_plate, seconds):
    client = pymongo.MongoClient("mongodb+srv://ted001:ted0014758@cluster0.stzaz.mongodb.net/Car_plate_DB?retryWrites=true&w=majority")
    db = client["Car_plate_DB"]                # 데이터베이스 불러오기
    collection = db["tables"]         # collection 불러오기
    
    time_duplicate_data = db.tables.find_one({"car license plate":"{}".format(time_duplicated_plate), "time_raw" : {'$gte' : seconds - 10, '$lte':seconds}})
    # time_raw -10 부터 time_raw까지 해당 번호판 데이터가 있는지 조회
    
    if time_duplicate_data == None:
        return True    # 없으면 True
    else:
        return False
    


def TrackingDB(plate_input):
    client = pymongo.MongoClient("mongodb+srv://ted001:ted0014758@cluster0.stzaz.mongodb.net/Car_plate_DB?retryWrites=true&w=majority")
    db = client["Car_plate_DB"]                # 데이터베이스 불러오기
    collection = db["Find_collection"]         # collection 불러오기

    collection.insert({"car license plate":"{}".format(plate_input)})          # 찾는 번호판 db2에 입력
    
    collection = db["tables"]          # collection 불러오기
    match_data = db.All_collection.find_one({"car license plate":"{}".format(plate_input)})    # 찾는 번호판 데이터를 db1에서 호출
    
    if match_data == None:    # 번호판 호출했을 때 없다면 False 있다면 True
        return False
    else:
        return True