import argparse
import time
from pathlib import Path
import sqlite3
import os
import re
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from models import yolo
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import personal_utils

import string
import torch.nn.functional as F
from OCR.utils import CTCLabelConverter, AttnLabelConverter
from model import Model
from easydict import EasyDict
import os

def detect(opt, OCR_model, OCR_pridiction_opt, ocr_device, converter, save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    ########################임시용############
    #only_once_detected = set()

    if not os.path.exists('parking_lot.db'):
        parking_db = sqlite3.connect('parking_lot.db')
        parking_db.execute('CREATE TABLE car_plate_data(id INTEGER, car_plate TEXT, detected_image_path_in_plate TEXT, time_in TEXT, detected_image_path_out_plate TEXT, time_out TEXT, time_in_sec REAL, time_out_sec REAL, price INTEGER, detected_image_path_in_car TEXT, detected_image_path_out_car TEXT)')
        db_io = parking_db.cursor()
        
        img_name_numbering = 0
    else:
        parking_db = sqlite3.connect('parking_lot.db')
        db_io = parking_db.cursor()
        db_io.execute("SELECT id FROM car_plate_data")
        temp = db_io.fetchall()
        if temp:
            img_name_numbering = temp.pop()[0]
        else:
            img_name_numbering = 0
        del temp

    db_io.execute("SELECT car_plate FROM car_plate_data WHERE detected_image_path_out_car == ' '")
    car_plate_in_parking_lot = db_io.fetchall()
    car_plate_in_parking_lot = [car_plate_name[0] for car_plate_name in car_plate_in_parking_lot]
    car_in_numbering = 0
    car_out_numbering = 0
    ##########################################
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            #save_path2 = str(save_dir / p.stem) + '_test' + str(p.suffix) 테스트해봄
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        #############################################
                        #수정부분####################################
                        #processed_image = personal_utils.take_picture(xyxy, im0, save_dir) #이미지 부분만 잘라내기
                        label = 'Detected'
                        plot_one_box(xyxy, im0, label=label, color=(0,0,255), line_thickness=3)


                        processed_image = personal_utils.take_picture(xyxy, im0)
                        processed_image_height, processed_image_width = processed_image.shape[:2]
                        predicted_plate_sting = personal_utils.predict_plate(OCR_model, OCR_pridiction_opt, processed_image, ocr_device, converter)
                        if len(predicted_plate_sting) >= 7 and re.findall('[0-9]{2}[가-힣]{1}[0-9]{4}', predicted_plate_sting) and predicted_plate_sting[-7:] == re.findall('[0-9]{2}[가-힣]{1}[0-9]{4}', predicted_plate_sting)[0]:
                            #if processed_image_height > 20 and processed_image_width > 70:
                            #if processed_image_height > 25 and processed_image_width > 80:
                            #if processed_image_height > 31 and processed_image_width > 99:
                            if predicted_plate_sting not in car_plate_in_parking_lot:
                                img_name_numbering += 1
                                car_in_numbering += 1
                                car_plate_in_parking_lot.append(predicted_plate_sting)
                                os.makedirs(str(save_dir / 'car_in_plate'), exist_ok=True)
                                image_path = str(save_dir / 'car_in_plate/{}.jpg'.format(car_in_numbering))
                                cv2.imwrite(image_path, processed_image)
                                os.makedirs(str(save_dir / 'car_in_car'), exist_ok=True)
                                image_path2 = str(save_dir / 'car_in_car/{}.jpg'.format(car_in_numbering))
                                cv2.imwrite(image_path2, im0)
                                current_time_sec = time.time()
                                current_time = time.strftime('%Y년 %m월 %d일 %H시 %M분 %S초', time.localtime(current_time_sec))
                                data_to_insert = (img_name_numbering, predicted_plate_sting, image_path, current_time, current_time_sec, image_path2)
                                #id INTEGER, car_plate TEXT, detected_image_path_in TEXT, time_in TEXT, detected_image_path_out TEXT, time_out TEXT, time_in_sec TEXT, time_out_sec TEXT, price INTEGER
                                #db_io.execute(f"INSERT INTO car_plate_data VALUES ({img_name_numbering}, {predicted_plate_sting}, {image_path}, {current_time}, ' ', ' ', {current_time_sec}, ' ', 0")
                                db_io.execute("INSERT INTO car_plate_data VALUES (?, ?, ?, ?, ' ', ' ', ?, 0, 0, ?, ' ')", data_to_insert)

                                parking_db.commit()

                            else:
                                car_out_numbering += 1
                                car_plate_in_parking_lot.remove(predicted_plate_sting)
                                os.makedirs(str(save_dir / 'car_out_plate'), exist_ok=True)
                                image_path = str(save_dir / 'car_out_plate/{}.jpg'.format(car_out_numbering))
                                cv2.imwrite(image_path, processed_image)
                                os.makedirs(str(save_dir / 'car_out_car'), exist_ok=True)
                                image_path2 = str(save_dir / 'car_out_car/{}.jpg'.format(car_out_numbering))
                                cv2.imwrite(image_path2, im0)
                                current_time_sec = time.time()
                                current_time = time.strftime('%Y년 %m월 %d일 %H시 %M분 %S초', time.localtime(current_time_sec))
                                price = 0
                                data_to_insert = (image_path, current_time, current_time_sec, price, image_path2, predicted_plate_sting)
                                #id INTEGER, car_plate TEXT, detected_image_path_in TEXT, time_in TEXT, detected_image_path_out TEXT, time_out TEXT, time_in_sec TEXT, time_out_sec TEXT, price INTEGER
                                #db_io.execute(f"INSERT INTO car_plate_data VALUES ({img_name_numbering}, {predicted_plate_sting}, {image_path}, {current_time}, ' ', ' ', {current_time_sec}, ' ', 0")
                                
                                db_io.execute("UPDATE car_plate_data SET detected_image_path_out_plate = ?, time_out =?, time_out_sec = ?, price = ?, detected_image_path_out_car = ? WHERE car_plate = ? AND detected_image_path_out_car = ' '", data_to_insert)
                                parking_db.commit()

                            

                        #label = f'{names[int(cls)]} {conf:.2f}'

                        

                        print(predicted_plate_sting)
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        """

                        if TrackingDB(predicted_plate_sting):
                            plot_one_box(xyxy, im0, label=predicted_plate_sting, color=(0,0,255), line_thickness=3)

                        else:
                            #label = f'{names[int(cls)]} {conf:.2f}'
                            label = 'not target'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)"""
                        ############여기까지 수정######################
                        #label = f'{names[int(cls)]} {conf:.2f}'
                        #label = f'{predicted_plate_sting} {conf:.2f}'
                        

                        """
                        MongoDB(source, save_path, image, time, plate1)
                        """
                        
                        #위에는 아직 테스트만


                #하단은 추가한 것
                #time_to_save = time.strftime('%Y년 %m월 %d일 %H시 %M분 %S초', time.localtime(time.time())) #데이터 베이스 저장용
                #MongoDB(source, save_path, im0, time_to_save, predicted_plate_sting) #데이터베이스에 저장

            t3 = time_synchronized()
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            print(f'{s}yolo+ocr. ({t3 - t1:.3f}s)')
            

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        #vid_writer2 = cv2.VideoWriter(save_path2, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)) 테스트해봄
                    vid_writer.write(im0)
                    #vid_writer2.write(im0) 테스트해봄

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    #print('detected all:', car_plate_in_parking_lot)
    parking_db.close()




def yolo_start(source):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    #opt = parser.parse_args()
    opt = EasyDict({
        'weights':'car_plate_detect.pt',
        'source':source,
        'img_size':640,
        'conf_thres':0.5,
        'iou_thres':0.45,
        'device':'',
        'view_img':False,
        'save_txt':False,
        'save_conf':False,
        'classes':None,
        'agnostic_nms':False,
        'augment':False,
        'update':False,
        'project':'runs/detect',
        'name':'exp',
        'exist_ok':False
        })

    print(opt)

    cudnn.benchmark = True
    cudnn.deterministic = True
    ocr_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    OCR_pridiction_opt = EasyDict({
    'workers':0,
    'batch_size':1,
    'saved_model':'best_accuracy.pth',
    'batch_max_length':80,
    'imgH':32,
    'imgW':100,
    'charachter':'대더아도사러주오광북조허경전호서5자마저2모인우어머4너남두8하버로구다바제소루고거원1강충천9수3부누나가산노기07라6무보울',
    'Transformation':'TPS',
    'FeatureExtraction':'ResNet',
    'SequenceModeling':'BiLSTM',
    'Prediction':'Attn',
    'num_fiducial':20,
    'input_channel':1,
    'output_channel':512,
    'hidden_size':256,
    })

    converter = AttnLabelConverter('대더아도사러주오광북조허경전호서5자마저2모인우어머4너남두8하버로구다바제소루고거원1강충천9수3부누나가산노기07라6무보울')
    OCR_pridiction_opt.num_class = len(converter.character)

    OCR_model = Model(OCR_pridiction_opt)
    OCR_model = torch.nn.DataParallel(OCR_model).to(ocr_device)

    # load model
    OCR_model.load_state_dict(torch.load(OCR_pridiction_opt.saved_model, map_location=ocr_device))
    OCR_model.eval()

    #욜로 실행
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt, OCR_model, OCR_pridiction_opt, ocr_device, converter)
                strip_optimizer(opt.weights)
        else:
            detect(opt, OCR_model, OCR_pridiction_opt, ocr_device, converter)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    #Createmongodb() #db 생성

    #텐서플로로 만든 번호판 읽는 모델 로드
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #if gpus:
    #try:
        # Currently, memory growth needs to be the same across GPUs
    #    for gpu in gpus:
    #    tf.config.experimental.set_memory_growth(gpu, True)
    #    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
    #    print(e)


    #checkpoint_path = "./checkpoints/InceptionV3"
    #ckpt = tf.train.Checkpoint(encoder=encoder,
    #                        decoder=decoder,
    #                        optimizer = optimizer)
    #ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=40)
    #ckpt.restore("./checkpoints/InceptionV3\\ckpt-{}".format(i))

    #######여기까지#####

    cudnn.benchmark = True
    cudnn.deterministic = True
    ocr_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #하단: OCR 모델불러오기
    OCR_pridiction_opt = EasyDict({
    'workers':0,
    'batch_size':1,
    'saved_model':'best_accuracy.pth',
    'batch_max_length':80,
    'imgH':32,
    'imgW':100,
    'charachter':'대더아도사러주오광북조허경전호서5자마저2모인우어머4너남두8하버로구다바제소루고거원1강충천9수3부누나가산노기07라6무보울',
    'Transformation':'TPS',
    'FeatureExtraction':'ResNet',
    'SequenceModeling':'BiLSTM',
    'Prediction':'Attn',
    'num_fiducial':20,
    'input_channel':1,
    'output_channel':512,
    'hidden_size':256,
    })

    converter = AttnLabelConverter('대더아도사러주오광북조허경전호서5자마저2모인우어머4너남두8하버로구다바제소루고거원1강충천9수3부누나가산노기07라6무보울')
    OCR_pridiction_opt.num_class = len(converter.character)

    OCR_model = Model(OCR_pridiction_opt)
    OCR_model = torch.nn.DataParallel(OCR_model).to(ocr_device)

    # load model
    OCR_model.load_state_dict(torch.load(OCR_pridiction_opt.saved_model, map_location=ocr_device))
    OCR_model.eval()

    #욜로 실행
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt, OCR_model, OCR_pridiction_opt, ocr_device, converter)
                strip_optimizer(opt.weights)
        else:
            detect(opt, OCR_model, OCR_pridiction_opt, ocr_device, converter)


