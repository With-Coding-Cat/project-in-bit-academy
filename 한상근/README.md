This is Sang-Geun's personal folder. It is made to specify member of this project team who really did his or her roll and this folder is made up of what I did or I adviced durring project.

Here's two main categories. One is python codes (or files) I got from other github, and another is python codes (or files) I wrote my self.

CODES FROM GIHUB:
1. yolov5 :https://github.com/ultralytics/yolov5
2. OCR program: https://github.com/clovaai/deep-text-recognition-benchmark
3. korean license plate generator: https://github.com/Usmankhujaev/KoreanCarPlateGenerator

First. yolov5
- It is used for detecting car plates in videos or pictures. Only small changes were applied for project.

Second. OCR program
- It is used to read car plates when yolov5 detect car plates. Only small changes were applied to be used in yolov5 process.

Third. korean license plate generator
- It is used to make artificial car plates to train OCR program. But original version had some errors such as missing specific image files and some codes or file paths that are not exists. So I fixed these thing.
- Also this program only made car plates of specific styles. So I applied some codes for augmentation.


CODE I WROTE:
1. image augmentation: Adding some noise, tilting, resizing and so on.
2. extra codes for project in yolov5: Taking pictures of detected parts. Putting processed data to DB.
3. GUI: It is made by using PyQt5. Most of this is made by myself.

My Roll in Project:
- github main manager in project.
- Checking codes from others and integrating codes into one program.
- Scheduling.
- Making deep learning model (and training).
