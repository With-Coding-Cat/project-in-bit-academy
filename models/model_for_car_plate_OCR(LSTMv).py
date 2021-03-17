from os import listdir
from pickle import dump, load
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from glob import glob
from numpy import array

import tensorflow as tf
import pathlib

#GPU 부족을 방지
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


#특징 추출
def extract_features(directory):
    model = NASNetLarge()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    #이미지 파일 별 특징을 담을 딕셔너리 만들기
    features = dict()
    for name in tqdm(listdir(directory)):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(331, 331))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
    return features


#이미지 파일에 적혀있는 영어를 한글로 전환하여 텍스트로 추출
def process_eng_to_hangle(directory):
    descriptions = {}
    dictToHangul = {"a": "아", "eo": "어", "o": "오", "u": "우", "ba": "바", "beo": "버", "bo": "보", "bu": "부",
                    "da": "다", "deo": "더", "do": "도", "du": "두", "ga": "가", "geo": "거", "go": "고", "gu": "구",
                    "ja": "자", "jeo": "저", "jo": "조", "ju": "주", "ma": "마", "meo": "머", "mo": "모", "mu": "무",
                    "na": "나", "neo": "너", "no": "노", "nu": "누", "la": "라", "leo": "러", "lo": "로", "lu": "루",
                    "sa": "사", "seo": "서", "so": "소", "su": "수", "heo":"허", "ha":"하", "ho":"호", "reo":"러",
                    "A": "서울", "B": "경기", "C": "인천", "D": "강원", "E": "충남", "F": "대전",
                    "G": "충북", "H": "부산", "I": "울산", "J": "대구", "K": "경북", "L": "경남",
                    "M": "전남", "N": "광주", "O": "전북", "P": "제주"}

    #한글로 전환한 텍스트를 저장하기(딕셔너리 형태)
    file_names = glob(directory + '/*')
    for name in tqdm(file_names):
        processed_name = ""
        temp = ""
        img_name = pathlib.PurePath(name).stem
        for char in img_name:
            if char.isnumeric():
                if temp:
                    engTohan = dictToHangul[temp]
                    processed_name += engTohan + " "
                    temp = ""
                processed_name += char + " "
            else:
                temp += char

    descriptions[img_name] = [processed_name]
    return = descriptions


#텍스트 파일 읽어오기
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


#["이미지파일명" "읽어낸 문자열"] 순으로 된 데이터를 저장하기 위함
#캡셔닝 용도로, 한 이미지를 여러가지로 묘사할 수 있을 때 사용가능
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


#이미지 파일명과 이를 설명하는 문구가 함께 적힌 텍스트 파일을 불러와서 이미지 파일명만 추출하기
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split(' ')[0]
        dataset.append(identifier)
    return set(dataset)


#딥러닝에서 문자열을 적절하게 처리하기 위함(시작과 끝 알려주기)
def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions


#저장해둔 이미지들의 특징 다시 불러오기
def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


#이미지를 나타내는 문자열이 있는 딕셔너리에서 문자열 값만 뽑아내서 리스트 만들기
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


#주어진 글자들에 적절하게 토큰화하기
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


#모델 넣기 전 가공
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            #전후 예측
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)


#주어진 문자열에서 가장 긴 문자열의 길이 반환
def max_length_of_words(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


#모델 정의
def define_model(vocab_size, max_length):
    #입력1
    inputs1 = Input(shape=(4032,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(128, activation='relu')(fe1)
    #입력2
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(128)(se2)
    #입력들을 합치기
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    #최종 모델(캡셔닝용)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    #모델 요약
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


if __name__ == "__main__":
    #이미지 파일 경로를 지정
    directory = './img/train/img'
    features = extract_features(directory)
    print('Extracted Features: %d' % len(features))
    #뽑아낸 특징 저장
    dump(features, open('features.pkl', 'wb'))

    #각 이미지 파일마다 적혀있는 번호판의 번호를 추출
    descriptions = process_eng_to_hangle(directory)
    #저장하기-텍스트 형식
    save_descriptions(descriptions, 'descriptions.txt')

    #토큰화 준비
    tokenizer = create_tokenizer(descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    #저장파일 불러서 훈련용 데이터 처리하기
    filename = './descriptions.txt'
    train = load_set(filename)
    print('Dataset: %d' % len(train))
    train_descriptions = load_clean_descriptions('descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))
    train_features = load_photo_features('features.pkl', train)
    print('Photos: train=%d' % len(train_features))
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    max_length = max_length_of_words(train_descriptions)
    print('Description Length: %d' % max_length)
    X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)
    

    #저장파일 불러서 테스트용 데이터 처리하기
    filename = 'descriptions.txt' #적절한 파일명으로 교체 필요
    test = load_set(filename)
    print('Dataset: %d' % len(test))
    test_descriptions = load_clean_descriptions('descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
    test_features = load_photo_features('features.pkl', test)
    print('Photos: test=%d' % len(test_features))
    X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)

    #모델 생성하고, 저장장소 설정
    model = define_model(vocab_size, max_length)
    filepath = 'model/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    #모델 학습
    history = model.fit([X1train, X2train], ytrain, epochs=300, verbose=1, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))

    