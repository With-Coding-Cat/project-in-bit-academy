import cv2
import os
from PIL import Image
from dataset import AlignCollate, personal_dataset
from OCR.utils import CTCLabelConverter, AttnLabelConverter
import torch.nn.functional as F
import torch

#img_name_numbering = 0
def take_picture(x, img):
    #global img_name_numbering
    #img_name_numbering += 1
    x1, y1, x2, y2 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
    roi = img[y1:y2,x1:x2].copy()
    #os.makedirs(str(save_path / 'img'), exist_ok=True)
    #cv2.imwrite(str(save_path / 'img/{}.jpg'.format(time.localtime(time.time()))), roi)
    #cv2.imwrite(str(save_path / 'img/{}.jpg'.format(img_name_numbering)), roi)
    return roi


def predict_plate(model, opt, image_to_give, device, converter):

    image_to_give = cv2.cvtColor(image_to_give, cv2.COLOR_BGR2RGB)
    image_to_give = cv2.cvtColor(image_to_give, cv2.COLOR_BGR2GRAY)
    image_to_give = Image.fromarray(image_to_give)

    AlignCollate_demo = AlignCollate()
    demo_data = personal_dataset(image=image_to_give, opt=opt)


    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=1,
        shuffle=False,
        num_workers=int(0),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    #model.eval()
    with torch.no_grad():
        for image_tensors, _ in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)


            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for pred in preds_str:
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

            #출력하기
            return pred