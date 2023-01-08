import torch
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation(model, criterion, valid_loader, converter, args):
  length_of_data = 0
  acc = 0
  loss = 0

  for image_tensors, labels in valid_loader:
    length_of_data += 1
    image = image_tensors.to(device)
    batch_size = image.size(0)

    length_for_pred = torch.IntTensor([args.text_max_length] * batch_size).to(device)
    text, length = converter.encode(labels, args.text_max_length)


    preds = model(image) # 192x45x206
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    
    cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text, preds_size, length)
    loss += cost

    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)

    for label, pred in zip(labels, preds_str):
      label = " ".join(label.lower().split())
      pred = " ".join(pred.lower().split())
      if label == pred:
        acc += 1

    # break

  acc = acc / (length_of_data * args.batch_size)
  loss = loss / length_of_data

  return preds_str, labels, acc, loss
