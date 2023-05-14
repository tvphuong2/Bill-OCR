# Vietnamese Receipt OCR ver 0.8
Một project nhỏ không cần dữ liệu huấn luyện có thể nhận diện các mục thanh toán trong hóa đơn tiếng Việt.
Độ chính xác đang đạt được: 63%.

## Các thành phần:

### Tạo mẫu huấn luyện
Tạo ra các mẫu mang các đặc trưng của dữ liệu thực. Các mẫu này có các đặc trưng như nhiễu, nghiêng, lệch, mờ, ... với đa dạng kiểu chữ.

(![image](https://github.com/tvphuong2/Bill-OCR/assets/83411225/0b43b207-51c6-45c5-8985-ee999e6ab422)


```
!python3 "DataGenerator/generator.py" \
--count 10000 \
--output_dir '/outputdir' \
--extension 'jpg' \
--dict_dir "DataGenerator/dict/Viet74K.txt" \
--font_dir "DataGenerator/fonts"
```


### Mô hình học sâu

```
Model(
  (feature_extraction): FeatureExtractor(
    (ConvNet): Sequential(
      (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU(inplace=True)
      (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU(inplace=True)
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (10): ReLU(inplace=True)
      (11): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
      (12): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (13): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (14): ReLU(inplace=True)
      (15): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (16): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (17): ReLU(inplace=True)
      (18): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1))
      (20): ReLU(inplace=True)
    )
  )
  (AdaptiveAvgPool): AdaptiveAvgPool2d(output_size=(None, 1))
  (sequence_modeling): Sequential(
    (0): BidirectionalLSTM(
      (rnn): LSTM(512, 256, batch_first=True, bidirectional=True)
      (linear): Linear(in_features=512, out_features=256, bias=True)
    )
    (1): BidirectionalLSTM(
      (rnn): LSTM(256, 256, batch_first=True, bidirectional=True)
      (linear): Linear(in_features=512, out_features=256, bias=True)
    )
  )
  (Prediction): Linear(in_features=256, out_features=206, bias=True)
)
```
Cấu trúc này dựa trên mô hình easy OCR. Để xây dựng mô hình, dùng lệnh.
```
!python "OCR/train.py" \
--train_dir "Training_data" \
--valid_dir "Valid_data" \
--saved_model "dir/name_model.pth" 
```

### Nhận diện ký tự trong hình

Sử dụng thư viện craft để nhận diện tối đa ký tự có trong khung hình

![image](https://user-images.githubusercontent.com/83411225/211195688-eb750b01-8e10-486e-9ea4-107c8b234af1.png)

Tính góc nghiêng của các ký tự, làm mịn góc

![image](https://github.com/tvphuong2/Bill-OCR/assets/83411225/cd012e32-a626-4c9c-9ba1-b3118be036a3)


Sử dụng mô hình đã huấn luyện để nhận dạng ký tự. Cuối cùng viết luật để nhận diện các mục thanh toán

```
%pip install craft_text_detector
!python "OCR/predict.py" \
--predict_dir "image.jpg" \
--saved_model "dir/name_model.pth" 
```

## Kết quả

![image](https://github.com/tvphuong2/Bill-OCR/assets/83411225/79d47606-0f25-4eaa-9006-652ff5f415b2)

![image](https://github.com/tvphuong2/Bill-OCR/assets/83411225/dfad6e24-8f21-43be-aa25-1e451a58e57f)

![image](https://github.com/tvphuong2/Bill-OCR/assets/83411225/cd282486-957c-49fb-b1d0-1ba12f89d579)
