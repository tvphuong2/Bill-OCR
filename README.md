# Bill_OCR ver 0.5
Một project nhỏ không cần dữ liệu huấn luyện có thể nhận diện các mục thanh toán trong hóa đơn tiếng Việt
Độ chính xác đang đạt được: 40%.

## Các thành phần:

### Tạo mẫu huấn luyện
Tạo ra các mẫu mang các đặc trưng của dữ liệu thực. Các mẫu này có các đặc trưng như nhiễu, nghiêng, lệch, mờ, ... với đa dạng kiểu chữ.

![image](https://user-images.githubusercontent.com/83411225/211195549-5572b1f8-83e3-48f0-9e46-1536c08918e3.png)

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
Cấu trúc này dựa trên [phương pháp của Baoguang Shi và các cộng sự](https://arxiv.org/pdf/1507.05717.pdf)


### Nhận diện ký tự trong hình

Sử dụng thư viện craft để nhận diện tối đa ký tự có trong khung hình

![image](https://user-images.githubusercontent.com/83411225/211195688-eb750b01-8e10-486e-9ea4-107c8b234af1.png)

Tính góc nghiêng của các ký tự, làm mịn góc

![image](https://user-images.githubusercontent.com/83411225/211196255-ad1c730c-39af-4f52-819e-aced9bd05f60.png)

Sử dụng mô hình đã huấn luyện để nhận dạng ký tự. Cuối cùng viết luật để nhận diện các mục thanh toán

## Kết quả

![image](https://user-images.githubusercontent.com/83411225/211196417-9e9309eb-34aa-4cec-ad68-d34410d78f73.png)


![image](https://user-images.githubusercontent.com/83411225/211196393-cce8a337-beb1-46ee-950d-db6efed3c364.png)

![image](https://user-images.githubusercontent.com/83411225/211196426-10420267-bba3-4767-a2c6-00ebe2c924aa.png)

![image](https://user-images.githubusercontent.com/83411225/211196433-afa83c46-a050-4e2d-bef4-2f60f5575167.png)
