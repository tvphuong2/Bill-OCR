import torch.nn as nn


class Model(nn.Module):

  def __init__(self, args):
    super(Model, self).__init__()
    self.args = args

    self.feature_extraction = FeatureExtractor(args.input_channel, args.output_channel)
    self.feature_extraction_output = args.output_channel
    self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

    self.sequence_modeling = nn.Sequential(
      BidirectionalLSTM(self.feature_extraction_output, args.hidden_size, args.hidden_size),
      BidirectionalLSTM(args.hidden_size, args.hidden_size, args.hidden_size))
    self.sequence_modeling_output = args.hidden_size

    self.Prediction = nn.Linear(self.sequence_modeling_output, args.num_class)
    # self.Prediction = nn.Sequential(
    #   nn.Linear(self.feature_extraction_output, args.num_class),
    #    nn.ReLU(True))

  def forward(self, input):
    visual_feature = self.feature_extraction(input)
    visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
    visual_feature = visual_feature.squeeze(3) #[b, w, c]

    contextual_feature = self.sequence_modeling(visual_feature)

    prediction = self.Prediction(contextual_feature.contiguous())
    # prediction = self.Prediction(visual_feature.contiguous())

    return prediction




class FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=512):
        super(FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x32x368
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x16x184
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x16x184
            nn.MaxPool2d(2, 2), # 256x8x92
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2)),  # 256x4x46
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x46
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x46
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x45

    def forward(self, input):
        return self.ConvNet(input)

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
