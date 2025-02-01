import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLay(nn.Module):
          def __init__(self, inp, out, stride=1):
              '''
              inp : number of input channels of the convolution layer
              out : number of output channels of the convolution layer
              stride : stride=1 means no change in shape of input image, if we alter stride then shape of the input image changes according to the following formula
                      output_width = (input_width - kernel_size + + 2*padding)/stride  + 1
                      similarly for output_height
              '''
              super(ConvLay, self).__init__()
              self.conv1 = nn.Conv2d(inp, out, kernel_size=3, stride=stride, padding=1, bias=False)
              self.bn1 = nn.BatchNorm2d(out)
              self.relu = nn.ReLU()

          def forward(self,x):
              out=self.relu(self.bn1(self.conv1(x)))
              return out

class net9(nn.Module):
    def __init__(self,num_classes=10):
        super(net9,self).__init__()


        self.layer0 = ConvLay(1 ,32 ,stride= 1)
        self.layer1 = ConvLay(32, 64, stride= 1)
        self.layer2 = ConvLay(64 ,128 ,stride= 1)

        #self.res1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer3 = ConvLay(128 ,128 ,stride= 1)
        self.layer4 = ConvLay(128 ,128 ,stride= 1)


        self.layer5 = ConvLay(128 ,256 ,stride= 1)
        self.layer6 = ConvLay(256 ,512 ,stride= 1)

        self.res2 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer7 = ConvLay(512 ,512 ,stride= 1)
        self.layer8 = ConvLay(512 ,512 ,stride= 1)

        #self.res3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)

        self.fclayer = nn.Sequential(
               nn.AdaptiveAvgPool2d((1,1)),
               nn.Flatten(),
               nn.Linear (512,num_classes),
               nn.Softmax(dim=-1))

    def forward (self,x):

        x= self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        out_n2 = x

        x = self.layer3(x)
        x = self.layer4(x)

        # Apply residual connection
        x = out_n2 + x
        out_n4 = x


        x = self.layer5(x)
        x = self.layer6(x)


        # Apply residual connection with convolution layer
        x = self.res2(out_n4) + x
        out_n6 = x


        x = self.layer7(x)
        x = self.layer8(x)

        # Apply residual connection
        x = out_n6 + x

        #x = x.view(x.size(0), -1)
        x = self.fclayer(x)

        return x

net = net9().to(device)
