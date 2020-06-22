import torch
import torch.nn as nn


def double_conv(in_c,out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv

def crop_img(tensor,target_tensor):
    target_size = target_tensor.size()[2] # bs,c,h,w Ex:[[1,1024,28,28]] --> target_size = [56]
    tensor_size = tensor.size()[2] # Similarly it returns tensor_size = [28]
    delta = tensor_size - target_size # Calculate the difference
    delta = delta // 2 # Divide the difference by 2
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta] # reshape the tensor 

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)

# One 
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512, # Only half of the 1024 channel comes from here.
            kernel_size=2,
            stride=2,
            )

        self.up_conv_1 = double_conv(1024,512) # because the previous output channels are combined with the cropped image

# Two
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256, # Only half of the 1024 channel comes from here.
            kernel_size=2,
            stride=2,
            )

        self.up_conv_2 = double_conv(512,256) # because the previous output channels are combined with the cropped image

# Three
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128, # Only half of the 1024 channel comes from here.
            kernel_size=2,
            stride=2,
            )

        self.up_conv_3 = double_conv(256,128) # because the previous output channels are combined with the cropped image


# Four
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64, # Only half of the 1024 channel comes from here.
            kernel_size=2,
            stride=2,
            )

        self.up_conv_4 = double_conv(128,64) # because the previous output channels are combined with the cropped image

# Output Layer
        self.out = nn.Conv2d(
            in_channels = 64,
            out_channels=2, # Given in the paper
            kernel_size=1 # check??
        )




       

    def forward(self, image):
        # encoder 
        # bs,c,h,w
        x1 =  self.down_conv_1(image)
        #print("Start of Encoder Size:  ",x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        #print("End of Encoder Size:  ", x9.size())
        #print("Cropped test:  ", x9.size()[2])

        # decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7,x)
        x = self.up_conv_1(torch.cat([x,y],axis=1))


        x = self.up_trans_2(x)
        y = crop_img(x5,x)
        x = self.up_conv_2(torch.cat([x,y],axis=1))

        x = self.up_trans_3(x)
        y = crop_img(x3,x)
        x = self.up_conv_3(torch.cat([x,y],axis=1))


        x = self.up_trans_4(x)
        y = crop_img(x1,x)
        x = self.up_conv_4(torch.cat([x,y],axis=1))

        x = self.out(x)
        print(x.size())
        return x
        






        #print("Upsampling Output",x.size())
        # remove the below comments for understanding cropping process
        #print("Original image size",x7.size()) # Original image is cropped to match up sampling image
        #print("Cropped image",y.size()) 


if __name__ == "__main__":
    image = torch.rand((1,1,572,572)) #(bs,c,h,w)
    model = UNet()
    print(model(image))
        


   