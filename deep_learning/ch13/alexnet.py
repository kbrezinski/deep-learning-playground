
'''
alexnet implementation with cifar-10
'''

# torchvision transforms:
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((70, 70)),  # scale up image from 32 x 32
    torchvision.transforms.RandomCrop((64, 64)),  # crop a 64 x 64 region; gets off center image; centerCrop for valid
    torchvision.transforms.ToTensor(),  # normalize (0, 1)
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # scale in range (-1, 1)

# to help with overfitting, use other data augmentation techniques
# input image dims, 256 x 3 x 64 x 64



