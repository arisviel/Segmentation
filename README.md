# Binary Segmentation project on "Carvana" dataset with pretrained model.

  I took a Deeplabv3 head with resnet101 blackbone as a model and used Pytorch as main framework. The project can be divided into several parts, which are described below:
-------------------------------------------------------------------------------------------------------------------------------
### 1. DataLoader

  The model expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N, 3, H, W), where N is the number of images, H and W are expected to be at least 224 pixels. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
  
  The model returns an OrderedDict with two Tensors that are of the same height and width as the input Tensor, but with 21 classes. output['out'] contains the semantic masks, and output['aux'] contains the auxillary loss values per-pixel.
  
  So we need to transform the data to this format. The data contains photos of cars and csv file,
  

