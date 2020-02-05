# Binary Segmentation project on "Carvana" dataset with pretrained model.

  I took a Deeplabv3 head with resnet101 blackbone as a model and used Pytorch as main framework. The project can be divided into several parts, which are described below:
-------------------------------------------------------------------------------------------------------------------------------
### 1. DataLoader

The model expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N, 3, H, W), where N is the number of images, H and W are expected to be at least 224 pixels. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
  
The model returns an OrderedDict with two Tensors that are of the same height and width as the input Tensor, but with 21 classes. output['out'] contains the semantic masks, and output['aux'] contains the auxillary loss values per-pixel.
  
So we need to transform the data to this format. The data contains photos of cars and csv file, in which the names of the photo and the mask code for it. After decode we get a tensor of 0&1 with shape H, W, 1. Data loader code:

    def batch_generator(phase, data_tr, data_val, batch_size=10, num_batches=100):
    if phase == 'train':
        data = data_tr
    else:
        data = data_val
        
    # Shuffle our data list
    train_list = (data.iloc[np.random.permutation(len(data))]).reset_index(drop=True)
    
    for batch in range(num_batches):
        for size in range(batch_size):

            index = batch * batch_size + size # Index of step
            imag, mask_rle = train_list.iloc[index] 
            
            # Read the photo, decode and transform mask into an image, so that we can use Torch transforms on the mask.
            image = Image.open('/content/drive/My Drive/train/'+ imag)
            mask_rle = rle_decode(mask_rle)
            mask_rle = mask_rle*255
            topil = transforms.ToPILImage()
            mask = topil(mask_rle)

            # Resize
            resize = transforms.Resize(size=(250, 250))
            image = resize(image)
            mask = resize(mask)

            if phase == 'train':
              # Random crop
              i, j, h, w = transforms.RandomCrop.get_params(
                  image, output_size=(226, 226))
              image = TF.crop(image, i, j, h, w)
              mask = TF.crop(mask, i, j, h, w)

              # Random horizontal flipping
              if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

              # Random vertical flipping
              if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            else:
              CentralCrop = transforms.CenterCrop(226)
              image = CentralCrop(image)
              mask = CentralCrop(mask)
            
            # Transform to tensor
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)
            tr_im = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            image = tr_im(image)
            
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
            # Stack batch
            if size == 0: 
                x_batch = image
                y_batch = mask
            else:
                x_batch = torch.cat((x_batch, image),0)
                y_batch = torch.cat((y_batch, mask),0)

            yield x_batch, y_batch
  

