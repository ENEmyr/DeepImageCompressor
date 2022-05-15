# # Datasets

Datasets that used in this project can be found at these follow references

For training the model:
> Hsankesara. (13 June 2018). Flickr Image dataset. Retrieved from Kaggle: [https://www.kaggle.com/hsankesara/flickr-image-dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset)

For testing and measure the model:
> Kodak Lossless True Color Image Suite. (15 November 1999). Retrieved from Kodak Lossless True Color Image Suite: [https://r0k.us/graphics/kodak/](https://r0k.us/graphics/kodak/)

## Structuring dataset directories
Because of this project use ImageDataGenerator to generate batches of tensor with real-time image augmentation, therefore it need to structuring a directory and  sub-directories to be in form that ImageDataGenerator can flow from... [more detail](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory)

In order to structuring if you already used Fish shell you can easily done this process with our prepared script
``` bash
fish split_dataset.fish
   ```
