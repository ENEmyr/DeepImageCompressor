# # SSDCAE Model
![SSDCAE Block Diagram](https://raw.githubusercontent.com/Untesler/DeepImageCompressor/main/experimental_result/model_block_diagram.png)

Essentially, this is an Autoencoder network with a little trick that uses PReLu as an activation function for each Up/Down-sampling block, and in each downsampling block, there is a convolution layer before the downsampling layer is applied on feature maps, and in the upsampling block, the Subpixel layer is used to upsampling incoming feature maps.

### Down-sampling Unit Block Diagram
![downsampling unit block diagram](https://raw.githubusercontent.com/Untesler/DeepImageCompressor/main/experimental_result/down_sampling_unit.png)
### Up-sampling Unit Block Diagram
![upsampling unit block diagram](https://raw.githubusercontent.com/Untesler/DeepImageCompressor/main/experimental_result/up_sampling_unit.png)

## Training Result 

The comparison of loss between using a normal transposed convolutional layer as an upsampling layer and using a subpixel layer as an upsampling layer. 

![the loss comparison](https://raw.githubusercontent.com/Untesler/DeepImageCompressor/main/experimental_result/real_model_loss.png)

## Setting up training environment
- To setup environment -> `conda env create -f environment.yml` 
- To install dependencies -> `pip install -r requirements.txt`
