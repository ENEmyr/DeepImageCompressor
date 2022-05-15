<h1 align="center">DeepImageCompressor</h1>

![SSIM Comparison](https://raw.githubusercontent.com/Untesler/DeepImageCompressor/main/experimental_result/ssim_comparison/ssim_comparison.png)
This project was created with the goal of reducing image file storage size by developing an image compression system using one of the neural networks named Stacked Denoising Autoencoder and using the special activation function PReLU [2] and Sub-pixel layer [11] as the up-sampling layer.
The model was trained and measured on the Flickr Image Dataset [4] and the Kodak Image Dataset [7], respectively, before developing a compression system to compress the encoder model's latent representation vector with the Deflate algorithm [9].
The trained model has a reconstruction accuracy of 76%, and there is still room for improvement. 
Furthermore, on an image resolution of 128x128 pixels, the compression system can reduce the image file size relative to the original by an average of 89.92%, which is 43.97% more than the popular image compression algorithm JPEG. 

> To be clear this project is not end-to-end compression network
> 
## System Block Diagram
![Image Compression System Block Diagram](https://raw.githubusercontent.com/Untesler/DeepImageCompressor/main/experimental_result/image_compression_system.png)
## Compression Efficiency

Latent representation vector from SSDCAE that was compressed with Deflate algorithm can reduce image file size by an average of 89.92%, which is 43.97% more than  JPEG algorithm on an image resolution of 128x128 pixels.

| Original File Size (bytes) |Latent Representation Size (bytes)|Compressed Latent Size (bytes)| Compressing Time (s) | Decompressing Time (s) | JPEG Compression Rate (%) | SSDCAE Compression Rate (%) |
|----------------|-------------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
|33,799|11,487|3,070|0.0018|3.9577E-5|44.46|90.92|
|31,532|11,414|3,102|0.0016|4.3154E-5|43.83|90.16|
|30,957|11,323|3,421|0.0013|4.4107E-5|41.40|88.95|
|33,680|11,423|3,491|0.0012|4.3631E-5|42.61|89.63|
|--------|--------|--------|--------|--------|--------|--------|
||Average||1.47ms|42.62μs|43.07|89.92|


## References

[1] Abir Jaafar, Ali Al-Fayadh, Naeem Radi Hussain. (2018). Image compression techniques: A survey in lossless and lossy algorithms. Neurocomputing, 300, 44-69.

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. 2015 IEEE International Conference on Computer Vision (ICCV) (Pages 1026-1034). Santiago, Chile: IEEE. doi:10.1109/ICCV.2015.123

[3] Hore, A., & Ziou, D. (2010). Image Quality Metrics: PSNR vs. SSIM. 2010 20th International Conference on Pattern Recognition (Pages 2366-2369). Istanbul, Turkey : IEEE. doi:10.1109/ICPR.2010.579

[4] Hsankesara. (13 June 2018). Flickr Image dataset. Retrieved from Kaggle: https://www.kaggle.com/hsankesara/flickr-image-dataset

[5] John Gantz, and David Reinsel. (2012). The digital universe in 2020: Big data, bigger digital shadows, and biggest growth in the far east. IDC iView: IDC Analyze the future, 2007, 1-16.

[6] Jonathan, et al. Masci. (2011). Stacked convolutional auto-encoders for hierarchical feature extraction. International conference on artificial neural networks (Pages 52-59). Berlin, Heidelberg: Springer.

[7] Kodak Lossless True Color Image Suite. (15 November 1999). Retrieved from Kodak Lossless True Color Image Suite: https://r0k.us/graphics/kodak/

[8] L. Gondara. (2016). Medical Image Denoising Using Convolutional Denoising Autoencoders. 2016 IEEE 16th International Conference on Data Mining Workshops (ICDMW) (Pages 241-246). Barcelona: IEEE. doi:10.1109/ICDMW.2016.0041

[9] P Deutsch. (1996). DEFLATE compressed data format specification version 1.3. IETF RFC 1951.
[10] Ryan Nash Keiron O'Shea. (2015). An Introduction to Convolutional Neural Networks. CoRR, abs/1511.08458. Retrieved from http://arxiv.org/abs/1511.08458

[11] Shi, W., Caballero, J., Theis, L., Huszar, F., Aitken, A., Ledig, C., & Wang, Z. (2016). Is the deconvolution layer the same as a convolutional layer? arXiv preprint arXiv:1609.07009.

[12] Theis, L., Shi, W., Cunningham, A., & Huszár, F. (2017). Lossy image compression with compressive autoencoders. arXiv preprint arXiv:1703.00395.

[13] Wang, Z., Bovik, A., Conrad, A., Sheikh, H. R., and Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. Transactions on Image Processing. 13(4), Pages 600–612. IEEE. doi:10.1109/TIP.2003.819861

[14] Yasi and Yao, Hongxun and Zhao, Sicheng Wang. (2016). Auto-encoder based dimensionality reduction. Neurocomputing, 184, 232-242.
