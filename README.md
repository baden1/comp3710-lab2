# Fourier

- Implemented function in pytorch by replacing numpy functions with torch functions, setting device to cuda, and running on the rangpur cluster. 
- The square wave plots show higher harmonics give a better representation of the original signal

![image](fourier/square_wave_h=17.png)

- However the overall result plot shows the low harmonics have the highest proportions

![image](fourier/fourier_results_N=2048.png)

- DFT is O(n<sup>2</sup>) so doubling n -> ~4x runtime. This is shown in logs. 

# Eigenfaces
![image](eigenfaces/eigenfaces.png)
![image](eigenfaces/compactness.png)

- Calculating PCA using Pytorch was ~2x faster.
- 63% accuracy using a random forest classifier on the PCA-transformed data. 

# CNNs
## CNN Classifier
- I created a CNN with 2 convolution layers with 32 filters each, max pooling after each convolution, and a fully connected layer. 
- It gets around 80-85% accuracy

![image](cnn/cnn_results/prediction_0.png)
![image](cnn/cnn_results/prediction_4.png)

## Dawn Bench

# Recognition

## VAE
- read images to 64x64 = dim 4096.
- made latent dimension 64. 

Original 

![image](recognition/vae_images/original/0.png)

Reconstructed

![image](recognition/vae_images/reconstructed/0.png)

- Reconstructed image loses some detail

Plot of the first two principal components of the mean vectors of the latent distributions where test images land.
![image](recognition/vae_images/plots/pca-test-to-latent.png)