# Pix2Pix
#### by Malte Kleine-Piening, Raphael Marx and Christopher Bröcker (20.04.2020)

## 1 Motivation / Introduction
Our group decided to implement the pix2pix paper from Isola et al. (2018).

At first we were a little confused because at first glance the pix2pix model seemed to be able to do everything. It could e.g. translate from edges to handbags, from aerial images to maps, from labels to facades and from black-and-white to color.

Other papers that we looked at like "Colorful Image Colorization" only focused on one of these tasks and built a spezialized architecture just for that task.

The authors of the pix2pix paper, however, showed that the task in a lot of image translation problems is very similar and boils down to a more general approch: predicting pixels from pixels. Therefore they introduce a general framework that can be used on numerous different tasks.

We quite like this wide applicability and the ease of adopting the model for a whole new task.

## 2 Background knowledge (reference to most important publications)

In the past if you wanted to do any image-to-image translation you had to build a specialized tool for that specific task. With the introduction of CNNs and eventually GANs image prediction became a lot easier to implement.

Because we are going to talk more about Generative Adversarial Networks (GANs) in the upcoming sections we should briefly explain what they are. GANs were first introduced by [Goodfellow, I. (2014)](https://papers.nips.cc/paper/5423-generative-adversarial-nets) .

The basic idea is that you have 2 neural networks that are trying to work against each other and therefore improve themselves. One NN is called the generator and aims to produce the best fake data that it can while the second NN is called the discriminator which gets both this fake input and real input from the dataset and tries to differentiate between the two. In our case the data that is being produced are images.

If the generator produces an image that fools the discriminator the weights of the discriminator are adapted to better recognize the generated image and therefore improves its detection rate.

The pix2pix paper doesn't use the base GANs but a variation called cGans which stands for conditional GANs. cGans don't just learn a mapping from random noise to output image but from observed image 
$x$ to an output image $y$. Often random noise is added to the input image $x$ to get non-deterministic results. We don't add random noise to the input, but use drop-out to generate randomness in our network while training and testing. They also (is it also or because the stuff above?) learn a structured loss which penalizes the joint configuration of the output.

It is also important to mention that the pix2pix paper is not the first paper to use cGANs but the first one to not build something specifically for one task.

## 3 Model
* focus on the c in cGAN
* conditional GANs learn a structured loss which penalizes the joint configuration of the output
* U-NET is notable
* PatchGAN classifier (from 38)

A lot of the base architecture is based on [Radford, A. (2016)](https://arxiv.org/abs/1511.06434) who gave some architecture guidelines for deep convolutional GANs that are also fully used here: 

>Architecture guidelines for stable Deep Convolutional GANs:
>* Replace any pooling layers with strided ,convolutions (discriminator) and fractional-strided convolutions (generator).
>* Use batchnorm in both the generator and the discriminator.
>* Remove fully connected hidden layers for deeper architectures.
>* Use ReLU activation in generator for all layers except for the output, which uses Tanh.
>* Use LeakyReLU activation in the discriminator for all layers.

The paper defines the following types of convolutional layers that we implemented first. 

* C(k)
  * Convolution (k filters; 4x4; stride 2)
  * BatchNorm (in testing and training)
  * ReLU

```python
class C(Layer):
    """This layer represents the C(k) layer described in the pix2pix paper. The activation function 
        is a parameter to allow the use of different activation functions like ReLU and leaky ReLU for 
        encoder and decoder. The sampling_factor gives a factor by which the convolution output will be 
        sampled up or down. A value of 2 will sample the tensor up by 2. A value of 0.5 will sample the 
        tensor down by 2."""
        
    def __init__(self, k, activation=None, sampling='down', batchnorm=True):
        super(C, self).__init__()
        if sampling == 'up':
            self.conv = tf.keras.layers.Conv2DTranspose(k, kernel_size=4, strides=2, activation=None, padding='same')
        elif sampling == 'down':
            self.conv = tf.keras.layers.Conv2D(k, kernel_size=4, strides=2, activation=None, padding='same')
        else:
            raise AttributeError('illegal sampling mode: "' + str(sampling) + '"')
            
        self.batchnorm = None
        if batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()
            
        self.activation = activation
        
    def call(self, x):
        x = self.conv(x)
        
        if self.batchnorm != None:
            x = self.batchnorm(x)
        
        if self.activation != None:
            x = self.activation(x)
            
        return x
```

The only difference to the C(k) layer is that the CD(k) layer also uses a dropout rate of 50% so CD(k) inherits from C(k). 
* CD(k)
  * Convolution (k filters; 4x4; stride 2)
  * BathNorm (in testing and training)
  * Dropout (rate of 50%)
  * ReLU

```python
class CD(C):
    """This layer represents the CD(k) layer described in the pix2pix paper. The activation function 
        is a parameter to allow the use of different activation functions like ReLU and leaky ReLU for 
        encoder and decoder. The sampling_factor gives a factor by which the convolution output will be 
        sampled up or down. A value of 2 will sample the tensor up by 2. A value of 0.5 will sample the 
        tensor down by 2."""
    
    def __init__(self, k, activation=None, sampling=None, batchnorm=True):
        super(CD, self).__init__(k, activation, sampling, batchnorm)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

    def call(self, x):
        x = self.conv(x)
        
        if self.batchnorm != None:
            x = self.batchnorm(x)
        
        x = self.dropout(x)
        
        if self.activation != None:
            x = self.activation(x)
            
        return x
```

Next we will talk about the architecture of the model. 

### Generator s
The generator should take an input image and create an image with the same structure, but different appearence. We use an encoder-decoder structure where we first downsample the input $x$ multiple times and then upsample it again to the original dimension.

<img src="misc/images/Encoder-decoder.png" height=200 width=336 />

 The downsampling creates an information bottleneck, but for many use cases we don't want to lose that much information while creating the new image. One example is image colorization, in which the edges of input $x$ and output $y$ should basically be the same. 

<!---<div>
<img src="misc/images/Encoder-decoder.png" height=200 width=336 hspace="20"/>
<img src="misc/images/unet.png" height=200 width=336 hspace="20"/>
</div>--->

To give the network the option to use the non-downsampled data, we add skip connections between mirrored layers of the encoder and the corresponding decoder layer.

 <img src="misc/images/unet.png" height=200 width=336 />

```python
class Generator(Model):
    def __init__(self, output_dim=3):
        super(Generator, self).__init__()
        
        # encoder:
        self.enc_conv1 = C(k=64, activation=tf.keras.layers.LeakyReLU(alpha=0.2), sampling="down", batchnorm=False)
        self.enc_conv2 = C(k=128, activation=tf.keras.layers.LeakyReLU(alpha=0.2), sampling="down")
        self.enc_conv3 = C(k=256, activation=tf.keras.layers.LeakyReLU(alpha=0.2), sampling="down")
        self.enc_conv4 = C(k=512, activation=tf.keras.layers.LeakyReLU(alpha=0.2), sampling="down")
        self.enc_conv5 = C(k=512, activation=tf.keras.layers.LeakyReLU(alpha=0.2), sampling="down")
        self.enc_conv6 = C(k=512, activation=tf.keras.layers.LeakyReLU(alpha=0.2), sampling="down")
        self.enc_conv7 = C(k=512, activation=tf.keras.layers.LeakyReLU(alpha=0.2), sampling="down")
        self.enc_conv8 = C(k=512, activation=tf.keras.layers.LeakyReLU(alpha=0.2), sampling="down")
        
        # decoder
        self.dec_conv1 = CD(k=512, activation=tf.keras.activations.relu, sampling="up")
        self.dec_conv2 = CD(k=1024, activation=tf.keras.activations.relu, sampling="up")
        self.dec_conv3 = CD(k=1024, activation=tf.keras.activations.relu, sampling="up")
        self.dec_conv4 = C(k=1024, activation=tf.keras.activations.relu, sampling="up")
        self.dec_conv5 = C(k=1024, activation=tf.keras.activations.relu, sampling="up")
        self.dec_conv6 = C(k=512, activation=tf.keras.activations.relu, sampling="up")
        self.dec_conv7 = C(k=256, activation=tf.keras.activations.relu, sampling="up")
        self.dec_conv8 = C(k=128, activation=tf.keras.activations.relu, sampling="up")
        
        self.out = tf.keras.layers.Conv2D(output_dim, kernel_size=3, strides=1, activation=tf.keras.activations.tanh, padding='same')
        
    def call(self, x):
        # encoder
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(x1)
        x3 = self.enc_conv3(x2)
        x4 = self.enc_conv4(x3)
        x5 = self.enc_conv5(x4)
        x6 = self.enc_conv6(x5)
        x7 = self.enc_conv7(x6)
        x8 = self.enc_conv8(x7)
        
        #decoder
        x = self.dec_conv1(x8)
        x = self.dec_conv2(tf.keras.layers.concatenate([x, x7]))
        x = self.dec_conv3(tf.keras.layers.concatenate([x, x6]))
        x = self.dec_conv4(tf.keras.layers.concatenate([x, x5]))
        x = self.dec_conv5(tf.keras.layers.concatenate([x, x4]))
        x = self.dec_conv6(tf.keras.layers.concatenate([x, x3]))
        x = self.dec_conv7(tf.keras.layers.concatenate([x, x2]))
        x = self.dec_conv8(tf.keras.layers.concatenate([x, x1]))
        
        # get three channels
        x = self.out(x)
        return x
```

### Discriminator


### 3.1 Experiments
#### facades
#### winter to summer
#### sparse mono depth perception
#### sparse to dense depthmap
#### image unblurring

## 4 Visualization and Results
