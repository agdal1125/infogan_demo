# infogan_demo
infogan demo module for crevasse AI


#### modified codes from: https://github.com/conan7882/tf-gans

### File Structure
    .
    ├── data                # your input & output data dir
    |   └──  celebA         # your celebA image data should be here
    |   |     └── README.md    # check README.md for more instructions
    |   |  
    |   └──  MNIST_data     # your MNIST_data should be here
    |   |     └── README.md    # check README.md for more instructions
    |   └──  out
    |   |     └──  gans
    |   |     |     └── infogan  # your output generated data is stored here
    | 
    ├── tf-gans            
    |     └── examples
    |     |     └── gans.py  # your python script to run for testing and predicting    
    |     └──  src
    |     └──  ...
    |     
    └── README.md
 
### Requirements

- Python 3.3+
- Tensorflow 1.10+
- TensorFlow Probability
- numpy
- Scipy
- Matplotlib
- skimage
- pillow
- imageio 2.4.1+

### Installation
`bash
$ git clone https://github.com/agdal1125/infogan_demo/
`
+ install all modules in the requirements


### Data Preparation

#### MNIST data 
###### (You can download them here: http://yann.lecun.com/exdb/mnist/)

1. Place your MNIST_data in infogan_demo/data/MNIST_data/ 
2. Delete README.md in that folder

The folder should look like this:

    .
    |   └──  MNIST_data     
    |   |     └── t10k-images-idx3-ubtye.gz
    |   |     └──  t10k-labels-idx1-ubtye.gz
    |   |     └──  train-images-idx3-ubtye.gz
    |   |     └──  train-labels-idx1-ubtye.gz

#### CelebA data 
###### (You can download image files here: https://www.kaggle.com/jessicali9530/celeba-dataset/downloads/img_align_celeba.zip/2)

1. Download the data
2. unzip *img_align_celeba.zip*
3. Move all the images to --> infogan_demo/data/celebA/
4. Delete README.md in that folder

The folder should look like this:

    .
    |   └──  celebA     
    |   |     └── 000001.jpg
    |   |     └──  000002.jpg
    |   |     └── ...


### General Usage

-- CelebA

`
$ cd YOUR_WORKING_DIRECTORY/infogan/tf-gans/examples/
$ python3 gans.py --train --dataset celeba --dir YOUR_WORKING_DIRECTORY/infogan_demo
`

-- mnist

`
$ cd /home/nowgeun1/Desktop/infogan/tf-gans/examples/
$ python3 gans.py --train --dataset mnist --dir YOUR_WORKING_DIRECTORY/infogan_demo
`



   
### Server Usage (for Crevasse)

1. 개발환경 접속
`
$source activate infogan
`

2. 훈련

-- CelebA

`
$ cd /home/nowgeun1/Desktop/infogan/tf-gans/examples/
$ python3 gans.py --train --dataset celeba (CelebA데이터)
`

-- mnist

`
$ cd /home/nowgeun1/Desktop/infogan/tf-gans/examples/
$ python3 gans.py --train --dataset mnist (mnist 데이터)
`

3. train log file & generated된 이미지 저장 위치:
/home/nowgeun1/Desktop/infogan/data/out/gans/infogan/
  
