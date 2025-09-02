# **Art Generation with Neural Style Transfer**

This repository contains a Jupyter Notebook (Art\_Generation\_with\_Neural\_Style\_Transfer.ipynb) that demonstrates **Neural Style Transfer** using **PyTorch**. The goal of this project is to create new images that combine the artistic style of one image with the content of another. The implementation is a Python-based solution that leverages a pre-trained VGG19 model from the torchvision library.

## **Getting Started**

### **Prerequisites**

* Python 3.x  
* Jupyter Notebook  
* Required Python libraries:  
  * torch  
  * torchvision  
  * matplotlib  
  * PIL (Pillow)

You can install the necessary libraries using pip:

pip install torch torchvision matplotlib Pillow

### **Usage**

1. **Clone the Repository**: Clone this repository to your local machine.  
2. **Open the Notebook**: Open the Art\_Generation\_with\_Neural\_Style\_Transfer.ipynb file in a Jupyter environment.  
3. **Prepare Images**: Place your desired content and style images in the same directory as the notebook. Update the my\_content\_path and my\_style\_path variables in the notebook to point to your image files.  
4. **Run the Cells**: Execute the notebook cells sequentially. The code will load the images, perform the style transfer, and display the final result.  
5. **View Output**: The generated image will be automatically saved in a newly created directory called generated\_images. The filename will be dynamically generated based on the current timestamp to prevent overwriting previous results.

## **How it Works**

The notebook performs style transfer by using a pre-trained VGG19 model as a feature extractor. The process involves defining loss functions for both content and style, and then iteratively optimizing an input image to minimize both losses simultaneously.

* **Content Loss**: Measures the difference in content representation between the input image and the original content image.  
* **Style Loss**: Measures the difference in style representation between the input image and the original style image. This is calculated using the **Gram Matrix**, which captures the correlation of features at different layers.  
* **Optimization**: A **LBFGS optimizer** is used to adjust the input image tensor to minimize the total loss. The process continues for a set number of iterations, gradually transforming the input image until it achieves the desired combination of content and style.

The notebook handles device selection, automatically using a CUDA-enabled GPU if available for faster processing.
