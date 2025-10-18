🧠 RAN-with-CBAM  
     Residual Attention Network integrated with Convolutional Block Attention Module for Image Classification

 Table of Contents  
- [Project Overview](#project-overview)   
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Installation & Setup](#installation--setup)   

📖 Overview  
    This project implements a Residual Attention Network (RAN) enhanced with a Convolutional Block Attention Module (CBAM) for image classification tasks.  
    By combining RAN’s hierarchical attention mechanism with CBAM’s spatial and channel attention, this model improves feature representation and overall classification accuracy — demonstrated on the CIFAR-10 dataset.

⚙️ Features  
    - 🧩 Implementation of Residual Attention Network (RAN).  
    - 🔍 Integration of CBAM (Convolutional Block Attention Module).  
    - 🖼️ Training and testing on the **CIFAR-10** dataset.  
    - 📊 Comparison of model accuracy with and without CBAM.  
    - 🔁 Modular, extensible architecture for future attention mechanisms.  

🧰 Tech Stack  
    - Language: Python  
    - Framework: PyTorch  
    - Libraries:  
    - `torch`, `torchvision`, `numpy`, `matplotlib`  
    - Dataset: CIFAR-10 (included in repository as `cifar-10-python/`)  

🚀 Installation & Setup
   1. Clone the repository
    
          git clone https://github.com/Gayathri2113/RAN-with-CBAM.git
          cd RAN-with-CBAM

   2. Install dependencies
      
          pip install torch torchvision numpy matplotlib
      
   4. Train the model

          python cv.py --model train --dataset cifar10 --epochs 50 --batch_size 128
    
   5.  Evaluate the model

           python cv.py --model eval --model_path path/to/saved_model.pth
    

