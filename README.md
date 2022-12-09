# Self-Holo
Diffraction model-informed neural network for unsupervised layer-based computer-generated holography.<br>
### [Paper](https://doi.org/10.1364/OE.474137)
X. Shui, H. Zheng, X. Xia, F. Yang, W. Wang, and Y. Yu, “Diffraction model-informed neural network for unsupervised <br> 
layer-based computer-generated holography,” Opt. Express 35(25), (2022). 

## Dataset
The RGB-D datasets are from **[TensorHolography](https://github.com/liangs111/tensor_holography)**.

## High-level Structure
The code is organized as follows:

./src/
* ```train.py``` trains the selfholo.
* ```dataLoader.py``` loads a set of images.
* ```complex_generator.py``` is the target complex_amplitude generator.
* ```holo_encoder.py``` is the phase encoder.
* ```selfholo.py``` is the pipeline of selfholo.
* ```propagation_ASM.py``` contains the angular spectrum method.
* ```perceptualloss.py``` contains mseloss and perceptualloss.
* ```predict.py``` predicts 2D holograms or 3D holograms.
*  ```utils.py``` contains utility functions. 

## Running the test
 ```
 python ./src/train.py  --run_id=selfholo
 ```

## Ackonwledgement
We are thankful for the open source of **[NeuralHolography](https://github.com/computational-imaging/neural-holography)**, 
**[HoloEncoder](https://github.com/THUHoloLab/Holo-encoder)**,and **[HoloEncoder-Pytorch-Version](https://github.com/flyingwolfz/holoencoder-python-version)**.
These works are very helpful for our research.
