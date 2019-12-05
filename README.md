A more detailed explaination and report can be found in [Gender and Race recognition: Transfer, Multi-task Learning for the Laziest](https://medium.com/@zhuyan/gender-and-race-recognition-transfer-multi-task-learning-for-the-laziest-88316e6e492).

## Dependencies
(after installing miniconda which comes with essential python packages)
1. Tensorflow (added path to CUDA toolkit)


## Dataset
### UTF faces (aligned and cropped):
[Source](https://susanqq.github.io/UTKFace/)

The labels of each face image is embedded in the file name, formated like [age]\_[gender]\_[race]\_[date&time].jpg

- [age] is an integer from 0 to 116, indicating the age
- [gender] is either 0 (male) or 1 (female)
- [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
- [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

Handle error files from UTKFace:
``` 
mv 39_1_20170116174525125.jpg.chip.jpg 39_0_1_20170116174525125.jpg.chip
mv 61_1_20170109150557335.jpg.chip.jpg 61_1_3_20170109150557335.jpg.chip
mv 61_1_20170109142408075.jpg.chip.jpg 61_1_1_20170109142408075.jpg.chip
```            

### Raw
**gender_labels**   
0: male, 1: female  
{'0': 12391, '1': 11317}

**race_labels**  
White, Black, Asian, Indian, and Others  
{'0': 10078, '1': 4528, '2': 3434, '3': 3976, '4': 1692}

### Augmentation
0: no, 1: x2, 2: x3, 3: x3, 4: x4  

- **train genders**  
Counter({'0': 22188, '1': 21128})    
- **val gender**  
Counter({'0': 2473, '1': 2336})  
- **train races**  
Counter({'3': 10809, '2': 9270, '0': 9057, '1': 8156, '4': 6024})  
- **val race**  
Counter({'3': 1116, '2': 1032, '0': 1021, '1': 896, '4': 744})  


### Scripts
```
make_features.py
```
Functionality: 
- get images from directory: `UTKFace/*.jpg`
- split data in to training, validation and test set
- data augmentation to balance the representation each race in the total population
- save to `*.tfrecords* files

```
multitask_model.py
```
Functionality: 
- read_and_decode: read, decode data from training `.tfrecord`; and preprocess the data ()
- build_model: load network graph and add layers
- a few options of how to add layers, `add_layer` as the most basic one, `add_layer_v2` added additional FC before auxiliary branches
- compute losses

```
trainer.py

python trainer.py --project_dir /data/gender_race_face/ --model_name model1_gender_0001 --learning_rate 0.0001 --num_epoch 2
```  
- set training parameters and log directory
- train steps
- log to tensorboard 

```
evaluator.py

python evaluator.py --project_dir /data/gender_race_face/ --model_name model1_gender_0001 
```
Functionality:  
- evaluate trained model
- log to tensorboard


```
run_evaluator.sh
```
Functionality:  
- run specified evaluator 50 times in a loop

```
model_analyst.py

python model_analyst.py --experiments_log experiemnts_combined4.txt --model_name model2_combined --project_dir /data/gender_race_face/
```
Dependencies: numpy   
Functionality:  
- if `--experiments_log` is not null, stats for multiple runs of the model will be computed
- if `--model_name` is not null, predictions will be genereated and saved in pickle file 
- a hidden function `additional_analysis` will take a pickle file and compute confusion matrix and draw montages for detailed error analysis
- for `additional_analysis`, additional dependences: `imutils, open-cv` are required

```
model_comparator.py

python model_comparator.py ----gender_logs experiemnts_gender.txt --race_logs experiemnts_gender.txt --multitask_logs experiemnts_combined4.txt
```
Dependencies: scipy   
Functionality:  
- if `--gender_logs and multitask_logs` are not null, t test is performed to compare gender models
- if `--race_logs and multitask_logs` are not null, t test is performed to compare gender models


```
predictor.py
```
Dependencies:
1. dlib.
Install on Windows with instruction from https://github.com/charlielito/install-dlib-python-windows.
2. cv2
Also, might have to use an older version of python.
- preprocess: chop and align the face
- predict use saved model

## Baseline models 
### model1_gender
0.0001 lr   
INFO:tensorflow:Average gender Accuracy: 0.94431586   
### model1_race
INFO:tensorflow:Average gender Accuracy: 0.8644865   

## Best multi-task model
lower learning rate for all pretrained layer, mid lr for gender layer   
batch normalization for added fc layers   
INFO:tensorflow:Average gender Accuracy: 0.951493   
INFO:tensorflow:Average race Accuracy: 0.87557212   

