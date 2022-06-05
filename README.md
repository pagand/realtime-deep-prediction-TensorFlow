## Notes:
This repository is inspired from the following github pages:
- [FaceNet](https://github.com/davidsandberg/facenet)
- [UTKFace](https://susanqq.github.io/UTKFace/) 
- [Race/Gender recognition](https://github.com/zZyan/race_gender_recognition)
- [Age/Gender Estimate](https://github.com/BoyuanJiang/Age-Gender-Estimate-TF)

More information: https://upaspro.com/tesorflow-facenet/

## Dependencies

Install dlib with instruction from [dlib](https://github.com/charlielito/install-dlib-python-windows).
``` 
conda create --name <myenv> python=3.5
conda activate <myenv>
pip install opencv-python
pip install -r requirements.txt
``` 

## Checkpoints
- [Race/Gender](https://drive.google.com/drive/folders/1DAjb04wEktKR8lx32DNtma-QnN23Aa3l?usp=sharing)
- [ARG](https://drive.google.com/drive/folders/1jk3tr4UzCIwPGdGffwc0pWriAmU8yfwo?usp=sharing)

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
Put UTKFace folder in the same level of hierarchy of either two version of CMPT726 (namely ARG or RaceGender)

### Raw
**gender_labels**   
0: male, 1: female  
{'0': 12391, '1': 11317}


### Scripts
```
make_features.py
```
Create a <data> folder inside CMPT726.
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
Before running:
- Create 'models' inside 'CMPT726' directory
- download the pretrained [VGGFace2](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) weights
- create 'detectors' inside 'models' directory
- create 'logs' inside 'CMPT726' folder
- create 'model_1' and 'model_2' inside the 'logs' directory
  
Functionality
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
Dependencies: dlib
Also, might have to use an older version of python.
- preprocess: chop and align the face
- predict use saved model

## Baseline models 
Training parameters after 50 epocks: 0.63, 0.38, and 0.15 cross-entropy loss for age, race and gender evaluation, respectively.  
Validation accuracy with bining: 73, 84, 93% for age, race and gender prediction, respectively.
