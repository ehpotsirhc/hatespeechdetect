# Hate Speech Detection Reproducibility Study
## GMU CS678 Final Project<br>Iris Chen • Christophe Leung • Kelvin Lu


## Overview
In this project, we explore the reproducibility of [Weakly-Supervised Hate Speech Classification](https://aclanthology.org/2023.woah-1.4.pdf). Specifically, we attempt to reproduce the experiments outlined in the paper and analyze whether weakly-supervised models can, in practice, outperform traditional supervised models.  


## Requirements
Python 3.6 - 3.11; tested on 3.10.7  
The following PIP dependencies:
```
torch
cuda-python
numpy
scipy
tqdm
scikit-learn
sentencepiece
transformers
tensorboard
```


## To Run...
1) Clone the repository  
2) Download, extract and move the data into the right place  
3) Run the XClass tests to reproduce the results  
- for "run.sh", arg0=CPU_number, arg1=Dataset
- datasets have been aggregate on a [personal server](https://pineapple.wtf/hate-speech-detection-reproducibility/) for easy access; the scripts will download from this server. You may alternatively opt to download the datasets from each of the classifiers' original data sources by following the instructions provided within each classifier's readme page.

Follow the steps below to download and run the tests...
```
cd hatespeechdetect/datasets/
./data_download.sh
./data_extract.sh
./data_install.sh

cd ../classifiers/XClass/scripts/
./run.sh 0 Yelp
```

If you run into an error or would like to delete all of the downloaded data, you can run the following script
```
./data_clean.sh
```


## Resources
#### Project Guidelines<br><https://nlp.cs.gmu.edu/course/cs678-fall23/project/>   
#### Paper<br><https://aclanthology.org/2023.woah-1.4.pdf>  
#### Code (Currently Private)<br><https://github.com/ehpotsirhc/hatespeechdetect>


## Original Classifier Repositories
https://github.com/yumeng5/LOTClass  
https://github.com/yumeng5/WeSTClass  
https://github.com/ZihanWangKi/XClass  

