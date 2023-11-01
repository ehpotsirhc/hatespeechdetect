# Hate Speech Classification Reproducibility Study
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



## To Run the Reproducibility Study - BERT
#### Data Preparation for BERT
1) Clone the repository  
2) CD into the SBIC dataset directory (datasets/SBIC)
3) The SBIC data is already preprocessed from its 
[original form](https://maartensap.com/social-bias-frames/), and included 
in the repository for your convenience. Run the additoinal scripts following 
the instructions below to prepare the data for the BERT versus XClass comparison:
```
cd hatespeechdetect/datasets/SBIC/
python3 csv2txt.py SBIC.v2.agg.cmb_processed.csv
python3 csv2classes.py SBIC.v2.agg.cmb_processed.csv
```
#### Running BERT
1) CD into the BERT classifier directory
2) Run main.py
```
cd hatespeechdetect/classifiers/BERT/src
python3 main.py
```
Optionally, you may supply your own dataset. Your dataset must follow the 
format of the supplied dataset `datasets/SBIC/SBIC.v2.agg.cmb_processed.csv`. 
To run your own dataset using our supplied BERT, use the following command:
```
python3 main.py <dataset.csv>
```
Finally, to skip the training process and only retrieve the model results, 
you may follow the command below. Do note that you must have run the training 
at least once and have a model available in order to be able to skip directly 
to the results analysis.
```
python3 main.py --testing-only
   or
python3 main.py <dataset.py> --testing-only
```


## To Run Using Demo (Classifier-Provided) Datasets...
1) Clone the repository  
2) Download, extract and move the data into the right place  
3) Run the XClass tests to reproduce the results  
- for "run.sh", arg0=GPU_number, arg1=Dataset
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

