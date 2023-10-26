#!/bin/bash

printf "----------------------------------------------------------------------"
printf "=== Extracting LOTClass Data... ===\n"
mv downloaded/LOTClass/agnews/* ../classifiers/LOTClass/datasets/agnews/
mv downloaded/LOTClass/amazon/* ../classifiers/LOTClass/datasets/amazon/
mv downloaded/LOTClass/dbpedia/* ../classifiers/LOTClass/datasets/dbpedia/
mv downloaded/LOTClass/imdb/* ../classifiers/LOTClass/datasets/imdb/

printf "----------------------------------------------------------------------"
printf "=== Extracting WeSTClass Data... ===\n"
mv downloaded/WeSTClass/agnews/ ../classifiers/WeSTClass/
mv downloaded/WeSTClass/yelp/ ../classifiers/WeSTClass/

printf "----------------------------------------------------------------------"
printf "=== Extracting XClass Data... ===\n"
mv downloaded/XClass/20News/* ../classifiers/XClass/data/datasets/20News/
mv downloaded/XClass/AGNews/* ../classifiers/XClass/data/datasets/AGNews/
mv downloaded/XClass/DBpedia/* ../classifiers/XClass/data/datasets/DBpedia/
mv downloaded/XClass/NYT-Locations/* ../classifiers/XClass/data/datasets/NYT-Locations/
mv downloaded/XClass/NYT-Small/* ../classifiers/XClass/data/datasets/NYT-Small/
mv downloaded/XClass/NYT-Topics/* ../classifiers/XClass/data/datasets/NYT-Topics/
mv downloaded/XClass/Yelp/* ../classifiers/XClass/data/datasets/Yelp/

