#!/bin/bash

echo "----------------------------------------------------------------------"
printf "=== Extracting LOTClass Data... ===\n"
cp downloaded/LOTClass/agnews/* ../classifiers/LOTClass/datasets/agnews/
cp downloaded/LOTClass/amazon/* ../classifiers/LOTClass/datasets/amazon/
cp downloaded/LOTClass/dbpedia/* ../classifiers/LOTClass/datasets/dbpedia/
cp downloaded/LOTClass/imdb/* ../classifiers/LOTClass/datasets/imdb/

echo "----------------------------------------------------------------------"
printf "=== Extracting WeSTClass Data... ===\n"
cp downloaded/WeSTClass/agnews/ ../classifiers/WeSTClass/
cp downloaded/WeSTClass/yelp/ ../classifiers/WeSTClass/

echo "----------------------------------------------------------------------"
printf "=== Extracting XClass Data... ===\n"
cp downloaded/XClass/20News/* ../classifiers/XClass/data/datasets/20News/
cp downloaded/XClass/AGNews/* ../classifiers/XClass/data/datasets/AGNews/
cp downloaded/XClass/DBpedia/* ../classifiers/XClass/data/datasets/DBpedia/
cp downloaded/XClass/NYT-Locations/* ../classifiers/XClass/data/datasets/NYT-Locations/
cp downloaded/XClass/NYT-Small/* ../classifiers/XClass/data/datasets/NYT-Small/
cp downloaded/XClass/NYT-Topics/* ../classifiers/XClass/data/datasets/NYT-Topics/
cp downloaded/XClass/Yelp/* ../classifiers/XClass/data/datasets/Yelp/

