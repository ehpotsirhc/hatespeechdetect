#!/bin/bash

printf "----------------------------------------------------------------------"
printf "=== Extracting LOTClass Data... ===\n"
unzip downloaded/LOTClass/LOTClass_agnews.zip -d downloaded/LOTClass/agnews/
unzip downloaded/LOTClass/LOTClass_amazon.zip -d downloaded/LOTClass/amazon/
unzip downloaded/LOTClass/LOTClass_dbpedia.zip -d downloaded/LOTClass/dbpedia/
unzip downloaded/LOTClass/LOTClass_imdb.zip -d downloaded/LOTClass/imdb/

printf "----------------------------------------------------------------------"
printf "=== Extracting WeSTClass Data... ===\n"
unzip downloaded/WeSTClass/WeSTClass_agnews.zip -d downloaded/WeSTClass/
unzip downloaded/WeSTClass/WeSTClass_yelp.zip -d downloaded/WeSTClass/

printf "----------------------------------------------------------------------"
printf "=== Extracting XClass Data... ===\n"
unzip downloaded/XClass/20News.zip -d downloaded/XClass/
unzip downloaded/XClass/AGNews.zip -d downloaded/XClass/
unzip downloaded/XClass/DBpedia.zip -d downloaded/XClass/
unzip downloaded/XClass/NYT-Locations.zip -d downloaded/XClass/
unzip downloaded/XClass/NYT-Small.zip -d downloaded/XClass/
unzip downloaded/XClass/NYT-Topics.zip -d downloaded/XClass/
unzip downloaded/XClass/Yelp.zip -d downloaded/XClass/

