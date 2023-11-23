#!/bin/bash

printf "----------------------------------------------------------------------"
printf "=== Downloading LOTClass Data... ===\n"
wget -P ./downloaded/LOTClass/ https://pineapple.wtf/content/files/2023/10/LOTClass_agnews.zip
wget -P ./downloaded/LOTClass/ https://pineapple.wtf/content/files/2023/10/LOTClass_amazon.zip
wget -P ./downloaded/LOTClass/ https://pineapple.wtf/content/files/2023/10/LOTClass_dbpedia.zip
wget -P ./downloaded/LOTClass/ https://pineapple.wtf/content/files/2023/10/LOTClass_imdb.zip

printf "----------------------------------------------------------------------"
printf "=== Downloading WeSTClass Data... ===\n"
wget -P ./downloaded/WeSTClass/ https://pineapple.wtf/content/files/2023/10/WeSTClass_agnews.zip
wget -P ./downloaded/WeSTClass/ https://pineapple.wtf/content/files/2023/10/WeSTClass_yelp.zip

printf "----------------------------------------------------------------------"
printf "=== Downloading XClass Data... ===\n"
wget -P ./downloaded/XClass/ https://pineapple.wtf/content/files/2023/10/20News.zip
wget -P ./downloaded/XClass/ https://pineapple.wtf/content/files/2023/10/AGNews.zip
wget -P ./downloaded/XClass/ https://pineapple.wtf/content/files/2023/10/DBpedia.zip
wget -P ./downloaded/XClass/ https://pineapple.wtf/content/files/2023/10/NYT-Locations.zip
wget -P ./downloaded/XClass/ https://pineapple.wtf/content/files/2023/10/NYT-Small.zip
wget -P ./downloaded/XClass/ https://pineapple.wtf/content/files/2023/10/NYT-Topics.zip
wget -P ./downloaded/XClass/ https://pineapple.wtf/content/files/2023/10/Yelp.zip

