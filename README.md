# ML2018_410321114_2

# 作業說明
  在本次作業中，我分別使用CNN、RNN及MLP，進行比較。  
  其中data資料夾中是mnist_handwriting的training Data和testing Data  
  mnist_train.py是我的training code  
  testing_data資料夾中的圖片是我嘗試下載的一組手寫數字  
  create_list.sh是用來產生testing.txt用方便將所有圖片一次丟入  
  testing.py是我嘗試下載一組手寫數字並試著丟入各個model中進行辨識用的code  
# Data下載
  在這裡我是直接透過keras的Datasets直接下載到指定的目錄中︰  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/data_download.png)  
# 預處理Data
  在這裡我的Data進行了簡單的reshape以及將像素質標準化︰  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/preprocessing1.png)  
  為了配合RNN及MLP的輸入，所以我另外進行reshape︰  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/preprocessing2.png)  
  (RNN)  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/preprocessing3.png)  
  (MLP)  
# 減少維數
  我只在CNN中進行pooling以減少維數︰  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/pooling.png)  
# 選擇classifier
  本次作業中，我分別使用CNN、RNN及MLP，來進行比較，以下分別是我的CNN、RNN及MLP的架構︰  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/CNN.png)  
  (CNN)  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/RNN.png)  
  (RNN)  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/MLP.png)  
  (MLP)  
# 效能評估
  以下是我的Training結果︰  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/CNN_Train.png)  
  (CNN)  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/RNN_Train.png)  
  (RNN)  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/MLP_Train.png)  
  (MLP)  
  根據以上的結果可以看出，CNN擁有較高的準確率，且loss也較低，而且與RNN及MLP不同，只需要epoch一次就可以得出很高的準確率  
# 其他
  我另外有抓了一組手寫數字嘗試直接進行辨識，以下是我抓的手寫數字︰  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/testing.jpg)  
  (取自︰https://imgur.com/XAHGrhJ)  
  以下是我用各個model進行辨識的結果︰
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/CNN_testing.png)  
  (CNN)  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/RNN_testing.png)  
  (RNN)  
  ![error](https://github.com/weiwow123/ML2018_410321114_2/blob/master/readme_data/MLP_testing.png)
  (MLP)  
  根據以上的結果，整合成以下的表格︰
  CNN:  
  
  |     |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  | 100 |  0  | 
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
  |  1  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  | 100 |  0  |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
  |  2  |  0  |  0  |94.58|  0  |  0  |  0  |  0  |  0  | 5.41|  0  |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
  |  3  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  | 100 |  0  |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
  |  4  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  | 100 |  0  |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
  |  5  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  | 100 |  0  |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
  |  6  |  0  |  0  |0.001|  0  |  0  |  0  |  0  |  0  |99.99|  0  |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
  |  7  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  | 100 |  0  |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
  |  8  |  0  |  0  |81.85|  0  |  0  |  0  |  0  |  0  |18.15|  0  |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
  |  9  |  0  |  0  |0.001|  0  |  0  |  0  |  0  |  0  |99.99|  0  |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
