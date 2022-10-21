# Ex.No.1---Data-Preprocessing
##AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


##ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

##PROGRAM:

      import pandas as pd
      df=pd.read_csv("/content/Churn_Modelling.csv")
      df.head()
      df.isnull().sum()
      df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)
      print(df)
      x=df.iloc[:,:-1].values
      y=df.iloc[:,-1].values
      print(x)
      print(y)
      from sklearn.preprocessing import MinMaxScaler
      scaler = MinMaxScaler()
      df1 = pd.DataFrame(scaler.fit_transform(df))
      print(df1)
      from sklearn.model_selection import train_test_split
      xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
      print(xtrain)
      print(len(xtrain))
      print(xtest)
      print(len(xtest))
      from sklearn.preprocessing import StandardScaler
      sc = StandardScaler()
      df1 = sc.fit_transform(df)
      print(df1)

##OUTPUT: 

Dataset 
![192950859-0b6f64ca-6e76-4c10-9830-287232ba545a](https://user-images.githubusercontent.com/83326978/193036226-b96a97c5-f452-47d7-bcc5-a9cb001d91a4.png)


checking for null values:



![192951831-c32485fa-ba7b-4625-ab0c-4361657d6d0b](https://user-images.githubusercontent.com/83326978/193036266-0788b734-593d-4629-9d4f-a041a5209c6a.png)



checking for duplicate values:



![192952680-9c2cf2d3-858b-4bdf-98cf-f356d84a571a](https://user-images.githubusercontent.com/83326978/193036293-082786eb-4c6d-42ac-86bc-14625fca29c6.png)



Describing Data:



![192953487-609c4eef-5d89-46d5-bb2b-0b15a4553082](https://user-images.githubusercontent.com/83326978/193036312-e86a80d9-3c42-43e3-975f-001be42b53ec.png)



Checking for outliers in Exited Column:


![192953810-78b49e03-da5d-44d3-9613-09ddb65b7ef7](https://user-images.githubusercontent.com/83326978/193036342-727e3f9c-d31c-4e3e-add5-c5e41217e4c9.png)



Normalized Dataset 


![192954241-b0d4947c-b534-48ff-b279-375acc32ff23](https://user-images.githubusercontent.com/83326978/193036364-bf80761e-e528-4963-917f-db291d69e1fa.png)



Describing Normalized Data: 



![192954334-58f7419a-c49d-4b1a-acf8-36ef015a6f27](https://user-images.githubusercontent.com/83326978/193036391-e3e4076f-cdc7-4ac6-b2ee-b4e8d4404b42.png)


X_test and Y_test values:



![190868454-d5e8ba89-62d7-49a9-9e00-67890302112a](https://user-images.githubusercontent.com/83326978/193037395-a326866e-1e96-4cc5-b5f3-d64eeb7edffd.png)




##RESULT


Thus the above program for standardizing the given data was implemented successfully
