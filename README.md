# Pelagic-bird-project

This is my project to use a machine learning model to understand the influences of weather patterns on pelagic bird migration. I have worked on creating a model for the Northern Gannet, so the code provided here is for gannet observations at Rockport, on the Massachusetts coast. I am working on hyperparameter tuning for my current Multilayer Perceptron (MLP) model, and I am planning to expand my project to other pelagic birds found in the northwest Atlantic. 
The MLP model I created predicts the number of gannets that are expected to be seen at Rockport on any given day. The input of the model is the weather data, specifically wind direction and wind speed, and the output of the model is the expected number of gannets. 
I used a custom eBird Basic Dataset (EBD) (see [1]) consisting of reports of Northern Gannets (NOGA) from citizen scientists in Massachusetts. My weather dataset is from the Meteostat library (see [2]) which takes data from the National Weather Service. I also used eBird’s Sampling Event Dataset (SED) which contains all bird reports in Massachusetts; once filtered to Rockport, this dataset provides additional data of when gannets were not seen from the coast. 
There are three files I created in the folder ‘code’, one for pre-processing the NOGA observation dataset, one for creating the corresponding weather dataset, and one for creating and training the machine learning model. 
The eBird Basic Dataset for NOGA is included in the ‘dataset’ folder. The two files of NOGA observations were created after filtering the original dataset, and the weather dataset is also contained in the ‘dataset’ folder. 


1.	Northern Gannet Dataset (‘noga_observations_dataset_github.ipynb’)

This code takes the NOGA EBD and filters out the unwanted data to only keep the reports of gannets along the Massachusetts coast. It then combines the resulting dataset with the SED filtered to the Massachusetts coast, so that all observations at the Massachusetts coast, with and without gannets, are included. The final dataset is ‘noga_obsv_200601-202312_masscoast.csv’ (NOGA dataset for the MA coast), located in the ‘dataset’ folder.


2.	Weather Dataset (‘noga_weather_dataset_github.ipynb’)

This code formats the month, day, and year of each observation in the NOGA dataset for the MA coast and splits them into three columns. It then converts the days into two-component vectors, linearizing the relationship between days and the seasonal abundance of gannets, and keeping any pair of adjacent days equally spaced. The data is then undersampled so as to keep every 50th zero count in the NOGA dataset. This NOGA dataset is used to create the corresponding weather dataset using data from the National Weather Service through the Meteostat library. The parameters that are included in the weather dataset are temperature, precipitation, wind direction, and wind speed taken every 3 hours from 12AM to 12PM on the day of the gannet observation. Although these 4 parameters are included, I chose to only use the wind speed and direction from 6AM and 9AM as they appear to show a stronger correlation with Northern Gannet counts. The final weather dataset is ‘noga_weather_dataset_github.ipynb’, and is located in the ‘dataset’ folder.


3.	Multilayer Perceptron (‘noga_mlp_model_github.ipynb’)

In this code, the datasets are pre-processed and filtered to a coastal region of Rockport, MA that includes Halibut Point and Andrews Point. The gannet counts are linearized by applying the logarithm function, and the weather data is filtered to include wind speed and direction readings from 6AM and 9AM on the day of each observation in the NOGA dataset. All of the data is normalized, the date vector components are included in the weather dataset, and the datasets are converted to NumPy arrays for model training. I chose the Multilayer Perceptron (MLP) model due to its capability of handling nonlinear data. The MLP model takes weather data as the input of the model and gannet count as the output. It uses a custom loss function, that gives a lower loss value when the predicted gannet count is higher than the actual gannet count compared to when the prediction is lower than the actual count. This helps the model not have a bias of predicting zero counts most of the time. Note that even though the data was undersampled, it is still skewed as there is relatively little data with high counts. The hyperparameters for the MLP model were chosen based on a paper by Panchal et. al. (see [3]). Three dense layers were used and all values for the number of hidden neurons between 1 and 12 were tested along with a Sigmoid, ReLU, and Linear activation function, and the model loss and R-Squared scores were recorded. The model is trained for 100 epochs. I also tried using the XGBoost model (not included in this GitHub project), which gave moderate accuracy. The weather dataset for testing the MLP model is created using the same code as in the weather dataset file on GitHub. With the prediction function, a location can be input (currently, this only works for Rockport, MA) and a day of the year can be provided, and the model will predict the number of gannets expected to be seen from the coast at Rockport, MA on that day. 

### Sources

[1] eBird Basic Dataset. Version: EBD_relJan-2024. Cornell Lab of Ornithology, Ithaca, New York. Jan 2024.

[2] Meteostat Hourly Weather Data, https://dev.meteostat.net/python/hourly.html

[3] Panchal, G., Ganatra, A., Kosta, Y.P. and Panchal, D., 2011. Behaviour analysis of multilayer perceptrons with multiple hidden neurons and hidden layers. International Journal of Computer Theory and Engineering, 3(2), pp.332-337.
