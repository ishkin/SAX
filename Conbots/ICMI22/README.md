# XGboost classifier for engagement in motor learning



The code contains two functions:

predict_from_existing_model

  This functions upload the pre-trained XGboost classifer and use it to predict the given test set.

  Function input
  1.  pre-trained model 
  2.  numpy matrix of the test data. The rows of the matrix are the samples while the columns represents the sensors featers
        The last column is the lable - the level of engadgment
        The features are (ordered):
        
```
IMU_B_1_jerk_rmssd , GYR_B_2_jerk_median , imu_b_jerk_magnitude_kurt , GYR_B_1_kurt , IMU_A_0_jerk_rmssd , gyr_a_jerk_magnitude_mean , IMU_A_0_rmssd , IMU_B_2_var , GYR_B_0_var , IMU_A_2_skew , GYR_B_1_jerk_var , imu_a_jerk_magnitude_rmssd , hrv_sdsd , IMU_B_1_jerk_kurt , GYR_B_2_jerk_kurt , hrv_total_power , GYR_A_2_jerk_skew , IMU_A_1_mean , IMU_B_1_skew , gyr_a_magnitude_median , IMU_A_1_jerk_median , IMU_A_1_jerk_skew , imu_a_jerk_magnitude_kurt , gyr_b_magnitude_var , IMU_B_0_kurt , imu_b_jerk_magnitude_skew , imu_a_magnitude_kurt , IMU_B_0_jerk_mean , GYR_A_2_jerk_median , GYR_B_2_jerk_skew , imu_b_magnitude_median , IMU_A_2_jerk_median , GYR_B_0_median , GYR_B_1_var , gsr_SHF_mean , IMU_A_1_iqr , gyr_b_jerk_magnitude_median , hrv_hf , IMU_B_1_jerk_mean , GYR_A_0_jerk_rmssd , gyr_a_jerk_magnitude_iqr , IMU_A_2_jerk_rmssd , IMU_B_1_kurt , GYR_B_2_rmssd , GYR_A_1_rmssd , GYR_A_2_mean , gyr_a_magnitude_kurt , IMU_B_0_jerk_median , GYR_A_1_iqr , IMU_B_2_jerk_rmssd , IMU_A_2_rmssd , gsr_SHS_var , IMU_B_0_jerk_var , gyr_b_magnitude_median , GYR_A_0_skew , IMU_A_1_jerk_rmssd , GYR_A_0_median , GYR_B_2_jerk_mean , imu_a_jerk_magnitude_var , IMU_A_2_jerk_iqr , IMU_B_2_rmssd , GYR_A_1_var , GYR_B_0_skew , gyr_a_jerk_magnitude_median , GYR_B_1_jerk_mean , IMU_A_1_jerk_iqr , gsr_SHA_mean , GYR_A_2_jerk_var , IMU_B_0_median , IMU_A_2_median , GYR_B_0_kurt , gyr_a_magnitude_skew , IMU_A_1_jerk_mean , imu_b_magnitude_kurt , GYR_A_1_jerk_var , IMU_B_0_skew , GYR_A_0_jerk_skew , GYR_A_1_jerk_skew , GYR_A_0_iqr , gyr_b_jerk_magnitude_mean , imu_a_magnitude_mean , GYR_A_0_jerk_mean , GYR_A_1_jerk_iqr , IMU_B_2_jerk_skew , GYR_B_0_rmssd , hrv_nni_50 , GYR_A_1_mean , imu_b_jerk_magnitude_var , GYR_B_1_jerk_kurt , imu_a_magnitude_rmssd , GYR_A_1_jerk_kurt , IMU_A_2_jerk_kurt , imu_a_magnitude_skew , GYR_B_2_mean , imu_b_magnitude_iqr , GYR_A_1_jerk_median , GYR_A_0_rmssd , IMU_A_2_jerk_skew , imu_b_jerk_magnitude_mean , gyr_b_jerk_magnitude_iqr , gyr_b_magnitude_mean , GYR_B_1_jerk_skew , hrv_pnni_50 , imu_a_jerk_magnitude_median , GYR_B_0_jerk_var , IMU_B_2_jerk_var , GYR_A_2_median , gyr_b_magnitude_kurt , hrv_sd1 , IMU_B_2_jerk_median , GYR_A_0_var , IMU_B_2_skew , imu_b_magnitude_rmssd , GYR_B_1_skew , GYR_B_1_median , GYR_B_1_jerk_rmssd , IMU_B_1_var , gyr_a_magnitude_var , GYR_A_1_kurt , GYR_B_0_jerk_kurt , imu_a_jerk_magnitude_skew , imu_a_magnitude_median , gyr_a_magnitude_mean , GYR_B_2_jerk_var , gyr_b_magnitude_rmssd , IMU_A_1_kurt , IMU_A_2_mean , IMU_A_2_iqr , hrv_range_nni , gsr_SHA_ae_2 , GYR_B_2_var , gyr_a_magnitude_rmssd , gyr_a_magnitude_iqr , hrv_mean_nni , IMU_B_1_jerk_skew , hrv_vlf , gsr_SHF_ae_3 , IMU_B_0_jerk_rmssd , IMU_B_1_iqr , IMU_A_0_kurt , IMU_A_2_var , gsr_SHA_ae_3 , IMU_A_2_kurt , IMU_B_2_jerk_mean , gsr_SHF_var , gsr_SHS_mean , IMU_B_2_iqr , GYR_B_2_jerk_rmssd , IMU_A_0_skew , imu_a_magnitude_var , IMU_A_1_skew , GYR_A_2_jerk_iqr , imu_b_jerk_magnitude_iqr , hrv_lf , IMU_B_2_median , IMU_B_0_var , GYR_A_2_jerk_rmssd , gsr_SHS_ae_3 , IMU_A_0_jerk_kurt , IMU_B_1_jerk_iqr , hrv_pnni_20 , GYR_B_0_mean , gyr_a_jerk_magnitude_kurt , IMU_A_0_var , GYR_B_0_iqr , GYR_A_0_jerk_kurt , GYR_A_2_rmssd , imu_a_jerk_magnitude_iqr , IMU_A_1_jerk_var , GYR_A_2_var , imu_b_magnitude_mean , GYR_B_2_kurt , GYR_B_0_jerk_median , IMU_B_1_rmssd , GYR_B_1_jerk_iqr , imu_b_jerk_magnitude_rmssd , IMU_B_0_rmssd , GYR_B_1_iqr , GYR_A_2_iqr , gsr_SHS_ae_2 , IMU_B_1_median , GYR_B_2_skew , GYR_B_1_mean , IMU_B_2_jerk_iqr , IMU_A_0_jerk_iqr , GYR_B_2_median , IMU_B_1_mean , GYR_B_1_rmssd , gyr_b_magnitude_skew , IMU_B_0_mean , gsr_SHA_ae_0 , GYR_B_1_jerk_median , IMU_B_2_jerk_kurt , gyr_b_jerk_magnitude_skew , gyr_a_jerk_magnitude_skew , IMU_A_0_mean , IMU_A_0_jerk_var , IMU_A_0_iqr , GYR_B_0_jerk_iqr , IMU_B_2_mean , IMU_A_1_jerk_kurt , IMU_A_1_var , gsr_SHF_ae_1 , gyr_b_magnitude_iqr , IMU_B_0_jerk_kurt , GYR_A_1_jerk_mean , GYR_A_1_median , IMU_B_0_jerk_skew , GYR_B_0_jerk_mean , GYR_B_2_jerk_iqr , IMU_A_0_median , IMU_B_0_iqr , IMU_B_0_jerk_iqr , gsr_SHA_ae_1 , IMU_B_1_jerk_median , gyr_b_jerk_magnitude_var , GYR_A_2_skew , gyr_a_jerk_magnitude_rmssd , imu_a_jerk_magnitude_mean , gyr_a_jerk_magnitude_var , imu_b_jerk_magnitude_median , GYR_A_0_jerk_var , IMU_A_1_rmssd , GYR_A_2_jerk_kurt , GYR_A_0_jerk_median , imu_b_magnitude_var , gyr_b_jerk_magnitude_rmssd , imu_b_magnitude_skew , hrv_sdnn , gsr_SHF_ae_2 , IMU_A_0_jerk_mean , GYR_A_0_kurt , gsr_SHS_ae_0 , IMU_B_1_jerk_var , GYR_A_2_kurt , gsr_SHA_var , IMU_A_0_jerk_median , IMU_A_1_median , gsr_SHF_ae_0 , GYR_B_0_jerk_skew , IMU_A_0_jerk_skew , imu_a_magnitude_iqr , GYR_A_1_jerk_rmssd , IMU_A_2_jerk_var , IMU_B_2_kurt , hrv_median_nni , GYR_A_0_jerk_iqr , hrv_rmssd , GYR_A_0_mean , gsr_SHS_ae_1 , GYR_A_2_jerk_mean , GYR_B_0_jerk_rmssd , hrv_nni_20 , GYR_B_2_iqr , GYR_A_1_skew , gyr_b_jerk_magnitude_kurt , IMU_A_2_jerk_mean , discomfort_bins, nirvana_bins
```



2. train_new_model



The last column in the file is the target column
    
All entries should be numbers
    




