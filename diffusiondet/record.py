# device: Intel Core i7-12700k, NVIDIA RTX4090

# CIFAR-10
# KMeans：              NMI: 0.759, ACC: 0.862, ARI: 0.726, time_cost: 11.341 second
# Agglomerative:       NMI: 0.722, ACC: 0.742, ARI: 0.632, time_cost: 775.431 second
# AffinityPropagation: NMI: 0.463, ACC: 0.0407, ARI: 0.031, time_cost: 62.304 second ?
# CC:                  NMI: 0.705,  ACC: 0.790,  ARI: 0.637  time_cost: -
# KNN（6）:            NMI: 0.909, ACC: 0.962, ARI: 0.919, time_cost: 353.718 second

# STL-10
# KMeans:              NMI: 0.945, ACC: 0.975, ARI: 0.944, time_cost: 0.999 second        done
# Agglomerative:       NMI: 0.930, ACC: 0.964, ARI: 0.924, time_cost: 11.397 second
# AffinityPropagation  NMI: 0.532, ACC: 0.054, ARI: 0.042, time_cost: 188.215 second   done
# CC:                  NMI: 0.764, ACC: 0.850, ARI: 0.726, time_cost: - second
# KNN(19):              NMI: 0.976, ACC: 0.991, ARI: 0.981, time_cost: 45.558 second      done

# ImageNet-10
# KMeans:              NMI: 0.972, ACC: 0.990, ARI: 0.978, time_cost: 1.470 second
# Agglomerative:       NMI: 0.958, ACC: 0.981, ARI: 0.958, time_cost: 32.619 second
# AffinityPropagation  NMI: 0.580, ACC: 0.090, ARI: 0.082, time_cost: 118.944 second
# CC:                  NMI: 0.859, ACC: 0.893, ARI: 0.822, time_cost: - second
# KNN(7):              NMI: 0.983, ACC: 0.994, ARI: 0.987, time_cost: 7.955 second(including time finding the best parameter)

# CIFAR-100
# KMeans:              NMI: 0.623, ACC: 0.482, ARI: 0.326, time_cost: 11.107 second done
# Agglomerative:       NMI: 0.622, ACC: 0.472, ARI: 0.308, time_cost: 18.489 second done
# AffinityPropagation  NMI: 0.637, ACC: 0.179, ARI: 0.131, time_cost: 97.377 second ?
# CC:                  NMI: 0.431, ACC: 0.429, ARI: 0.266, time_cost: - second
# KNN(11):             NMI: 0.801, ACC: 0.818, ARI: 0.677, time_cost: 17.436 second done

############################################## DINO compare ##################################################
# CIFAR-10:
# base8(6):    NMI: 0.909, ACC: 0.962, ARI: 0.919 done
# base16(6):   NMI: 0.897, ACC: 0.956, ARI: 0.905, time_cost: 17.623 second done
# small8(12)： NMI: 0.855,  ACC: 0.934,  ARI: 0.860  time_cost: 5.513
# small16(8):  NMI: 0.832, ACC: 0.922, ARI: 0.837, time_cost: 10.129 second

# CIFAR-100:
# base8(11):    NMI: 0.801, ACC: 0.818, ARI: 0.677, time_cost: 17.436 second done
# base16(9):   NMI: 0.800, ACC: 0.818, ARI: 0.678, time_cost: 17.623 second
# small8(14)： NMI: 0.758,  ACC: 0.775,  ARI: 0.612  time_cost: 5.513         done
# small16(18):  NMI: 0.715, ACC: 0.734, ARI: 0.552, time_cost: 10.129 second done

# Class: airplane
# Accuracy: 0.8966
# Precision: 0.8747609942638623
# Recall: 0.915
#
# Class: automobile
# Accuracy: 0.8966
# Precision: 0.9319066147859922
# Recall: 0.958
#
# Class: bird
# Accuracy: 0.8966
# Precision: 0.914409534127844
# Recall: 0.844
#
# Class: cat
# Accuracy: 0.8966
# Precision: 0.7749546279491834
# Recall: 0.854
#
# Class: deer
# Accuracy: 0.8966
# Precision: 0.8870967741935484
# Recall: 0.88
#
# Class: dog
# Accuracy: 0.8966
# Precision: 0.8565615462868769
# Recall: 0.842
#
# Class: frog
# Accuracy: 0.8966
# Precision: 0.9425403225806451
# Recall: 0.935
#
# Class: horse
# Accuracy: 0.8966
# Precision: 0.9176225234619395
# Recall: 0.88
#
# Class: ship
# Accuracy: 0.8966
# Precision: 0.9407035175879397
# Recall: 0.936
#
# Class: truck
# Accuracy: 0.8966
# Precision: 0.9408163265306122
# Recall: 0.922