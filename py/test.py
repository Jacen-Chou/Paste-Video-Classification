# print("\t\tPredicted", end='')
# print("75\t77\t79\t81\n")
#
# for i in range(0, 4):
#     print(i)

# -------------------------------------------

# conc = {
#     '0': '75  ',
#     '1': '77  ',
#     '2': '79  ',
#     '3': '81  '
# }
#
# print("\t\t   Predicted\n")
# print("\t\t   75\t77\t79\t81\n")
# for i in range(0, num_classes):
#     print("Actual ", end='')
#     print(conc[str(i)], end='')
#     for j in range(0, num_classes):
#         print(str(matrix[i][j]) + '\t', end='')
#     print('\n')

# -------------------------------------------

acc_val_vgg13 = [0.25, 0.5, 0.6214285714285714, 0.7357142857142858, 0.7380952380952381, 0.6595238095238095, 0.7452380952380953, 0.7333333333333333, 0.7523809523809524, 0.75, 0.8047619047619048, 0.8714285714285714, 0.680952380952381, 0.8071428571428572, 0.9071428571428571, 0.8404761904761905, 0.9523809523809523, 0.9619047619047619, 0.9857142857142858, 0.9595238095238096, 0.9809523809523809, 0.9547619047619048, 0.9761904761904762, 0.9642857142857143, 0.9928571428571429, 0.969047619047619, 0.9, 0.9952380952380953, 0.9785714285714285, 0.9523809523809523, 1.0, 0.9976190476190476, 0.9952380952380953, 0.9976190476190476, 0.9976190476190476, 0.9166666666666666, 0.9857142857142858, 0.9714285714285714, 0.9833333333333333, 0.9761904761904762]
acc_val_vgg16 = [0.416667, 0.516667, 0.361905, 0.735714, 0.488095, 0.759524, 0.745238, 0.688095, 0.690476, 0.752381, 0.745238, 0.728571, 0.75, 0.730952, 0.742857, 0.745238, 0.75, 0.804762, 0.773810, 0.747619, 0.859524, 0.916667, 0.583333, 0.9, 0.904762, 0.830952, 0.954762, 0.947619, 0.983333, 0.983333, 0.988095, 0.954762, 0.840476, 0.983333, 0.983333, 0.985714, 0.966667, 0.969048, 0.980952, 0.992857]
acc_val_vgg19 = [0.25, 0.5071428571428571, 0.2714285714285714, 0.4023809523809524, 0.5428571428571428, 0.6333333333333333, 0.4666666666666667, 0.6571428571428571, 0.55, 0.5595238095238095, 0.75, 0.5119047619047619, 0.7238095238095238, 0.6904761904761905, 0.7, 0.5880952380952381, 0.7476190476190476, 0.75, 0.7214285714285714, 0.7452380952380953, 0.7404761904761905, 0.7404761904761905, 0.7523809523809524, 0.7380952380952381, 0.7428571428571429, 0.7571428571428571, 0.75, 0.7619047619047619, 0.7952380952380952, 0.7166666666666667, 0.7666666666666667, 0.7238095238095238, 0.7547619047619047, 0.7880952380952381, 0.7547619047619047, 0.7428571428571429, 0.7428571428571429, 0.7738095238095238, 0.7452380952380953, 0.7714285714285715]
acc_val_resnet50 = [0.4595238095238095, 0.6761904761904762, 0.7428571428571429, 0.7476190476190476, 0.7476190476190476, 0.7452380952380953, 0.7214285714285714, 0.7523809523809524, 0.9, 0.8, 0.9166666666666666, 0.8976190476190476, 0.8595238095238096, 0.819047619047619, 0.9952380952380953, 0.9642857142857143, 0.9761904761904762, 1.0, 0.85, 1.0, 0.9928571428571429, 1.0, 0.9642857142857143, 0.969047619047619, 0.9928571428571429, 1.0, 1.0, 0.9952380952380953, 1.0, 0.9952380952380953, 0.9976190476190476, 1.0, 0.9952380952380953, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
acc_val_resnet101 = [0.7142857142857143, 0.7119047619047619, 0.75, 0.7071428571428572, 0.8023809523809524, 0.7404761904761905, 0.8571428571428571, 0.7428571428571429, 0.8833333333333333, 0.8452380952380952, 0.888095238095238, 0.969047619047619, 0.9595238095238096, 0.9666666666666667, 0.9261904761904762, 0.9523809523809523, 0.9738095238095238, 0.9047619047619048, 0.9738095238095238, 0.9761904761904762, 0.969047619047619, 0.9547619047619048, 0.930952380952381, 0.9619047619047619, 0.9666666666666667, 0.9714285714285714, 0.9738095238095238, 0.9380952380952381, 0.9904761904761905, 0.9595238095238096, 0.969047619047619, 0.95, 0.9642857142857143, 0.9761904761904762, 0.9261904761904762, 0.9619047619047619, 0.9738095238095238, 0.9976190476190476, 0.9833333333333333, 0.9619047619047619]
acc_val_resnet152 = [0.6785714285714286, 0.7023809523809523, 0.7547619047619047, 0.7476190476190476, 0.7285714285714285, 0.7047619047619048, 0.75, 0.7476190476190476, 0.6833333333333333, 0.7666666666666667, 0.7666666666666667, 0.8452380952380952, 0.7880952380952381, 0.8785714285714286, 0.9357142857142857, 0.8738095238095238, 0.9333333333333333, 0.969047619047619, 0.9, 0.9428571428571428, 0.9166666666666666, 0.9261904761904762, 0.8333333333333334, 0.95, 0.969047619047619, 0.9619047619047619, 0.9642857142857143, 0.9333333333333333, 0.9714285714285714, 1.0, 0.9761904761904762, 1.0, 0.9571428571428572, 0.9285714285714286, 0.9761904761904762, 0.9547619047619048, 0.9642857142857143, 0.9809523809523809, 0.9833333333333333, 0.9904761904761905]
acc_val_densenet121 = [0.9666666666666667, 0.9880952380952381, 0.9952380952380953, 0.9952380952380953, 0.9880952380952381, 0.9952380952380953, 1.0, 1.0, 1.0, 0.9476190476190476, 1.0, 1.0, 0.9761904761904762, 1.0, 0.9880952380952381, 1.0, 0.9666666666666667, 1.0, 1.0, 1.0, 0.9976190476190476, 1.0, 1.0, 0.9857142857142858, 1.0, 0.9976190476190476, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9928571428571429, 1.0, 0.9976190476190476, 1.0]