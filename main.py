# -*- coding: utf-8 -*-
import os, sys

import packages.preprocess as ppp
import packages.conv as conv
import packages.rotation as rot
import packages.ml_alg as ml
# import packages.ml_alg as ml
import time
from natsort import natsorted
import pandas as pd
import csv

def main():
    # Invert color for the white dataset since the resize fill the background with black color
    # invertColors('/home/bioprinting/axel/final_data/Thumbnail')

    # Location of the directory where the data will be (ready to process)
    finalDataPath = '/media/axel/DATA/Axel/Documents/DATASET/final_data_2'

    #Location of the data before data augmentation and resizing (Original data)
    curDataPath = '/media/axel/DATA/Axel/Documents/DATASET/JPEG_Img_Seq'

    # # Store the size of the biggest image that will be used to resize all images without loosing information
    largerSize = ppp.getLargerSize(stackList,curDataPath)

    # For each stack of the original data, make it png if not (processDir),
    # Resize it according to the largerSize variable (resize)
    for stack in os.listdir(curDataPath):
        stackPath = os.path.join(curDataPath, stack)
        pngStack = ppp.processDir(stackPath)
        resizedStack = ppp.resize(pngStack, largerSize, finalDataPath)

    # Create a flipped stack of each of the stacks in finalDataPath (mirror image = data augmentation)
    ppp.flipDataset(finalDataPath)



    # SECOND PART


    #create lists to store the paths of vectors (data) and rotations (labels)
    vec_files = []
    rot_files = []

    print (os.listdir(finalDataPath))
    # Update the list of stacks of data according to the final directory and reorder those stacks
    # to make sure that each is process in a certain order
    stackList = natsorted(os.listdir(finalDataPath))
    print 'sorted list : ' + str(stackList)



    conv_time = time.time()
    # For each stack rotate all images and save them in directories of pair of images
    # Save the difference of rotations between pairs of images and save the path of this file in the rotFil
    # Apply convolution to each image and save the output vector corresponding to a pair of slices, and save the path in vecFile
    for stack in stackList :
        # Create the full path for each stack
        stackPath = os.path.join(finalDataPath, stack)
        rotFile = rot.create_pairs(stackPath)
        print 'RotFile length : ' + str(len(rotFile))
        rot_files.append(rotFile)

        vecFile = conv.fc(stackPath)
        print 'VecFile length : ' + str(len(vecFile))
        vec_files.append(vecFile)
        print ('vecFile = ' + vecFile)

    print("--- %s seconds --- TOTAL CONV TIME" % (time.time() - conv_time))


    # List of all vectors representing the output of the convoltion to avoid to have to run the convoltion everytime
    vec_files =  [
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17561/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17565/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17569/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17573/vecImg.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20115/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17577/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17563/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17567/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17571/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17575/vecImg.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20113/vecImg.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20117/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17565_flipped/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17561_flipped/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17569_flipped/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17573_flipped/vecImg.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17577_flipped/vecImg.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20115_flipped/vecImg.csv'
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17563_flipped/vecImg.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17567_flipped/vecImg.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17571_flipped/vecImg.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17575_flipped/vecImg.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20113_flipped/vecImg.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20117_flipped/vecImg.csv',
        ]
    #     # '/media/axel/DATA/Axel/Documents/DATASET/final_data/mouse_kidney_png_flipped/vecImg.csv',
    #     # '/media/axel/DATA/Axel/Documents/DATASET/final_data/kidney_2_png_flipped/vecImg.csv',
    #     # '/media/axel/DATA/Axel/Documents/DATASET/final_data/atlas_0_png_flipped/vecImg.csv',
    #     # '/media/axel/DATA/Axel/Documents/DATASET/final_data/atlas_1_png_flipped/vecImg.csv'
    # ]
    #
    # Same as vectors but for rotations of the slices
    rot_files = [
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17561/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17565/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17569/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17573/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17577/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17563/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20115/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17567/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17571/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17575/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20113/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20117/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17561_flipped/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17565_flipped/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17573_flipped/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17569_flipped/rotations.csv',
        '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17577_flipped/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20115_flipped/rotations.csv'
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T1_vp.tif_17563_flipped/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17567_flipped/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2star_vp.tif_17571_flipped/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/canon_T2W_vp.tif_17575_flipped/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20113_flipped/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data_2/CNissl.tif_20117_flipped/rotations.csv',
        ]
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data/mouse_kidney_png_flipped/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data/kidney_2_png_flipped/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data/atlas_0_png_flipped/rotations.csv',
        # '/media/axel/DATA/Axel/Documents/DATASET/final_data/atlas_1_png_flipped/rotations.csv'
    ]
    #
    all_score = []
    nb = 1
    for  i in range(nb):
        print i
        # Arrange the data to make it ready to be the input of the machine learner
        training_data, training_labels, test_data, test_labels = ml.arrange_data(vec_files, rot_files, i)

        # Run the machine learner on the data, learn on training and predict on test data
        res = ml.learn(training_data, training_labels, test_data, test_labels, i)
        all_score.append(res)


    with open('/media/axel/DATA/Axel/Documents/DATASET/final_data_2/all_score_bis_61.csv', "wb") as f:
        writer = csv.writer(f, delimiter=';', lineterminator='\n')
        writer.writerows(all_score)


    out = open('/media/axel/DATA/Axel/Documents/DATASET/final_data_2/all_score_1.csv', "w")
    for row in all_score:
        for column in row:
            for val in column :
                print val
                f.write('%d;' % val)
            f.write('\n')


    # rfr_score_df = pd.DataFrame(all_score)
    # rfr_score_df.to_csv('/home/bioprinting/axel/data_res/all_score_bis_61.csv', index=False, header=False)
    return 0



if __name__ == '__main__':
    main()
