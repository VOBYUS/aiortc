import HMSLSTM_Main as Main
import tensorflow as tf
import numpy as np
import pdb

def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

def regression_per_blink(labels, predictions, starts_list,
                         idx):  # starts_list includes the index of the start index in each video in the main input file
    #BSRE
    labels_pool = np.array([0, 5, 10])
    np.clip(predictions, 0, 10, out=predictions)
    LOSS = 0
    for i, start in enumerate(starts_list):
        if (i + 1) == len(starts_list):
            predicts = predictions[start:]
            Y = labels[start:]
        else:
            predicts = predictions[start:starts_list[i + 1]]
            Y = labels[start:starts_list[i + 1]]
        if Y[0, 0]==0:
            L=3.3
        if Y[0, 0] == 10:
            L = 6.6

        predicted_index = (predicts // 3.34).astype(np.int8)
        predicted_labels = labels_pool[predicted_index]
        if Y[0,0]==5:
            loss=np.sum((predicts[np.logical_and(predicted_labels != Y[0, 0],predicted_labels<5)] - 3.3) ** 2)
            loss=loss+np.sum((predicts[np.logical_and(predicted_labels != Y[0, 0],predicted_labels >= 5)] - 6.6) ** 2)
        else:
            loss = np.sum((predicts[predicted_labels != Y[0, 0]] - L) ** 2)
        LOSS = LOSS + loss

    if idx % 15 == 0 or idx==79:
        print('Per Blink Sequence Regression Error is :%f ' % (LOSS / len(labels)))

    return LOSS / len(labels)

def calc_first_threshold_acc(labels_init, m_predicts_init, threshold):  #Y_size=[Batch_size,1]
    labels = np.clip(labels_init,0,5)
    m_predicts = np.clip(m_predicts_init,0,5)
    t_predicts = np.array([0 if x[0] <= threshold else 5 for x in m_predicts])
    # print(m.shape)
    # prin_predictst(t_predicts.shape)
    # for index in range(0,len(t_predicts)):
    #     print("ACTUAL: " + str(labels[index][0]) + " | ML OUTPUT: " + str(m_predicts[index][0]) + " | PREDICTED: " + str(t_predicts[index]))
    #count
    true_positive = np.sum(np.logical_and([1 if x == 5 else 0 for x in t_predicts], [1 if x[0]==5 else 0 for x in labels]))
    false_positive= np.sum(np.logical_and([1 if x == 5 else 0 for x in t_predicts], [1 if x[0]==0 else 0 for x in labels]))
    true_negative= np.sum(np.logical_and([1 if x == 0 else 0 for x in t_predicts],[1 if x[0]==0 else 0 for x in labels]))
    false_negative= np.sum(np.logical_and([1 if x == 0 else 0 for x in t_predicts],[1 if x[0]==5 else 0 for x in labels]))
    n = len(labels)
    pdb.set_trace()
    return [true_positive/n, false_positive/n, true_negative/n, false_negative/n]

def calc_second_threshold_acc(Y_init, predicts_init, threshold):
    labels = np.clip(Y_init,5,10)
    m_predicts = np.clip(predicts_init,5,10)
    t_predicts = np.array([5 if x[0] <= threshold else 10 for x in m_predicts])
    #count
    true_positive = np.sum(np.logical_and([1 if x == 10 else 0 for x in t_predicts], [1 if x[0]==10 else 0 for x in labels]))
    false_positive= np.sum(np.logical_and([1 if x == 10 else 0 for x in t_predicts], [1 if x[0]==5 else 0 for x in labels]))
    true_negative= np.sum(np.logical_and([1 if x == 5 else 0 for x in t_predicts],[1 if x[0]==5 else 0 for x in labels]))
    false_negative= np.sum(np.logical_and([1 if x == 5 else 0 for x in t_predicts],[1 if x[0]==10 else 0 for x in labels]))
    n = len(m_predicts)
    return [true_positive/n, false_positive/n, true_negative/n, false_negative/n]

def Predict(total_input,total_labels,TestB,TestL,output_size,feature_size,batch_size,num_epochs,Pre_fc1_size,Post_fc1_size_per_layer,embb_size,
          embb_size2,Post_fc2_size,hstate_size,num_layers,step_size,drop_out_p,lr,th,start_i,load,fold_num, first_threshold, second_threshold):  #total_input is the shuffled input with size=[Total data points, T,F]

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('my_model%d.meta'%fold_num)
        print('loading variables...')
        saver.restore(sess, './my_model%d'%fold_num)
        ####plotting_setup\
        y = np.zeros([num_epochs])
        yy = np.zeros([num_epochs])
        y_test = np.zeros([num_epochs])
        yy_test = np.zeros([num_epochs])
        y_v = np.zeros([num_epochs])
        x=np.linspace(1, num_epochs, num_epochs, endpoint=True)
        loss_per_epoch = 0
        sum=0
        ## loss_values_Test, predicts_Test,mid_vT = sess.run([loss, output,end_points],feed_dict={input_net: TestB, labels: TestL,keep_p:1.0,training:False})
        predicts_Test = []
        loss_values_Test = []
        for i, testB in enumerate(TestB):
            predict = sess.run(['last_fc/mul:0'],feed_dict={ 'bacth_in:0': [TestB[i]], 'labels_net:0': [TestL[i]],'Placeholder:0':1.0,'phase_train:0':False})
            predicts_Test.append(predict[0][0])
            loss_values_Test.append(99)
        print("first threshold: " + str(first_threshold) + " second threshold: " + str(second_threshold))
        first_confmat = calc_first_threshold_acc(TestL, np.array(predicts_Test), first_threshold)
        print("first threshold config (tp, fp, tn, fn) " + str(first_confmat))
        second_confmat = calc_second_threshold_acc(TestL, np.array(predicts_Test), second_threshold)
        print("second threshold config (tp, fp, tn, fn) " + str(second_confmat))
        regression_per_blink(TestL,np.array(predicts_Test),start_i,0) #BSRE
        print("----------------------------------------------")
    return x,y,yy,y_test,yy_test,y_v, first_confmat, second_confmat

def test_accuracy(first_threshold, second_threshold):
    load=True  # We are loading the model that was trained already
    for i in range(7): #Cross validation but recommended to run each fold a few times to see the best perfomrance as you may
        print('######################')
        print(i)
        print('######################')
        # get caught up in a local minimum
        Blinks = np.load('Blinks_30_%d.npy'%(i+1))
        Labels = np.load('Labels_30_%d.npy'%(i+1))
        BlinksTest = np.load('BlinksTest_30_%d.npy'%(i+1))
        LabelsTest = np.load('./LabelsTest_30_%d.npy'%(i+1))
        start_indices = np.load('./StartIndices_30_%d.npy'%(i+1))
        #####################Normalizing the input#############Second phase
        BlinksTest[:,:,0]=(BlinksTest[:,:,0]-np.mean(Blinks[:,:,0]))/np.std(Blinks[:,:,0])
        Blinks[:,:,0]=(Blinks[:,:,0]-np.mean(Blinks[:,:,0]))/np.std(Blinks[:,:,0])
        BlinksTest[:,:,1]=(BlinksTest[:,:,1]-np.mean(Blinks[:,:,1]))/np.std(Blinks[:,:,1])
        Blinks[:,:,1]=(Blinks[:,:,1]-np.mean(Blinks[:,:,1]))/np.std(Blinks[:,:,1])
        BlinksTest[:,:,2]=(BlinksTest[:,:,2]-np.mean(Blinks[:,:,2]))/np.std(Blinks[:,:,2])
        Blinks[:,:,2]=(Blinks[:,:,2]-np.mean(Blinks[:,:,2]))/np.std(Blinks[:,:,2])
        BlinksTest[:,:,3]=(BlinksTest[:,:,3]-np.mean(Blinks[:,:,3]))/np.std(Blinks[:,:,3])
        Blinks[:,:,3]=(Blinks[:,:,3]-np.mean(Blinks[:,:,3]))/np.std(Blinks[:,:,3])
        x,loss,accuracy,loss_Test,accuracy_Test,acc_per_Vid, first_confmat, second_confmat=Predict(total_input=Blinks,total_labels=Labels,TestB=BlinksTest,TestL=LabelsTest,
                        output_size=1,feature_size=4,batch_size=64,num_epochs=80,Pre_fc1_size=32,Post_fc1_size_per_layer=16,
                        embb_size=16,embb_size2=16,Post_fc2_size=8,hstate_size=[32,32,32,32],num_layers=4,step_size=30,drop_out_p=1.0,
                                                    lr=0.000053,th=1.253,start_i=start_indices,load=load,fold_num=i,
                                                    first_threshold = first_threshold, second_threshold = second_threshold)
        return first_confmat, second_confmat

        # if load==False:
        #     np.save(open('./x%d.npy' %ii, 'wb'),x)
        #     np.save(open('./loss%d.npy'%ii, 'wb'),loss) #for training
        #     np.save(open('./accuracy%d.npy' %ii, 'wb'),accuracy) #for training
        #     np.save(open('./loss%dTest.npy'%ii, 'wb'),loss_Test) #for test
        #     np.save(open('./accuracy%dTest.npy'%ii, 'wb'),accuracy_Test) #for test (BSA)
        #     np.save(open('./accuracy%dVTest.npy'%ii, 'wb'),acc_per_Vid) #for test    (VA)

first_thresholds = np.linspace(0, 10, num=5)
second_thresholds = np.linspace(0, 10, num=21)
first_threshold_cm = []
second_threshold_cm = []
for f_threshold in first_thresholds:
    first_threshold_cm.append(test_accuracy(f_threshold, 9.99)[0])
# for s_threshold in second_thresholds:
#     second_threshold.cm.append(test_accuracy(3.33, s_threshold)[1])

