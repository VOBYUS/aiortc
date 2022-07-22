import HMSLSTM_Main as Main
import tensorflow as tf
import numpy as np

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


def save_variables(sess,path,f):
        saver = tf.train.Saver()
        print('saving variables...\n')
        saver.save(sess,path+'my_model%d'%f)

def calc_accuracy_per_batch(Y, predicts):  #Y_size=[Batch_size,1]
    labels_pool = np.array([0, 5, 10])
    np.clip(predicts,0,10,out=predicts)
    predicted_index = (predicts // 3.34).astype(np.int8)
    predicted_labels = labels_pool[predicted_index]
    is_correct = np.equal(predicted_labels, Y.astype(np.int8))
    accuracy = np.sum(is_correct)/len(is_correct)

    return accuracy

def Predict(total_input,total_labels,TestB,TestL,output_size,feature_size,batch_size,num_epochs,Pre_fc1_size,Post_fc1_size_per_layer,embb_size,
          embb_size2,Post_fc2_size,hstate_size,num_layers,step_size,drop_out_p,lr,th,start_i,load,fold_num):  #total_input is the shuffled input with size=[Total data points, T,F]

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
        accuracy_Test = calc_accuracy_per_batch(TestL, np.array(predicts_Test)) #BSA
        print("BSA: " + str(accuracy_Test))
        regression_per_blink(TestL,np.array(predicts_Test),start_i,0) #BSRE
        print("----------------------------------------------")

    return x,y,yy,y_test,yy_test,y_v


################################################
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
    x,loss,accuracy,loss_Test,accuracy_Test,acc_per_Vid=Predict(total_input=Blinks,total_labels=Labels,TestB=BlinksTest,TestL=LabelsTest,
                    output_size=1,feature_size=4,batch_size=64,num_epochs=80,Pre_fc1_size=32,Post_fc1_size_per_layer=16,
                    embb_size=16,embb_size2=16,Post_fc2_size=8,hstate_size=[32,32,32,32],num_layers=4,step_size=30,drop_out_p=1.0,
                                                  lr=0.000053,th=1.253,start_i=start_indices,load=load,fold_num=i)


    # if load==False:
    #     np.save(open('./x%d.npy' %ii, 'wb'),x)
    #     np.save(open('./loss%d.npy'%ii, 'wb'),loss) #for training
    #     np.save(open('./accuracy%d.npy' %ii, 'wb'),accuracy) #for training
    #     np.save(open('./loss%dTest.npy'%ii, 'wb'),loss_Test) #for test
    #     np.save(open('./accuracy%dTest.npy'%ii, 'wb'),accuracy_Test) #for test (BSA)
    #     np.save(open('./accuracy%dVTest.npy'%ii, 'wb'),acc_per_Vid) #for test    (VA)


