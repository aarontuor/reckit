import input
import tensorflow as tf
import numpy as np
import argparse
import sys

args = sys.argv

'''if len(args) < 3:
	print('Usage: <Lwitem> <Lwuser> <Lvitem> <Lvuser> <L> <stepsize> <batchsize> <dataset folder>')
	sys.exit()
 
Lwitem = args[1]
Lwuser = args[2]
Lvitem = args[3]
Lvuser = args[4]
L = args[5]
stepsize = args[6]
batchsize = args[7]
dataset_folder = args[8]'''

data = input.read_data_sets('mat/')
Litem = 15
Luser = 15
L = 1

xuser = tf.placeholder("float",[None, None],name='user')
xitem = tf.placeholder("float",[None, None],name='item')

witem = tf.Variable(tf.truncated_normal([data.train.item_tfidf_vectors.shape[1],Litem], stddev=.1),name='Witem')
b1item = tf.Variable(tf.constant(.1,shape=[Litem]),name='B1item')

wuser = tf.Variable(tf.truncated_normal([data.train.useronehots.shape[1],Luser], stddev=.1),name='Wuser')
b1user = tf.Variable(tf.constant(.1,shape=[Luser]),name='B1user')

hitem = tf.sigmoid(tf.matmul(xitem,witem)+b1item) #numexamples X numwords mult numwords X Litem = numexamples X Litem
huser = tf.sigmoid(tf.matmul(xuser,wuser)+b1user) #numexamples X numusers mult numusers X Luser = numexamples X Luser

vitem = tf.Variable(tf.truncated_normal([Litem,L], stddev=.1),name='vitem') # Litem X L
b2item = tf.Variable(tf.constant(.1,shape=[1, L]),name='B2item') # 1 X L

vuser = tf.Variable(tf.truncated_normal([Luser,L], stddev=.1),name='vuser') #Luser X L
b2user = tf.Variable(tf.constant(.1,shape=[1, L]),name='B2user') #1 X L

hfinalitem = tf.sigmoid(tf.matmul(hitem,vitem) + b2item)  #numexamples X L
hfinaluser = tf.sigmoid(tf.matmul(huser,vuser) + b2user) #numexamples X L

#bfinalitem = tf.Variable(tf.constant(.01,shape=[int(tf.shape(xitem))]),name='Bfinalitem')
#bfinaluser = tf.Variable(tf.constant(.01,shape=[int(tf.shape(xuser))]),name='Bfinaluser')

y = 5*tf.cos(tf.matmul(hfinaluser, tf.transpose(hfinalitem)))

y_ = tf.placeholder("float", [None, None], name='Y_')

objective = tf.reduce_sum(tf.square(y_-y))
rmse = tf.sqrt(tf.div(tf.reduce_sum(tf.square(y_-y)), tf.to_float(tf.size(y_))))
tf.scalar_summary('Loss',objective)

train_step = tf.train.GradientDescentOptimizer(.00001).minimize(objective)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs',session.graph.as_graph_def())
print(tf.rank(y))
print(tf.rank(y_))
for i in range(2):
    batch_user, batch_item, batch_y = data.train.next_batch(100)
    session.run(train_step, feed_dict={xuser: batch_user.todense(), y_: batch_y, xitem: batch_item.todense()})
    print "Dev rmse: ", i, " ", session.run(rmse,feed_dict={xitem: data.dev.item_tfidf_vectors.todense(), y_: data.dev.ratings, xuser: data.dev.useronehots.todense()})

   # print "Objective : ", i, " ", session.run(objective,feed_dict={xitem: data.train.item_tfidf_vectors.todense(), y_:data.train.ratings, xuser: data.train.useronehots.todense()})
print(tf.rank(y))
print(tf.rank(y_))



