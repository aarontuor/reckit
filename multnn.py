import input
import tensorflow as tf
import numpy as np

data = input.read_data_sets('mat/')
Litem = 15
Luser = 5
L = 2

xuser = tf.placeholder("float",[None, None],name='user')
xitem = tf.placeholder("float",[None, None],name='item')

witem = tf.Variable(tf.truncated_normal([data.train.item_tfidf_vectors.shape[1],Litem], stddev=.1),name='Witem')
bitem = tf.Variable(tf.constant(.1,shape=[Litem]),name='Bitem')

wuser = tf.Variable(tf.truncated_normal([data.train.useronehots.shape[1],Luser], stddev=.1),name='Wuser')
buser = tf.Variable(tf.constant(.1,shape=[Luser]),name='Buser')

hitem = tf.tanh(tf.matmul(xitem,witem)+bitem) 
huser = tf.tanh(tf.matmul(xuser,wuser)+buser)

vitem = tf.Variable(tf.truncated_normal([Litem,L], stddev=.1),name='vitem')
q = tf.Variable(tf.constant(.1,shape=[L]))

vuser = tf.Variable(tf.truncated_normal([Luser,L], stddev=.1),name='vuser')

hfinal = tf.tanh(tf.matmul(hitem,vitem)+tf.matmul(huser,vuser)+q)

u = tf.Variable(tf.truncated_normal([L,1], stddev=.1),name='U')
bfinal = tf.Variable(tf.constant(.1))

y = tf.matmul(hfinal,u)+bfinal

y_ = tf.placeholder("float", [None, None], name='Y_')

objective = tf.reduce_sum(tf.square(y_-y))
rmse = tf.sqrt(tf.div(tf.reduce_sum(tf.square(y_-y)), tf.to_float(tf.size(y_))))
tf.scalar_summary('Loss',objective)

train_step = tf.train.GradientDescentOptimizer(.00015).minimize(objective)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs',session.graph.as_graph_def())

for i in range(1000):
    batch_user, batch_item, batch_y = data.train.next_batch(1000)
    session.run(train_step, feed_dict={xuser: batch_user.todense(), y_: batch_y, xitem: batch_item.todense()})
    print "Dev rmse: ", i, " ", session.run(rmse,feed_dict={xitem: data.dev.item_tfidf_vectors.todense(), y_: data.dev.ratings, xuser: data.dev.useronehots.todense()})
   # print "Objective : ", i, " ", session.run(objective,feed_dict={xitem: data.train.item_tfidf_vectors.todense(), y_:data.train.ratings, xuser: data.train.useronehots.todense()})




