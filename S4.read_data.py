#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Section4. TensorFlow로 파일에서 데이터 읽어오기(numpy 이용)


# In[11]:


import tensorflow as tf
import numpy as np

xy=np.loadtxt('C:\\Users\\DoHyun\\Desktop\\data.csv',delimiter=',',dtype=np.float32)
x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

#data_shape:(#행,#열)
print(x_data.shape,x_data,len(x_data))
print(y_data.shape,y_data)

#X는 data가 모여있는 행렬. 변수가 3개(x1,x2,x3)이므로 [none,3]
#Y는 X로부터 마지막에 나오는 결과값. 변수가 1개
X=tf.placeholder(tf.float32,shape=[None,3])
Y=tf.placeholder(tf.float32,shape=[None,1])

#W는 X들의 계수. 변수가 3개(w1,w2,w3)이고 마지막에 구할 Y값은 1개이므로 [3,1]
#b는 bias. 값이 1개
W=tf.Variable(tf.random_normal([3,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

#Hypothesis
hypothesis=tf.matmul(X,W)+b #tf.matmul(A,B) : A와 B를 내적

#여기서부터는 동일#

#cost/Loss function
cost=tf.reduce_mean(tf.square(hypothesis - Y))
#최소화
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val,hy_val,_ = sess.run([cost,hypothesis,train], feed_dict={X:x_data,Y:y_data})
    if step %10 == 0:
        print(step, "Cost: ",cost_val, "\nPrediction:\n",hy_val)


# In[14]:


#Ask my score
print("Your score will be ",sess.run(hypothesis,feed_dict={X:[[100,70,101]]}))
print("Other scores will be ",sess.run(hypothesis,feed_dict={X:[[60,70,110],[90,100,80]]}))


# Queue runners 구현
# (numpy로 하면 메모리가 부족할 때)

# In[23]:


import tensorflow as tf
# 1. filename을 리스트로 작성
filename_queue = tf.train.string_input_producer(['C:\\Users\\DoHyun\\Desktop\\data.csv'], shuffle=False,name='filename_queue')

# 2. 텍스트파일을 읽을 Reader를 정하기
reader = tf.TextLineReader()
key,value=reader.read(filename_queue)

# 3. 읽어온 value값을 어떻게 이해할 것인가를 decode_csv로 표현.
#여기서는 csv 속 수들을 ',(콤마)'로 분리하고 실수 데이터타입으로 가져오고 싶어서 [[0.],[0.],[0.],[0.]] 로 표현.
record_defaults=[[0.],[0.],[0.],[0.]]
xy=tf.decode_csv(value,record_defaults=record_defaults)

#collect batches of csv in
#노드 이름 정해주기
#데이터 읽기.  tf.train.batch([x데이터,y데이터],한번에 가져올 개수)
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1],xy[-1:]],batch_size=10)

#placeholders for a tensor that will be always fed
X=tf.placeholder(tf.float32,shape=[None,3])
Y=tf.placeholder(tf.float32,shape=[None,1])
W=tf.Variable(tf.random_normal([3,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

#Hypothesis
hypothesis=tf.matmul(X,W)+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

#Launch the graph in a session
sess=tf.Session()
#전역변수 초기화
sess.run(tf.global_variables_initializer())

#Start populating the filenname queue
#여기서부터 일반적으로 쓰는 방법
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)

for step in range(2001):
    x_batch,y_batch=sess.run([train_x_batch,train_y_batch])
    cost_val,hy_val,_=sess.run([cost,hypothesis,train],feed_dict={X:x_batch,Y:y_batch})
    if step%10==0:
        print(step,"Cost: ",cost_val, "\nPrediction:\n:",hy_val)
        
#끝날때
coord.request_stop()
coord.join(threads)

