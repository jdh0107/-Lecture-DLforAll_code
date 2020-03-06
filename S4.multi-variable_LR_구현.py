#!/usr/bin/env python
# coding: utf-8

# Section4.multi-variable linear regression 구현(matrix없이)

# In[1]:


import tensorflow as tf

x1_data=[73.,93.,89.,96.,73.]
x2_data=[80.,88.,91.,98.,66.]
x3_data=[75.,93.,90.,100.,70.]
y_data=[152.,185.,180.,196.,142.]

#placeholders for a tensor that will be always fed
x1=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)
x3=tf.placeholder(tf.float32)

Y=tf.placeholder(tf.float32)

w1=tf.Variable(tf.random_normal([1]),name='weight1')
w2=tf.Variable(tf.random_normal([1]),name='weight2')
w3=tf.Variable(tf.random_normal([1]),name='weight3')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=x1*w1+x2*w2+x3*w3+b

#cost/Loss function
cost=tf.reduce_mean(tf.square(hypothesis - Y))
#최소화
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val,hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,Y:y_data})
    if step %10 == 0:
        print(step, "Cost: ",cost_val, "\nPrediction:\n",hy_val)


# Matrix 사용

# In[11]:


x_data=[[73.,80.,75.],[93.,88.,93.],[89.,91.,90.],[96.,98.,100.],[73.,66.,70.]]
y_data=[[152.],[185.],[180.],[196.],[142.]]

#placeholders for a tensor that will be always fed.
X=tf.placeholder(tf.float32,shape=[None,3]) #[[]의 개수,[]안 변수의 수]
Y=tf.placeholder(tf.float32,shape=[None,1])

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


# In[ ]:




