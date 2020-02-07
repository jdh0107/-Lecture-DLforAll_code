#!/usr/bin/env python
# coding: utf-8

# Session3:Linear Regression의 cost 최소화 구현

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
X=[1,2,3]
Y=[1,2,3]

W=tf.placeholder(tf.float32)
#H(x)=Wx 의 가설
#W가 placeholder라는 것은 우리가 W값을 임의로 바꿀 수 있다는 뜻
hypothesis=X*W

#cost func
cost=tf.reduce_mean(tf.square(hypothesis-Y))
#Session 안에 있는 그래프 시작
sess=tf.Session()
#Session 안의 전역 변수들 모두 초기화
sess.run(tf.global_variables_initializer())

#W,cost값을 저장한 리스트 생성
W_val=[]
cost_val=[]
for i in range(-30,50):
    feed_W=i*0.1
    curr_cost,curr_W=sess.run([cost,W],feed_dict={W:feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)
    
plt.plot(W_val,cost_val)
implt.show()


# In[5]:


import tensorflow as tf
x_data=[1,2,3]
y_data=[1,2,3]

W=tf.Variable(tf.random_normal([1]),name='weight')
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

hypothesis=X*W

cost=tf.reduce_mean(tf.square(hypothesis-Y))

#미분을 이용해서 최소화하는 과정(=밑의 'Minimize:Gradient Descent Magic')
learning_rate=0.1
gradient=tf.reduce_mean((W*X-Y)*X)
descent=W-learning_rate*gradient
update=W.assign(descent)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update,feed_dict={X:x_data,Y:y_data})
    print(step,sess.run(cost,feed_dict={X:x_data,Y:y_data}),sess.run(W))


# In[6]:


#Minimize:Gradient Descent Magic
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
train=optimizer.minimize(cost)

