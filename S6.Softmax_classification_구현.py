#!/usr/bin/env python
# coding: utf-8

# Session6. Softmax Classification 구현

# In[21]:


import tensorflow as tf

#[x1,x2,x3,x4]의 결과로 [0,1,2]중에 하나가 나옴
x_data=[[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
#one-hot식으로 표현하기. ex)[0,0,1]->세자리 중에 세번째 것
y_data=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

X=tf.placeholder("float32",[None,4])
Y=tf.placeholder("float32",[None,3])
#class의 개수=y의 label의 개수
nb_classes=3

#W : [#입력값, #출력값] // b: [#출력값]
W=tf.Variable(tf.random_normal([4,nb_classes]),name='weight')
b=tf.Variable(tf.random_normal([nb_classes]),name='bias')

#nn.softmax : S(y_i)의 함수 라이브러리
hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)

#cross entropy cost/Loss
#axis=1 : 행 기준으로 계산(가로로 계산한다는 뜻)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#학습시키기
with tf.Session() as sess:
    #초기화
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
        if step%200==0:
            print(step,sess.run(cost,feed_dict={X:x_data,Y:y_data}))


# 위의 모델로 Y값 예측하기

# In[19]:


sess=tf.Session()
sess.run(tf.global_variables_initializer())
all=sess.run(hypothesis,feed_dict={X:[[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
#arg_max : softmax의 결과로 가장 큰 요소를 보여줌
print(all,sess.run(tf.arg_max(all,1)))
#=>확률값으로 출력됨. 전체합은 1


# In[25]:


import numpy as np
np.max([[1,2,3],[4,5,6]],axis=0)

