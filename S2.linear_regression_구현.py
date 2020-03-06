#!/usr/bin/env python
# coding: utf-8

# 섹션2. linear Regression 구현

# In[2]:


import tensorflow as tf
x_train=[1,2,3]
y_train=[1,2,3]

W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=x_train*W+b


# In[3]:


#cost(loss)function
cost=tf.reduce_mean(tf.square(hypothesis-y_train)) #reduce_mean:평균내기


# In[4]:


#minimize
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)


# In[5]:


#그래프 만들기 시작하기
sess=tf.Session()
#initializes global variables in the graph
sess.run(tf.global_variables_initializer())


# In[6]:


#fit the line
for step in range(2001):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))


# In[12]:


#placeholder 로 구현

W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')
X=tf.placeholder(tf.float32,shape=[None])
Y=tf.placeholder(tf.float32,shape=[None])
#None: 1차원 array이고 아무개수의 값이 들어올수있다는 것

hypothesis=X*W+b
cost=tf.reduce_mean(tf.square(hypothesis-Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
#GradientDescentOptimizer = W:=W-(learning_rate * cost(W)의 미분값)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val,W_val, b_val, _= sess.run([cost,W,b,train], #리스트에 묶어서 한번에 실행시키기
        feed_dict={X:[1,2,3,4,5],Y:[2.1,3.1,4.1,5.1,6.1]}) #값 넘겨주기
    if step %20 ==0:
        print(step,cost_val, W_val,b_val)


# In[14]:


print(sess.run(hypothesis,feed_dict={X:[5]}))
print(sess.run(hypothesis,feed_dict={X:[2.5]}))
print(sess.run(hypothesis,feed_dict={X:[1.5,3.5]}))


# <정리>
# 1. H(x)=Wx+b 와 cost(W,b)의 식을 그래프로 만든다
# 2. sess.run(값을 넣을 변수,feed_dict={데이터 x,y}
# 3. W,b가 업데이트됨
