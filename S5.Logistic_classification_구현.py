#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

#x=>[x1,x2], y=[0 or 1]
x_data=[[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data=[[0],[0],[0],[1],[1],[1]]

#X,Y는 유동적이므로 placeholder로 설정
X=tf.placeholder(tf.float32,shape=[None,2])
Y=tf.placeholder(tf.float32,shape=[None,1])
#W=>[들어오는 var 개수,나가는 Y 개수]
W=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

#sigmoid를 사용해서 가설 설정. 1/(1+e^(WX)) = sigmoid 함수
hypothesis=tf.sigmoid(tf.matmul(X,W)+b)

#cost func 그래로 적어주기
cost=-tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

#기울기로 최소값을 찾는 방법
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#정확도 계산. 가설>0.5이면 True, 아니면 False
#'hypothesis >0.5'를 'dtype=tf.float32'로, 즉 실수로 casting.
# float으로 casting : True=>1, False=>0 로 나타내는 것
predicted=tf.cast(hypothesis >0.5, dtype=tf.float32)
#정확도 : 예측한 값과 Y값이 같은지('equal(predicted,Y)') cast후에 평균을 구하면 정확도 알수있음.
    #10번 시도 중 predicted=Y 가 7번 -> accuracy=0.7
accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

#model training 시작
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val,_ = sess.run([cost,train],feed_dict={X:x_data, Y:y_data})
        if step %200 == 0:
            print(step,cost_val)
            
    #Accuracy report
    h,c,a = sess.run([hypothesis,predicted,accuracy],feed_dict={X:x_data,Y:y_data})
    print("\nHypothesis: \n",h, "\nCorrect(Y): \n",c,"\nAccuracy: ",a)

