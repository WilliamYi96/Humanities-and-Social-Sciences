# 深入MNIST
前一部分主要是对于TF for Beginners部分的内容进行了代码优化，说明基本相同，因此这部分可以参照：[MNIST for ML Beginners](./MNIST_for_ML_beginners.md)     
以下内容主要是讨论如何将训练的准确率进行提升。

# 构建一个多层卷积网络
## 权值初始化
我们为不引致对称性和0梯度问题，因此需在模型中添加少量噪声。因此我们使用较小的正数来初始化偏置项，来避免神经元结点输出恒为0的问题。
我们使用如下两个函数进行初始化：

```python  
def weight_variable(shape):    
  initial = tf.truncated_normal(shape, stddev=0.1)    
  return tf.Variable(initial)    

def bias_variable(shape):    
  initial = tf.constant(0.1, shape=shape)    
  return tf.Variable(initial)    
```

## 卷积和池化
我们将卷积层和池化层定义为如下函数：

```python
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```
