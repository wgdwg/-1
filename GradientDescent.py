#梯度下降求解线性回归
import numpy as np #导入计算包
import matplotlib.pyplot as plt #导入绘图包
#1.加载数据源,并画出散点图
points = np.genfromtxt('data.csv', delimiter=',')

#提取points的两列数据，分别作为x,y
x = points[:,0] #第一列数据
y = points[:,1] #第2列数据

#用plt画出散点图
plt.scatter(x,y)
plt.show()

#2.定义损失函数,损失函数是系数的函数
def cost_function(w, b, points):
    total_const = 0 #初始化损失函数
    m = len(points) #数据个数
    #逐点计算平方误差，然后求平均数
    for i in range(m):
        x = points[i,0] #第i行，第1列
        y = points[i,1] #第i行，第2列
        total_const += (y - w * x - b) ** 2
    return total_const / m

#3.定义模型的超参数
alpha = 0.0001 #学习速率
init_w = 0 #初始参数
init_b = 0 #初始参数
num_iter = 10 #迭代次数

#4.定义核心梯度下降算法函数
def grad_desc(points, init_w, init_b, num_iter):
    w = init_w
    b = init_b
    #定义一个列表cost_list保存所有损失函数值
    cost_list = []
    for i in  range(num_iter):
        cost_list.append(cost_function(w,b,points)) #存入损失函数
        w, b = step_grad_desc(w,b,alpha,points) #不挺的更新参数w,b
    return [w, b, cost_list]

def step_grad_desc(current_w, current_b, alpha, points):
    sum_grad_b = 0
    sum_grad_w = 0
    m = len(points)
    #对每个点，带入公式求和
    for i in range(m):
        x = points[i, 0]  # 第i行，第1列
        y = points[i, 1]  # 第i行，第2列
        sum_grad_w += (current_w * x + current_b - y) * x
        sum_grad_b += (current_w * x + current_b - y) ;
    #求当前的偏导
    grad_w = 2 / m * sum_grad_w
    grad_b = 2 / m * sum_grad_b
    #梯度下降，沿负梯度方向，更新当前的w和b
    update_w = current_w - alpha * grad_w
    update_b = current_b - alpha * grad_b

    return update_w, update_b

#5.测试，运行梯度下降，计算最优的w和b
w, b, cost_list = grad_desc(points, init_w, init_b, num_iter)
cost = cost_function(w,b,points) #得到损失函数

print("参数w = ", w)
print("参数b = ", b)
print("损失函数 = ", cost)
plt.plot(cost_list)
plt.show()

#6.画出拟合曲线
plt.scatter(x,y) #原始的散点图
pred_y = (w * x) + b #预测的y
plt.plot(x,pred_y,c='r') #红色的拟合直线
plt.show() #显示绘图

