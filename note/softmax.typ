#set page(
  paper: "a4",
  margin: (x: 1.8cm, y: 1.5cm)
)

#set heading(numbering: "1.")
#set math.equation(numbering: equation => locate(loc => {
	let chapter = counter(heading).at(loc).at(0)
	[(#numbering("1", chapter)  - #numbering("1", equation))]
}))
#show heading.where(level: 1): it => [
	#counter(math.equation).update(0) #it
]
#set figure(numbering: equation => locate(loc => {
	let chapter = counter(heading).at(loc).at(0)
	[#numbering("1", chapter)  - #numbering("1", equation)]
}))

#set align(center)
#set text(24pt)
Softmax 回归

#set text(11pt)
#set align(left)
= 线性回归

#link("https://promhub.top/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92.html")


= 逻辑回归
#set par(first-line-indent: 2em)
#let indent = h(2em)

#indent 在逻辑回归中可以通过一条直线来划分两种不同的类型，以此可以解决二分类问题。
在逻辑回归中通过表示为： $ hat(y) := cases(
  1 "if" f(x) > 0.5,
  0 "if" f(x) <= 0.5
) $ 以输出值为0.5作为界限进行区分。

拟合的方程$f(x)$ 写为：$ f(x) = w dot x $ 与线性回归相似，在单特征时有一个特征w，一个输入w，一个输出y，偏置b。

这样进而就可以由 $f(x)$ 写出一个x关于$hat(y)$的表达式。
$ hat(y) = cases(
  1 "if" x > f^(-1)(0.5) ,
  0 "if" x <= f^(-1)(0.5)
) $这样就可以通过输入x得出输出y。

这个函数是间断的，不适用于使用梯度下降法进行处理，这样就可以使用更加平滑的 sigmond 函数来代替。sigmond函数可以写为$ "f(z)" = frac(1, 1 + e^(-z)) \
                   z = w dot x $
== 极大似然估计

#indent 概率描述的是一个事件发生的可能性，似然性是放过来在事件发生后反推在什么参数条件下，这个事件发生的概率最大（例如下雨（果）的时候有乌云（因/证据/观察的数据）的概率，即已经有了果，对证据发生的可能性描述）。数学语言可以将其写为：
$ P(x|beta) = a \
 L(beta|x) = a \
 P(x|beta) = L(beta|x) $ 二者是相等的。

在二分类问题中可以概率写为：
$ P(y|x, beta)=P(y=1)^(y)P(Y=0)^(1-y) \
  P(y|x, beta) = frac(1, 1 + e^(-z))^y (1-frac(1, 1 + e^(-z)))^(1-y) $。

#show math.equation: set text(13pt)
得到最大值：$ L(beta) = attach(
  Pi,
  t: n,
  b: i=1
)P(y_i|x_i,beta) = attach(
  Pi,
  t: n,
  b: i=1
)  frac(1, 1 + e^(-w dot x_i))^(y_i) (1-frac(1, 1 + e^(-w dot x_i)))^(1-y_i)  $

#show math.equation: set text(11pt)
对其取对数可得：
$ log(L(beta)) = sum^(n)_(i=1)(y_i dot log(frac(1, 1 + e^(-w dot x_i))^(y_i)) + log((1-y_i) dot (1-frac(1, 1 + e^(-w dot x_i)))^(1-y_i))) $

== 损失函数

#indent 损失函数是用于衡量预测值与实际值的偏离程度，即模型预测的错误程度。通过寻找误差最小时的参数来确定出最佳的模型。

使用平方损失：
$ Q = sum^n_(i = 1)(y_i - hat(y_i))^2 $

待入到逻辑回归中损失函数就是：
$ Q = sum^n_(i = 1)(y_i - 1 / (1 + e^(-w * x))) $因为这个函数不死凸函数，所以使用对数损失函数。
$ Q = -log(L(beta))  $

= 多元线性回归

#indent 多元线性回归相比于线性回归，特征与输入多余线性回归。形式为：$ hat(y) = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + ......... + b $ 可以将其写成矩阵的形式：$ mat(x_1, x_2, x_3) dot mat(w_1;w_2;w_3) + mat(b) = w_1x_1 + w_2x_2 + w_3x_3 + b$
相当于是一个点乘。

进而可以用矩阵$X = mat(x_11, x_12, x_13;x_21, x_22, x_23;x_31, x_32, x_33)$ 来表示数据集，一行表示一组数据，一列对应与某特征值。使用矩阵$W = mat(w_1, w_2, w_3)$ 来表示特征, 偏置可以表示为：$mat(b)$。这样计算矩阵：$ hat(y) = X times W^T + b= mat(mat(x_11, x_12, x_13) dot mat(w_1;w_2;w_3) + b;
mat(x_21, x_22, x_23) dot mat(w_1;w_2;w_3) + b;
mat(x_31, x_32, x_33) dot mat(w_1;w_2;w_3) + b) $

使用平方损失函数：
$ Q = sum^n_(i = 1)(y_i - hat(y_i))^2 $

= Softmax 回归

#indent 在前面的逻辑回归，它主要应用与二分类问题，对于多分类问题，有多个输入，多个特征，多个输出。

#figure(
  image("./static/softmaxreg.svg", width: 70%),
  caption: [
    网络图
  ],
)
可以表示为：$ o_1 = x_1w_11 + x_2w_12 + x_2w_13+ b_1 \
o_2 = x_1w_21 + x_2w_22 + x_2w_23+ b_2 \
o_3 = x_1w_31 + x_2w_32 + x_2w_33+ b_3 $ 使用三个线性的表达式表示。

将其写成矩阵形式，可以表示为：

因为由上述的线性表达式得出的结果可能为正可能为负，三个输出之和也可能大于1，这样都会违背概率的定理，所以使用softmax函数来规范得到的结果。