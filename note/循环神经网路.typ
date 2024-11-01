#import "@preview/tablex:0.0.8": tablex, rowspanx, colspanx

#set page(
  paper: "a4",
  margin: (x: 1.8cm, y: 1.5cm)
)

#set heading(numbering: (..nums) => {
if nums.pos().len()==1 {
numbering("一、", nums.pos().at(0))
}
else {
numbering("1.1", ..nums)
}
})

#show heading.where(level: 1): it => [
	#counter(math.equation).update(0) #it
]
#set math.equation(numbering: equation => locate(loc => {
	let chapter = counter(heading).at(loc).at(0)
	[(#numbering("1", chapter)  - #numbering("1", equation))]
}))
#set figure(numbering: equation => locate(loc => {
	let chapter = counter(heading).at(loc).at(0)
	[#numbering("1", chapter)  - #numbering("1", equation)]
}))
#set text(12pt,font:("STIX Two Text","Source Han Serif SC"))

#set text(14pt)
#set align(left)
#set par(justify: true,first-line-indent: 2em)

#show heading: it =>  {
    it
    par()[#text()[#h(0.0em)]]
}

#show figure.where(
  kind: table
): set figure.caption(position: top)
#show figure: set block(breakable: true)

#align(center, text(20pt)[
  循环神经网络
])

= 循环神经网络

== 序列模型

=== 自回归模型

利用前期若干时刻的随机变量的线性组合来描述以后某时刻随机变量的线性回归模型。
可以表示为：
$
y_t = phi_0 + phi_1y_(t - 1) + phi_2y_(t - 2) + ··· + phi_p y_(t - p) + e_t
$\

$phi_0$ ———— 常数项 \

$phi_1 ~ phi_p$ ———— 模型参数\

$e_t$ ———— 噪声

因为随着x与时间t不存在函数关系，所以相比于线性回归是用x预测y，自回归模型是使用x预测后面的x。

自然语言处理的输入输出基本上都是序列，序列问题是自然语言处理最本质的问题。

整个序列的估计值：

$
P(x_1,...,x_tau) = product_(t = 1)^(tau) P(x_t|x_(t - 1),..., x_1)
$

=== 马尔可夫模型

在已知目前状态（现在）的条件下，它未来的演变（将来）不依赖于它以往的演变 (过去 )。例如森林中动物头数的变化构成——马尔可夫过程。在现实世界中，有很多过程都是马尔可夫过程，如液体中微粒所作的布朗运动、传染病受感染的人数、车站的候车人数等，都可视为马尔可夫过程。（这里虽然我也不清楚这些现象到底是不是，姑且就认为是吧！）

在马尔可夫性的定义中，"现在"是指固定的时刻，但实际问题中常需把马尔可夫性中的“现在”这个时刻概念推广为停时（见随机过程）。

在马尔科夫过程中，在给定当前知识或信息的情况下，过去（即当前以前的历史状态）对于预测将来（即当前以后的未来状态）是无关的。这种性质叫做无后效性。简单地说就是将来与过去无关，值与现在有关，不断向前形成这样一个过程。

马尔可夫模型可以写为：

$
P(x_1,....,x_tau) = product_(t = 1)^(tau)P(x_t | x_(t - 1)) 当 P(x_1 | x_0) = P(x_1) 
$\

时间和状态都是离散的马尔可夫过程称为马尔可夫链,简记为Xn=X(n),n=0,1,2…马尔可夫链是随机变量X1,X2,X3…的一个数列。

这种离散的情况其实草是我们所讨论的重点，很多时候我们就直接说这样的离散情况就是一个马尔科夫模型。

=== 因果关系

将$P(x_1,...,x_tau)$倒叙展开，基于条件概率公式：
$
P(x_1,...,x_tau) = product_(x_t)^(1)x_(t + 1),...,x_tau
$

== 文本预处理

+ 将文本作为字符串加载到内存中。

+ 将字符串拆分为词元（如单词和字符）。

+ 建立一个词表，将拆分的词元映射到数字索引。

+ 将文本转换为数字索引序列，方便模型操作。
\

将文本数据映射为词元， 以及将这些词元可以视为一系列离散的观测，例如单词或字符。

== 语言模型和数据集

=== 学习语言模型

基本概率规则：见前面章节

$
P(x_1, x_2,...,x_tau) = product_(t = 1)^(tau) P(x_t|x_1,...,x_(t - 1))
$
这样一个有四个单词的文本序列就是：
$
P("deep", "learning", "is", "fun") = P("deep")P("learning"|"deep") \ P("is" | "deep", "learning") | P("fun"| "deep", "learning", "is")
$
可以写出：
$
hat(P)("learning" | "deep") = n("deep", "learning") / n("deep")
$\
因为单词组合不一定会出现，所以n可能为0，可以改为：
$
hat(P)(x) = (n(x) + epsilon_1 / m) / (n + epsilon_1)
$

$
hat(x'|x) = (n(x, x') + epsilon_2 hat(P)(x')) / (n(x) + epsilon_2) 
$

$
hat(P)(x''|x, x') = (n(x, x', x'') + epsilon_3 hat(P)(x'')) / (n(x, x') + epsilon_3)
$\

然而，这样的模型很容易变得无效，原因如下： 首先，我们需要存储所有的计数； 其次，这完全忽略了单词的意思。 例如，“猫”（cat）和“猫科动物”（feline）可能出现在相关的上下文中， 但是想根据上下文调整这类模型其实是相当困难的。 

=== 马尔可夫模型与 n 元语法

用于序列建模的近似公式：

$
P(x_1, x_2, x_3, x_4) = P(x_1)P(x_2)P(x_3)P(x_4)
$

$
P(x_1, x_2, x_3, x_4) = P(x_1)P(x_2 | x_1)P(x_3 | x_2)P(x_4 | x_3)
$

$
P(x_1, x_2, x_3, x_4) = P(x_1)P(x_2 | x_1)P(x_3 | x_1, x_2)P(x_4 | x_2, x_3)
$

== 循环神经网络

隐变量模型：

$
P(x_t|x_t - 1,....,x_1) approx P(x_t|h_(t - 1))
$\

$h_(t - 1)$ ———— 隐状态

使用当前输入$x_t$和先前隐状态$h_(t - 1)$来计算时间步$t$处的任何时间的隐状态：
$
h_t = f(x_t, h_(t - 1))
$

=== 无隐状态的神经网络

可以表示为：

$
H = phi("XW"_("xh") + b_h)
$

$
O = "HW"_("hq") + b_q
$

=== 有隐状态的循环神经网络

当前时间步隐藏变量由当前时间步的输入 与前一个时间步的隐藏变量一起计算得出：

$
H_t = phi(X_t W_("xh") + H_(t - 1)W_("hh") + b_h)
$

对于时间步 t ，输出层的输出类似于多层感知机中的计算：

$
O_t = H_t W_("hq") + b_q
$

#figure(
  image("/note/static/rnn.svg", width: 100%),
  caption: [ ResNet 结构 ],
  kind: image
)\

=== 基于循环神经网络的字符级语言模型

#figure(
  image("/note/static/rnn-train.svg", width: 100%),
  caption: [ ResNet 结构 ],
  kind: image
)\

困惑度的最好的理解是“下一个词元的实际选择数的调和平均数”。

== 梯度剪裁



= 引用

+ 《8. 循环神经网络 — 动手学深度学习 2.0.0 documentation》. 见于 2024年7月15日. https://zh.d2l.ai/chapter_recurrent-neural-networks/index.html.
+ 《马尔科夫模型系列文章（一）——马尔科夫模型-CSDN博客》. 见于 2024年7月15日. https://blog.csdn.net/qq_27825451/article/details/100117715.
+ 《自回归模型（AR Model）-CSDN博客》. 见于 2024年7月15日. https://blog.csdn.net/shigangzwy/article/details/69525576.


