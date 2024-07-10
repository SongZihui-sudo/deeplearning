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
  数据标准化
])

*让每一个特征都服从标准正态分布，消除了每个特征分布不同从而导致结果不同的这个因素*

数据标准化就是将原来分布范围不同的数据缩放在一个范围之内，一般来说是标准化到均值为0，标准差为1的标准正态分布，均值为0是为了让数据中心化，让各个维度的特征都平衡，标准差为1是为了让数据在各个维度上的长度都呈现一个单位向量(矩阵)，也就是把原来大小范围分布不通的数据缩放成统一量纲。

数据可以分为有界数据和无界数据。有界数据就是有着明确的数据边界，也就是数据是固定的，无界数据就是没有明确数据边界，也就是说数据是有可能会改变的。

对于有界数据的标准化方法，由于数据的边界是固定的，显然会和数据的数量以及最小值、最大值有关系，所以一般会将原数据与数据数量大小、最小值及最大值进行计算操作，从而将数据按照一定的比例缩放在一定的范围内。无界数据是指数据的边界不一定是固定的，也就是数据的边界不确定，随时有可能发生变化，或者不知道数据的边界，这类数据主要是使用均值方差的方法标准化。

= 数据标准化的方法

== 最大值标准化

最大值标准化就是让数据中的每个值都除以最大值，把数据缩放到[0,1]之间： 
$
x = x / max(x)
$
这种方法适合数据都是正数的情况，比如图像像素是[0,255]之间的正数

== 绝对最大值标准化

数据中存在负数时，先对数据取绝对值，然后再进行最大值标准化把数据缩放到[0,1]之间：
$
x=x/max(abs(x))
$

== 最大最小值标准化

就是让每一个数据都减去最小值，然后除以最大值减去最小值的结果，将数据缩放到[0,1]之间：
$
x=(x−min(x))/(max(x)−min(x))
$

== 均值方差标准化

均值方差（标准差）标准化是最为常用的数据标准化方法，操作过程就是让每个数据都减去均值，然后除以标准差：$x=(x-"mean"(x))/sqrt(x-"mean"(x)^2)$  ，目的是将数据缩放到均值为0，标准差为1的标准正态分布N[0,1]。

= 批量规范化

仅仅对原始输入数据进行标准化是不充分的，因为虽然这种做法可以保证原始输入数据的质量，但它却无法保证隐藏层输入数据的质量。浅层参数的微弱变化经过多层线性变换与激活函数后被放大，改变了每一层的输入分布，造成深层的网络需要不断调整以适应这些分布变化，最终导致模型难以训练收敛。

在 BN 层中，输入 $x:B = {x_1,...,m}$，输出:规范化后的网络响应${y_i = "BN"_(gamma, beta)(x_i)}$

== 计算数据均值

$
mu_B = 1 / m sum^m_(i = 1)x_i
$

== 计算数据方差

$
sigma^2_B = 1 / m sum^m_(i = 1)(x_i - mu_B)^2
$

== 规范化

$
hat(x_i) = (x_i - mu_B) / sqrt(sigma^2_B + epsilon )
$

= 引用

[1] 《7.5. 批量规范化 — 动手学深度学习 2.0.0 documentation》. 见于 2024年7月9日. https://zh.d2l.ai/chapter_convolutional-modern/batch-norm.html.\

[2] 《[1502.03167] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》. 见于 2024年7月9日. https://arxiv.org/abs/1502.03167.
《批量规范化（Batch Normalization，BN） 激活函数和批量标准化的区别-CSDN博客》. 见于 2024年7月9日. https://blog.csdn.net/jgj123321/article/details/105291672.\

[3] 《深度学习之数据标准化方法综述 ai数据标准化-CSDN博客》. 见于 2024年7月9日. https://blog.csdn.net/DeepAIedu/article/details/124281964.\

[4] Ioffe, Sergey, 和Christian Szegedy. 《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》. arXiv, 2015年3月2日. https://doi.org/10.48550/arXiv.1502.03167.\

[5] 《Pytorch归一化(MinMaxScaler、零均值归一化)-CSDN博客》. 见于 2024年7月9日. https://blog.csdn.net/qq_36158230/article/details/120925154.\

