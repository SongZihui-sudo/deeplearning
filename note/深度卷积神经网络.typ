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
  深度卷积神经网络
])

= Alexnet

计算机视觉研究人员相信，从对最终模型精度的影响来说，更大或更干净的数据集、或是稍微改进的特征提取，比任何学习算法带来的进步要大得多。

#figure(
  image("/note/static/alexnet.svg", width: 85%),
  caption: [Alexnet],
  kind: image
)\

在AlexNet的第一层，卷积窗口的形状是$11 times 11$。 由于ImageNet中大多数图像的宽和高比MNIST图像的多10倍以上，因此，需要一个更大的卷积窗口来捕获目标。 第二层中的卷积窗口形状被缩减为$5 times 5$，然后是$3 times 3$。 此外，在第一层、第二层和第五层卷积层之后，加入窗口形状为$3 times 3$、步幅为2的最大汇聚层。 而且，AlexNet的卷积通道数目是LeNet的10倍。

在最后一个卷积层后有两个全连接层，分别有4096个输出。

AlexNet将sigmoid激活函数改为更简单的ReLU激活函数。 一方面，ReLU激活函数的计算更简单，它不需要如sigmoid激活函数那般复杂的求幂运算。 另一方面，当使用不同的参数初始化方法时，ReLU激活函数使训练模型更加容易。

AlexNet通过暂退法（ 4.6节）控制全连接层的模型复杂度，而LeNet只使用了权重衰减。

= 暂退法

一个好的模型需要对输入数据的扰动鲁棒, 与之前加入的噪音不一样，之前是固定噪音，丢弃法是随机噪音，丢弃法不是在输入加噪音，而是在层之间加入噪音，所以丢弃法也算是一个正则。

丢弃法对上一层输出向量的每一个元素做如下扰动：

$
x' = cases(
  0 "if" "概率为p" ,
  x / (1 - p) "if" "其他情况"
)
$\

= 引用

[1] 《4.6. 暂退法（Dropout） — 动手学深度学习 2.0.0 documentation》. 见于 2024年7月7日. https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html#sec-dropout. \

[2] 《7.1. 深度卷积神经网络（AlexNet） — 动手学深度学习 2.0.0 documentation》. 见于 2024年7月7日. https://zh.d2l.ai/chapter_convolutional-modern/alexnet.html.

