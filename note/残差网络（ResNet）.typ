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
  残差网络（ResNet）
])

= ResNet

传统的卷积神经网络有以下几个问题：
\
\
- 梯度消失或梯度爆炸。
- 退化问题。
\

在网络中使用BN（Batch Normalization）层能够解决梯度消失或者梯度爆炸问题；ResNet论文提出了residual结构（残差结构）来减轻退化问题。

残差即观测值与估计值之间的差。

设要求解的映射为$F(x)$，输入为x，那么残差可以表示为：
$
H(x) = F(x) - x
$\

当 $x = F(x)$ 时，残差为0，这是就没有出现退化，输出等于上一层的输入。实际情况下残差$F(x)$不会为0，x肯定是很难达到最优的

#figure(
  image("/note/static/residual-block.svg", width: 70%),
  caption: [ 设输入为 x
 。假设图中最上方激活函数输入的理想映射为 f(x)
 。左图虚线框中的部分需要直接拟合出该映射 f(x)
 ，而右图虚线框中的部分需要拟合出有关恒等映射的残差映射 f(x)−x],
  kind: image
)\

#figure(
  image("/note/static/resnet1.png", width: 80%),
  kind: image
)
#figure(
  image("/note/static/resnet2.png", width: 80%),
  caption: [ ResNet 结构 ],
  kind: image
)\

= 引用

+ 《5.11. 残差网络（ResNet） — 〈动手学深度学习〉 文档》. 见于 2024年7月11日. https://zh-v1.d2l.ai/chapter_convolutional-neural-networks/resnet.html.
+ 《7.6. 残差网络（ResNet） — 动手学深度学习 2.0.0 documentation》. 见于 2024年7月11日. https://zh.d2l.ai/chapter_convolutional-modern/resnet.html.
+ 《Deep Residual Learning for Image Recognition | IEEE Conference Publication | IEEE Xplore》. 见于 2024年7月11日. https://ieeexplore.ieee.org/document/7780459.
+ 《Resnet详解：从原理到结构 resnet block-CSDN博客》. 见于 2024年7月11日. https://blog.csdn.net/m0_54487331/article/details/112758795.
+ 《ResNet——CNN经典网络模型详解(pytorch实现) resnet-cnn-CSDN博客》. 见于 2024年7月11日. https://blog.csdn.net/weixin_44023658/article/details/105843701.
