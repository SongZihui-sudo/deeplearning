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
  使用块的网络（VGG）
])

= VGG

#figure(
  image("/note/static/vgg.svg", width: 85%),
  caption: [VGG],
  kind: image
)\

一个VGG块与之类似，由一系列卷积层组成，后面再加上用于空间下采样的最大汇聚层。在最初的VGG论文中 (Simonyan and Zisserman, 2014)，作者使用了带有$3 times 3$卷积核、填充为1（保持高度和宽度）的卷积层，和带有汇聚窗口 $2 times 2$ 、步幅为2（每个块后的分辨率减半）的最大汇聚层。

VGG神经网络连接 图7.2.1的几个VGG块（在vgg_block函数中定义）。其中有超参数变量conv_arch。该变量指定了每个VGG块里卷积层个数和输出通道数。

原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。 第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11。

= 引用

[1] 《7.2. 使用块的网络（VGG） — 动手学深度学习 2.0.0 documentation》. 见于 2024年7月8日. https://zh.d2l.ai/chapter_convolutional-modern/vgg.html.

[2] 《CNN经典网络模型（三）：VGGNet简介及代码实现（PyTorch超详细注释版） vggnet是哪年发明的-CSDN博客》. 见于 2024年7月8日. https://blog.csdn.net/qq_43307074/article/details/126027852.
