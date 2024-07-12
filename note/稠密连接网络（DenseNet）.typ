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
  稠密连接网络（DenseNet）
])

= DenseNet

稠密连接网络（DenseNet） (Huang et al., 2017)在某种程度上是ResNet的逻辑扩展。

DenseNet（密集卷积网络）的核心思想（创新之处）是密集连接，使得每一层都与所有之前的层直接连接，即某层的输入除了包含前一层的输出外还包含前面所有层的输出。

整个网络主要是包含了三个核心结构，分别是DenseLayer（模型中最基础的原子单元，利用卷积完成一次最基础的特征提取）、DenseBlock(整个模型密集连接的基础单元，整个网络最核心的部分)和Transition(通常用于两个相邻的Dense块之间，主要的两个作用是减小特征图的大小和特征图的数量），通过上述的三个核心的结构的拼接加上其他层来完成整个模型的搭建。

#figure(
  image("/note/static/densenet.png", width: 100%),
  caption: [ ResNet 结构 ],
  kind: image
)\

== DenseLayer

== DenseBlock

== Transition

= 引用

1. 《5.12. 稠密连接网络（DenseNet） — 〈动手学深度学习〉 文档》. 见于 2024年7月12日. https://zh-v1.d2l.ai/chapter_convolutional-neural-networks/densenet.html.
2. 《7.7. 稠密连接网络（DenseNet） — 动手学深度学习 2.0.0 documentation》. 见于 2024年7月12日. https://zh.d2l.ai/chapter_convolutional-modern/densenet.html.
3. 《深度学习——稠密连接网络（DenseNet）原理讲解+代码（torch） densenet源码-CSDN博客》. 见于 2024年7月12日. https://blog.csdn.net/m0_74055982/article/details/137960751.
