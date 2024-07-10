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
  含并行连结的网络（GoogLeNet）
])

= Inception 块

在GoogLeNet中，基本的卷积块被称为Inception块（Inception block）。

Inception块由四条并行路径组成。 前三条路径使用窗口大小为$1 times 1$、$3 times 3$和$5 times 5$ 的卷积层，从不同空间大小中提取信息。 中间的两条路径在输入上执行 $1 times 1$
卷积，以减少通道数，从而降低模型的复杂性。 第四条路径使用 $3 times 3$
最大汇聚层，然后使用 $1 times 1$
卷积层来改变通道数。 这四条路径都使用合适的填充来使输入与输出的高和宽一致，最后我们将每条线路的输出在通道维度上连结，并构成Inception块的输出。

#figure(
  image("/note/static/inception.svg", width: 85%),
  caption: [inception 块],
  kind: image
)\

= GoogLeNet模型

GoogLeNet一共使用9个Inception块和全局平均汇聚层的堆叠来生成其估计值。Inception块之间的最大汇聚层可降低维度。 

第一个模块使用64个通道、$7 times 7$ 卷积层。

第二个模块使用两个卷积层：第一个卷积层是64个通道、$1 times 1$ 卷积层；第二个卷积层使用将通道数量增加三倍的 $3 times 3$ 卷积层。 这对应于Inception块中的第二条路径。

第三个模块串联两个完整的Inception块。 

第四模块更加复杂， 它串联了5个Inception块。

第五模块包含输出通道数为 $256 + 320 + 128 + 128 = 832$
和 $384 + 384 + 128 + 128 = 1024$
的两个Inception块。

#figure(
  image("/note/static/inception-full.svg", width: 55%),
  caption: [GoogleNet],
  kind: image
)\

= 引用

[1] 《7.4. 含并行连结的网络（GoogLeNet） — 动手学深度学习 2.0.0 documentation》. 见于 2024年7月8日. https://zh.d2l.ai/chapter_convolutional-modern/googlenet.html.

