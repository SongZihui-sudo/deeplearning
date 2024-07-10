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
  卷积神经网络
])

= 卷积

== 定义

卷积是当把一个函数“翻转”并移位 x 时，测量
和之间的重叠。

卷积 $f * g(x)$ 可以表示为：
$
  f * g(x) = integral f(z)g(x - z) d z
$

对象为离散的形式为：求和的形式

$
  (f * g)(i) = sum_i f(a)g(i - a)
$

对于二维的张量：
$
  (f * g)(i, j) = sum_a sum_b f(a, b)g(i -a, j -b)
$

== 卷积的计算过程

通常在卷积层中使用更加直观的互相关（cross-correlation）运算。在二维卷积层中，一个二维输入数组和一个二维核（kernel）数组通过互相关运算输出一个二维数组。 

为了得到卷积运算的输出，我们只需将核数组左右翻转并上下翻转，再与输入数组做互相关运算。

#figure(
  image("/note/static/conv.gif", height: 55%),
  caption: [卷积的计算过程],
  kind: image
)

== 感受野

在卷积神经网络中，感受野是指在输入图像上，一个神经元（或特征图中的一个元素）可以看到或响应的区域大小。更具体地说，感受野是指输入图像的一个区域，这个区域中的像素会影响到卷积层或池化层中特定位置的输出值。
感受野的计算取决于卷积层和池化层的排列方式、卷积核大小、步幅和填充方式。

感受野的计算可以表示为：

$
  R_n = R_(n - 1) + (k_n - 1) dot s_(n - 1)
$\
· $R_n$ 是第n层的感受野大小\
· $R_(n - 1)$ 是第n - 1层的感受野大小\
· $k_n$ 是第n层的卷积核大小\
· $s_(n - 1)$ 是第n - 1层的步幅大小

== 输出大小

$
  cases("height"_"out" = ("height"_"in" - "height"_"kernel" + 2 times "padding") / "stride" + 1,
  "width"_"out" = ("width"_"in" - "width"_"kernel" + 2 times "padding") / "stride" + 1
  )
$\

全连接层的计算实例

#figure(
  image("/note/static/conv_example_linear.jpg", height: 10%),
  caption: [全连接层的计算实例],
  kind: image
)

== 平移不变性

不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。


== 局部性

神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。

== 多输入通道的情况

#figure(
  image("/note/static/conv-multi-in.svg", width: 85%),
  caption: [多通道情况 卷积的计算过程],
  kind: image
)

$ (1 times 1 + 2 times 2+4 times 3+5 times 4) + (0 times 0+1 times 1+3 times 2+4 times 3) = 56 $

= 池化层 （汇聚层）

降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。

不同于卷积层里计算输入和核的互相关性，池化层直接计算池化窗口内元素的最大值或者平均值。该运算也分别叫做最大池化或平均池化。

#figure(
  image("/note/static/pooling.svg", width: 85%),
  caption: [池化],
  kind: image
)\

$
max(0,1,3,4)=4,\
max(1,2,4,5)=5,\
max(3,4,6,7)=7,\
max(4,5,7,8)=8
$\

池化层也可以在输入的高和宽两侧的填充并调整窗口的移动步幅来改变输出形状。

在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加。这意味着池化层的输出通道数与输入通道数相等。

= 卷积神经网络

卷积层尝试解决这两个问题。一方面，卷积层保留输入形状，使图像的像素在高和宽两个方向上的相关性均可能被有效识别；另一方面，卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大。

== LeNet模型

LeNet分为卷积层块和全连接层块两个部分。

卷积层块里的基本单位是卷积层后接最大池化层：卷积层用来识别图像里的空间模式，如线条和物体局部，之后的最大池化层则用来降低卷积层对位置的敏感性。卷积层块由两个这样的基本单位重复堆叠构成。在卷积层块中，每个卷积层都使用 5×5 的窗口，并在输出上使用sigmoid激活函数。第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。这是因为第二个卷积层比第一个卷积层的输入的高和宽要小，所以增加输出通道使两个卷积层的参数尺寸类似。卷积层块的两个最大池化层的窗口形状均为 2×2 ，且步幅为2。由于池化窗口与步幅形状相同，池化窗口在输入上每次滑动所覆盖的区域互不重叠。

卷积层块的输出形状为(批量大小, 通道, 高, 宽)。当卷积层块的输出传入全连接层块时，全连接层块会将小批量中每个样本变平（flatten）。

#figure(
  image("/note/static/lenet.svg", width: 85%),
  caption: [Lenet],
  kind: image
)\

#figure(
  image("/note/static/lenet-vert.svg", height: 30%),
  caption: [Lenet],
  kind: image
)\

= 引用

[1] 【深度学习】一文搞懂卷积神经网络（CNN）的原理（超详细） 卷积神经网络原理-CSDN博客[EB/OL]. [2024-07-03]. https://blog.csdn.net/AI_dataloads/article/details/133250229.


[2] 【深度学习入门】——亲手实现图像卷积操作 输入是图像和使用的卷积核-CSDN博客[EB/OL]. [2024-07-03]. https://blog.csdn.net/rocling/article/details/103831994.

[3] 5. 卷积神经网络 — 《动手学深度学习》 文档[EB/OL]. [2024-07-03]. https://zh-v1.d2l.ai/chapter_convolutional-neural-networks/index.html.

[4] 6. 卷积神经网络 — 动手学深度学习 2.0.0 documentation[EB/OL]. [2024-07-03]. https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html.