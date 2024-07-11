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
  正向传播和反向传播
])

= 正向传播和反向传播

#figure(
  image("/note/static/nn.drawio.svg", width: 80%),
  caption: [ Net ],
  kind: image
)\

== 正向传播

在上图中有 $x_1、x_2、x_3$ 三个输入，中间的隐藏层有$H_1, H_2$两个，最后输出。

在神经元计算时，就是$
  mat(x_1, x_2,x_3) dot mat(
    w_111, w_112, w_113;
    w_121, w_122, w_123;
    w_131, w_132, w_133)^T \
    = mat(x_1, x_2, x_3) dot mat(
      w_111, w_121, w_131; w_112, w_122, w_132; w_113, w_123, w_133; ) \
  = mat(x_1 dot w_111 + x_2 dot w_112 + x_3 dot w_113;
   x_1 dot w_121 + x_2 dot w_122 + x_3 dot w_123;
   x_1 dot w_131 + x_2 dot w_132 + x_3 dot w_133)\
   = mat(H_11;H_12;H_13)
$

$
mat(H_11; H_12;H_13)^T dot mat(
  w_211, w_212, w_213;
  w_221, w_222, w_223
)^T\
 = mat(H_11, H_12, H_13) dot mat(w_211, w_221;w_212, w_222; w_213, w_223)\
= mat(H_11 dot w_211 + H_12 dot w_212 + H_13 dot w_213;
H_11 dot w_221 + H_12 dot w_222 + H_13 dot w_223)\
= mat(H_21;H_22)
$

$
mat(H_21, H_22) dot mat(w_321, w_322)^T \
= mat(H_21 dot w_21 + H_22 dot w_22) \
 = mat("OUTPUT")
$\

从左到右的计算。在全连接神经网络中，每一层的每个神经元都会与前一层的所有神经元或者输入数据相连，每一个神经元的输出=使用激活函数激活前一层函数的累加和。

== 反向传播

反向传播算法(Backpropagation，简称BP算法)。P算法的学习过程由正向传播过程和反向传播过程组成。

在正向传播过程中，输入信息通过输入层经隐含层，逐层处理并传向输出层。如果预测值和教师值不一样，则取输出与期望的误差的平方和作为损失函数（损失函数有很多，这是其中一种）。将正向传播中的损失函数传入反向传播过程，逐层求出损失函数对各神经元权重的偏导数，作为目标函数对权重的梯度。根据这个计算出来的梯度来修改权重，网络的学习在权重修改过程中完成。误差达到期望值时，网络学习结束。

#figure(
  image("/note/static/nn_back.drawio.svg", width: 120%),
  caption: [ Net ],
  kind: image
)\

=== 误差计算

输出的误差可以写为 $delta = "cost"("OUTPUT")$，通过反向传播
$
  delta_(H_21) = delta dot w_321\
  delta(H_22) = delta dot w_322
$

$
delta_(H_11) = delta_(H_21) dot w_211 + delta_(H_22) dot w_221\
delta_(H_12) = delta_(H_21) dot w_212 + delta_(H_22) dot w_222\
delta_(H_13) = delta_(H_21) dot w_213 + delta_(H_22) dot w_(223)
$

=== 梯度下降

最后使用梯度下降法来更新权重

梯度下降可以表示为：
$
theta = theta - alpha partial(J(theta)) / partial(theta)
$

#figure(
  image("/note/static/梯度下降图片.png", width: 100%),
  caption: [ 梯度下降 ],
  kind: image
)\

可以看出在A点时，函数单调递增，导数为正，减去学习率乘导数，x逐渐向最小值靠近；在B点时，函数单调递减，导数为负，这样就相当于加上学习率乘导数，x向右移动，逐渐向最小值逼近。

$
w_111 = w_111 - alpha_11 theta_(H_11) partial(H_11 (a)) / partial(a) x_1\
w_211 = w_211 - alpha_12 theta_(H_21) partial(H_21 (b)) / partial(b) a\
w_311 = w_311 - alpha_13 theta_("OUTPUT") partial("OUTPUT"(c)) / partial(c) b
$

= 引用

1. 《机器学习笔记丨神经网络的反向传播原理及过程（图文并茂+浅显易懂） 神经网络反向传播原理-CSDN博客》. 见于 2024年7月11日. https://blog.csdn.net/fsfjdtpzus/article/details/106256925.
2. 《小白零基础学习：详解梯度下降算法：完整原理+公式推导+视频讲解 标准梯度下降算法 损函数-CSDN博客》. 见于 2024年7月11日. https://blog.csdn.net/zhouaho2010/article/details/102756411.
