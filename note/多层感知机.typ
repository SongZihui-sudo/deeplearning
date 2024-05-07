#set page(
  paper: "a4",
  margin: (x: 1.8cm, y: 1.5cm)
)

#set heading(numbering: "1.1")
#show heading: it=>{
  if it.level == 1 {
    set align(center)
    counter(heading).display("一、")
    it.body
  } else {
    counter(heading).display()
    it.body
    parbreak()
  }
}
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
#set text(12pt,font:("STIX Two Text","Source Han Serif SC"))
#align(center, text(20pt)[
  多层感知机
])

#set text(14pt)
#set align(left)
#set par(justify: true,first-line-indent: 2em)

#show heading: it =>  {
    it
    par()[#text()[#h(0.0em)]]
}

= 简介
因为在一些任务中使用线性模型可能会存在较大的误差，所以可以通过增加隐蔽层的方式来来更好表示神经网络，这样就可以采用非线性的模型进行预测。
网络可以表示为：
#figure(
  image("./static/mlp.svg", width: 80%, height: 60%),
  caption: [
    网络图
  ],
) \

再上图中的神经网络中，输入为4，输出为3，隐藏层的神经元个数为5，总的层数为2。#text(font: "SimHei", style: "italic")[
  全连接层，是每一个结点都与上一层的所有结点相连
]。*隐藏层*和*输出层*都是全连接层。

= 隐蔽层

设隐藏层的输出为$X_h$，隐藏层的权重和偏差为$W_h$和$b_h$，输出为$O_h$。输出层的输入就是$X_o = O_h$也就是隐藏层的输出，输出层的权重和偏差为$W_I$和$W_I$，输出是$O_I$，这样就可以将这个网络表示为：$ O_h = W_h dot X_h + b_h \
                      hat(y) = O_I = W_I dot O_h + b_I $\

隐藏层的输出写成矩阵形式为:$ mat(o_("h1"); 
o_("h2"); 
o_("h3"); 
o_("h4");
o_("h5")) = mat(h_11, h_12,h_13, h_14;
              h_21, h_22,h_23, h_24,;
              h_31, h_32,h_33, h_34;
              h_41, h_42,h_43, h_44;
              h_51, h_52,h_53, h_54) times mat(x_1; x_2; x_3; x_4) + mat(b_"h1";b_"h2";b_"h3";b_"h4";b_"h5") $\

输出层的输出写成矩阵形式为：$ mat(y_1;y_2;y_3) = mat(o_11, o_12, o_13;
        o_21, o_22, o_23;
        o_31, o_32, o_33) times mat(o_("h1"); 
o_("h2"); 
o_("h3"); 
o_("h4");
o_("h5")) + mat(b_"o1";b_"o2";b_"o3") $\

#show math.equation: set text(12pt)
写到一起就是: $ mat(y_1;y_2;y_3) = mat(o_11, o_12, o_13;
        o_21, o_22, o_23;
        o_31, o_32, o_33) times  (mat(h_11, h_12,h_13, h_14;
              h_21, h_22,h_23, h_24,;
              h_31, h_32,h_33, h_34;
              h_41, h_42,h_43, h_44;
              h_51, h_52,h_53, h_54) times mat(x_1; x_2; x_3; x_4) + mat(b_"h1";b_"h2";b_"h3";b_"h4";b_"h5")) + mat(b_"o1";b_"o2";b_"o3") $ \

简写为：$ hat(y) = O_I * (H * X + b_("h")) + b_("I") = X W_h W_o + b_h W_o + b_o$

= 激活函数
在仿射变换之后对每个隐藏单元应用非线性的激活函数（activation function）$sigma$
。 激活函数的输出（例如，$sigma(.)$）被称为活性值（activations）。 一般来说，有了激活函数，就不可能再将我们的多层感知机退化成线性模型。

== ReLU函数
#show math.equation: set text(14pt)
ReLU函数可以写为：$ "ReLU"(x) = max(x, 0) $
 
#figure(
  image("./static/relu.svg", width: 100%, height: 30%),
  caption: [
    Relu 图像
  ],
) \

可以看出在这个函数中只保留了正数，负数全部清零。 

== sigmoid函数

simgmoid函数可以表示为：$ "sigmoid"(x) = 1 / (1 + exp(-x)) $

#figure(
  image("./static/sigmoid.svg", width: 80%, height: 30%),
  caption: [
    Sigmoid 图像
  ],
) \

sigmoid函数可以将元素的值变换到0和1之间

== tanh函数

tanh（双曲正切）函数可以表示为：$ tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x)) $

#figure(
  image("./static/tanh.svg", width: 80%, height: 30%),
  caption: [
    Tanh 图像
  ],
) \

tanh（双曲正切）函数可以将元素的值变换到-1和1之间。

= 多层感知机

在分类问题中，我们可以对输出 O 做softmax运算，并使用softmax回归中的交叉熵损失函数。 在回归问题中，我们将输出层的输出个数设为1，并将输出 O 直接提供给线性回归中使用的平方损失函数。

= 引用
[1] 4.1. 多层感知机 — 动手学深度学习 2.0.0 documentation[EB]. 
[2024-05-07]. https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html#id2.

[2] 3.8. 多层感知机 — 《动手学深度学习》 文档[EB]. [2024-05-07]. https://zh-v1.d2l.ai/chapter_deep-learning-basics/mlp.html.

[3] MLP公式推导及numpy实现numpy实现mlp-CSDN博客[EB]. [2024-05-07]. https://blog.csdn.net/qq_38955142/article/details/117475283.
