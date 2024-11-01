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
  概率，语言模型
])

= 语言模型

语言也可以是视为一种时间序列模型。时间序列模型，输入的 $x_1, x_2, ......., x_t$ 本身也与 $t_1, t_2, ......, t_t$ 这些时间有关。通过自回归模型，可以使用前期的数据来预测未来某一时刻的数据。
$
y_t = phi_0 + phi_1y_(t - 1) + phi_2y_(t - 2) + ········· + phi_p"y"_(t-p) + e_t
$
其中 $phi_0$ 是常数项，$phi_1, ........., phi_p$ 是模型参数，$e_t$ 是白噪声。\
  
形势来看这个模型与线性回归相似。

