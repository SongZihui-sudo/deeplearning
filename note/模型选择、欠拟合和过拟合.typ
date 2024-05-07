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
  模型选择、欠拟合和过拟合
])

#set text(13pt)
#set align(left)
#set par(justify: true,first-line-indent: 2em)

#show heading: it =>  {
    it
    par()[#text()[#h(0.0em)]]
}


#set text(11pt)
#set align(left)
= 训练误差和泛化误差
#set par(first-line-indent: 2em)
#let indent = h(2em)

........................

= 模型选择

== 验证数据集

== k折交叉验证
............................

= 模型复杂度

= 欠拟合与过拟合

== 欠拟合

== 过拟合