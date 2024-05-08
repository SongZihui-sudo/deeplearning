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
  Title
])

#set text(14pt)
#set align(left)
#set par(justify: true,first-line-indent: 2em)

#show heading: it =>  {
    it
    par()[#text()[#h(0.0em)]]
}