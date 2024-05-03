#set page(
  paper: "a4",
  margin: (x: 1.8cm, y: 1.5cm)
)

#set heading(numbering: "1.")
#set math.equation(numbering: equation => locate(loc => {
	let chapter = counter(heading).at(loc).at(0)
	[(#numbering("1", chapter)  - #numbering("1", equation))]
}))
#show heading.where(level: 1): it => [
	#counter(math.equation).update(0) #it
]

#set align(center)
#set text(20pt)
Title

#set text(11pt)
#set align(left)
= Title1
#set par(first-line-indent: 2em)
#let indent = h(2em)

........................

= Title2
............................