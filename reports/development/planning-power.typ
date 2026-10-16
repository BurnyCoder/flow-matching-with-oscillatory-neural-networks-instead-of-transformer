#let horizontalrule = line(start: (25%,0%), end: (75%,0%))

#show terms.item: it => block(breakable: false)[
  #text(weight: "bold")[#it.term]
  #block(inset: (left: 1.5em, top: -0.4em))[#it.description]
]

#set table(
  inset: 6pt,
  stroke: none
)

#show figure.where(
  kind: table
): set figure.caption(position: top)

#show figure.where(
  kind: image
): set figure.caption(position: bottom)

#let content-to-string(content) = {
  if content.has("text") {
    content.text
  } else if content.has("children") {
    content.children.map(content-to-string).join("")
  } else if content.has("body") {
    content-to-string(content.body)
  } else if content == [ ] {
    " "
  }
}
#let conf(
  title: none,
  subtitle: none,
  authors: (),
  keywords: (),
  date: none,
  abstract-title: none,
  abstract: none,
  thanks: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: none,
  fontsize: 11pt,
  mathfont: none,
  codefont: none,
  linestretch: 1,
  sectionnumbering: none,
  linkcolor: none,
  citecolor: none,
  filecolor: none,
  pagenumbering: "1",
  doc,
) = {
  set document(
    title: title,
    keywords: keywords,
  )
  set document(
      author: authors.map(author => content-to-string(author.name)).join(", ", last: " & "),
  ) if authors != none and authors != ()
  set page(
    paper: paper,
    margin: margin,
    numbering: pagenumbering,
    columns: cols
  )

  set par(
    justify: true,
    leading: linestretch * 0.65em
  )
  set text(lang: lang,
           region: region,
           size: fontsize)

  set text(font: font) if font != none
  show math.equation: set text(font: mathfont) if mathfont != none
  show raw: set text(font: codefont) if codefont != none

  set heading(numbering: sectionnumbering)

  show link: set text(fill: rgb(content-to-string(linkcolor))) if linkcolor != none
  show ref: set text(fill: rgb(content-to-string(citecolor))) if citecolor != none
  show link: this => {
    if filecolor != none and type(this.dest) == label {
      text(this, fill: rgb(content-to-string(filecolor)))
    } else {
      text(this)
    }
  }

  block(below: 1em, width: 100%)[
    #if title != none {
      align(center, block[
          #text(weight: "bold", size: 1.5em, hyphenate: false)[#title #if thanks != none {
              footnote(thanks, numbering: "*")
              counter(footnote).update(n => n - 1)
            }]
          #(
            if subtitle != none {
              parbreak()
              text(weight: "bold", size: 1.25em, hyphenate: false)[#subtitle]
            }
           )])
    }

    #if authors != none and authors != [] {
      let count = authors.len()
      let ncols = calc.min(count, 3)
      grid(
        columns: (1fr,) * ncols,
        row-gutter: 1.5em,
        ..authors.map(author => align(center)[
          #author.name \
          #author.affiliation \
          #author.email
        ])
      )
    }

    #if date != none {
      align(center)[#block(inset: 1em)[
          #date
        ]]
    }

    #if abstract != none {
      block(inset: 2em)[
        #text(weight: "semibold")[#abstract-title] #h(1em) #abstract
      ]
    }
  ]

  doc
}
#show: doc => conf(
  abstract-title: [Abstract],
  paper: "a4",
  pagenumbering: "1",
  cols: 1,
  doc,
)


= Planning power analysis
<planning-power-analysis>
This reproduces the pre-training simulation in the protocol. It is not a
post-hoc estimate from observed KID differences. Six pairs were retained
despite low moderate-effect power because the 24-hour budget defines an
exploratory study. Common Gaussian draws are shifted across effect
scenarios; all 64 sign assignments are enumerated for every simulated
study. The null rejection rate is below 0.05 because the exact test is
discrete.

```json
{
  "seed": 20261016,
  "trials": 20000,
  "pairs": 6,
  "alpha": 0.05,
  "model": "independent Gaussian paired differences",
  "common_random_numbers_across_effects": true,
  "results": [
    {
      "standardized_effect": 0,
      "power": 0.02985,
      "monte_carlo_se": 0.0012033074731754973
    },
    {
      "standardized_effect": 0.5,
      "power": 0.1133,
      "monte_carlo_se": 0.0022412397239028226
    },
    {
      "standardized_effect": 0.8,
      "power": 0.2406,
      "monte_carlo_se": 0.0030225125309913934
    },
    {
      "standardized_effect": 1,
      "power": 0.35465,
      "monte_carlo_se": 0.0033828492243965
    },
    {
      "standardized_effect": 1.5,
      "power": 0.6567,
      "monte_carlo_se": 0.0033574179811277595
    },
    {
      "standardized_effect": 2,
      "power": 0.8685,
      "monte_carlo_se": 0.002389641709545596
    }
  ]
}
```

Source:
#link("https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html")[SciPy permutation-test documentation].
