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


= Training: mnist-smoke
<training-mnist-smoke>
== Purpose and interpretation
<purpose-and-interpretation>
This run fits the pre-specified velocity MSE objective. Completion and
held-out error establish execution; generation quality requires its
separate evaluation. No checkpoint was selected by best validation or
test score.

== Recorded evidence
<recorded-evidence>
```json
{
  "status": "step_limit",
  "dataset": "mnist",
  "family": "onn",
  "seed": 101,
  "step": 100,
  "train_seconds": 33.96222671099531,
  "parameters": 1289488,
  "device": "cuda",
  "initial_validation_mse": 2.111804723739624,
  "final_validation_mse": 2.0086912736296654,
  "images_per_second": 376.88930437108195,
  "source_revisions": [
    "a3dd6228d6e9296a8555102d0b854b53e4a547aa"
  ],
  "peak_memory_bytes": 203779584,
  "checkpoint_sha256": "8f38d4c79f09a651583cc67b6596ab438a75269ad8fcb6a89db4135257a2a9db",
  "failure": null,
  "id": "mnist-smoke",
  "artifact": "runs/checks/mnist-smoke/training.json"
}
```

== Exact configuration
<exact-configuration>
```json
{
  "model": {
    "family": "onn",
    "channels": 1,
    "size": 28,
    "width": 128,
    "depth": 4,
    "patch": 4,
    "rotating_dim": 4,
    "inner_steps": 4,
    "step_size": 0.1,
    "ffn_hidden": 0
  },
  "dataset": "mnist",
  "seed": 101,
  "device": "cuda",
  "batch_size": 128,
  "microbatch": 32,
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "clip_grad": 1.0,
  "ema_decay": 0.999,
  "minutes": 3.0,
  "max_steps": 100,
  "validation_size": 5000,
  "data_root": "data",
  "runs_root": "runs",
  "budget_hours": 24.0,
  "log_every": 50,
  "checkpoint_seconds": 300.0,
  "amp": true
}
```

== Method choices and learning
<method-choices-and-learning>
The shared optimizer, data policy, architecture, and stopping rule were
chosen before final comparison and are described in the technical paper.
The 3e-4 learning rate is a conservative AdamW starting point, not a
tuned optimum. Microbatches bound laptop memory; four internal updates
bound recurrent cost; EMA smooths noisy parameter updates. These choices
trade study breadth for a fixed local compute budget. This record
supports only the measurements above; causes of quality differences
require the trained controls and uncertainty analysis. Full
trajectories, populations, RNG states, and logs remain in the referenced
artifact bundle. Sources: #link("../../docs/protocol.md")[protocol],
#link("https://github.com/atong01/conditional-flow-matching")[TorchCFM],
#link("https://arxiv.org/html/2410.13821v2")[AKOrN].
