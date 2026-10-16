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


= Evaluation: mnist-smoke/validation-check
<evaluation-mnist-smokevalidation-check>
== Purpose and interpretation
<purpose-and-interpretation>
This evaluation measures a fixed checkpoint and fixed noise population.
KID subset SD describes metric subsampling variability; it is not a
confidence interval across trained seeds. The first 64 images are shown
without selection.

== Recorded evidence
<recorded-evidence>
```json
{
  "dataset": "mnist",
  "family": "onn",
  "training_seed": 101,
  "split": "validation",
  "intervention": "none",
  "generated_count": 128,
  "real_count": 5000,
  "sampling_seed": 20261016,
  "metric_seed": 20261016,
  "nfe": 32,
  "kid_subsets": 100,
  "kid_subset_size": 128,
  "checkpoint_sha256": "8f38d4c79f09a651583cc67b6596ab438a75269ad8fcb6a89db4135257a2a9db",
  "data_sha256": "39da09b6ccbbfae8c6424de59b611882eba7f0fb590999ef1851e80f2d027b1c",
  "source_revision": "5b69c903ddac223b5d44e6967dcd49ebd4e136cc",
  "primary_eligible": false,
  "device": "cuda",
  "status": "completed",
  "sampling": {
    "count": 128,
    "seed": 20261016,
    "nfe": 32,
    "batch_size": 32,
    "seconds": 2.681424114998663,
    "seconds_per_image": 0.020948625898427053,
    "precision": "float32",
    "timing_excludes": "model load and warmup",
    "clipped_fraction": 0.3916613459587097,
    "peak_memory_bytes": 55247872
  },
  "metrics": {
    "pixel_sliced_wasserstein": 1.1010091948096694,
    "projections": 128
  },
  "validation_velocity_mse": 2.0086912736296654,
  "metric_seconds": 0.24426259099709569,
  "id": "mnist-smoke/validation-check",
  "artifact": "runs/checks/mnist-smoke/validation-check/evaluation.json"
}
```

== Fixed sample grid
<fixed-sample-grid>
#figure(image("../figures/evaluation-mnist-smoke--validation-check.png"),
  caption: [
    First 64 generated samples
  ]
)

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
