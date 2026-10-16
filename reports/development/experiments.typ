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


= Experiment ledger and glossary
<experiment-ledger-and-glossary>
Primary comparison incomplete; no claim about an oscillator advantage is
supported.

All networks generate unconditional images in pixel space. They share
independent conditional flow matching, AdamW, continuous time
conditioning, 4×4 patches, and an EMA sampling checkpoint. MNIST is
development data; CIFAR-10 is the final comparison domain. There is no
curriculum or augmentation. Classification accuracy, text fill rate, and
task-completion accuracy are not applicable; run completion is reported
separately from image quality.

#figure(
  align(center)[#table(
    columns: (25%, 25%, 25%, 25%),
    align: (auto,auto,auto,auto,),
    table.header([Network], [Feature computation], [Training
      objective], [Role],),
    table.hline(),
    [onn], [Coupled 4D unit-vector rotations; four recurrent
    blocks], [Velocity MSE], [Proposed generator],
    [dit], [Four time-conditioned attention/MLP blocks], [Velocity
    MSE], [Parameter-matched transformer],
    [conv], [Convolution + SiLU + GroupNorm recurrence], [Velocity
    MSE], [Conventional recurrent control],
    [no\_coupling], [Oscillators with coupling convolution
    removed], [Velocity MSE], [Trained removal; lower capacity],
    [no\_frequency], [Oscillators with natural-frequency term
    removed], [Velocity MSE], [Trained removal; slightly lower
    capacity],
  )]
  , kind: table
  )

== Training outcomes
<training-outcomes>
#figure(
  align(center)[#table(
    columns: (14.29%, 14.29%, 14.29%, 14.29%, 14.29%, 14.29%, 14.29%),
    align: (auto,auto,auto,auto,auto,auto,auto,),
    table.header([Run], [Status], [Parameters], [Updates], [Train
      seconds], [Images/s], [Validation MSE],),
    table.hline(),
    [mnist-smoke], [step\_limit], [1.28949e+06], [100], [33.9622], [376.889], [2.00869],
  )]
  , kind: table
  )

== Distribution and cost measurements
<distribution-and-cost-measurements>
#figure(
  align(center)[#table(
    columns: (12.5%, 12.5%, 12.5%, 12.5%, 12.5%, 12.5%, 12.5%, 12.5%),
    align: (auto,auto,auto,auto,auto,auto,auto,auto,),
    table.header([Evaluation], [Generated/real N], [NFE], [KID
      mean], [KID subset SD], [FID], [Pixel SWD], [ms/image],),
    table.hline(),
    [mnist-smoke/validation-check], [128/5000], [32], [---], [---], [---], [1.10101], [20.9486],
  )]
  , kind: table
  )

== Completeness
<completeness>
Recorded diagnostics: 0. Recorded failed attempts: 0. Completed primary
pairs: 0/6. Missing values mean unmeasured, not zero. Individual records
and PDFs are in `experiments/`\; raw records remain in the run
directories and published artifact manifests.

== Glossary
<glossary>
#figure(
  align(center)[#table(
    columns: (50%, 50%),
    align: (auto,auto,),
    table.header([Term], [Meaning],),
    table.hline(),
    [Flow matching], [Learn a velocity that transports Gaussian noise
    toward the image distribution.],
    [Oscillator], [A hidden vector whose direction changes under natural
    rotation and learned interactions.],
    [Coupling], [A learned convolution that lets hidden oscillators
    influence one another.],
    [Flow time / internal time], [Noise-to-image position / recurrent
    computation inside one velocity prediction.],
    [NFE], [Number of velocity-network evaluations used to generate each
    image.],
    [KID], [Unbiased squared kernel discrepancy between fixed Inception
    features; smaller is better and estimates may be negative.],
    [KID subset SD], [Variation across random metric subsets of one
    fixed pair of image populations, not across trained models.],
    [FID], [Gaussian feature-distribution distance; finite-sample
    biased, so sample count must accompany it.],
    [Pixel SWD], [Wasserstein distance averaged over random pixel-space
    projections; a MNIST diagnostic, not semantic accuracy.],
    [Seed pair], [Two independently initialized architectures using
    paired data/noise/time stream prefixes.],
    [Confidence interval], [A model-based repeated-study procedure; here
    an approximate t interval across six seed differences.],
    [Sign-flip test], [Enumerate all signs of paired differences; exact
    only under the specified exchangeability/symmetry null.],
    [EMA], [Exponential moving average of trainable weights, used for
    all reported samples.],
    [Ablation / intervention], [Retrain with a mechanism removed /
    perturb a trained model during inference.],
    [Power], [Probability of detecting an assumed effect under a
    planning model; six pairs have about 24% power at effect 0.8.],
  )]
  , kind: table
  )
