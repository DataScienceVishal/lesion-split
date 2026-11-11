# Words this project does not use in prose

Enforced by `scripts/check_fingerprint.py`, which parses the fenced blocks below.
The pre-commit hook runs it over the staged diff and CI runs it over the tree.

## Single words

```banned-words
delve
delves
delved
delving
leverage
leverages
leveraging
leveraged
robust
robustly
seamless
seamlessly
comprehensive
comprehensively
cutting-edge
harness
harnesses
harnessed
harnessing
unlock
unlocks
unlocked
unlocking
elevate
elevates
elevated
elevating
streamline
streamlines
streamlined
streamlining
empower
empowers
empowered
empowering
showcase
showcases
showcased
showcasing
boast
boasts
boasting
underscore
underscores
underscored
underscoring
pivotal
meticulous
meticulously
realm
realms
landscape
landscapes
tapestry
tapestries
intricate
intricately
intricacies
interplay
multifaceted
nuanced
noteworthy
commendable
invaluable
unparalleled
groundbreaking
transformative
burgeoning
uncharted
encompass
encompasses
encompassed
encompassing
elucidate
elucidates
elucidating
bolster
bolsters
bolstered
bolstering
garner
garners
garnered
garnering
foster
fosters
fostered
fostering
unveil
unveils
unveiled
unveiling
unraveling
crucial
additionally
```

## Phrases

```banned-phrases
at its core
it is worth noting
it's worth noting
a testament to
navigating the complexities
in today's fast-paced
let us dive in
let's dive in
in conclusion
the key takeaway
dive deep
deep dive
game changer
game-changing
best-in-class
state-of-the-art
paradigm shift
worth your time
thank me later
bookmark this
save this for later
don't sleep on
you won't want to miss
do yourself a favor
i keep coming back to
i can't stop thinking about
rattling around in my head
what surprised me most
what struck me
hits different
hit differently
hot take
fun fact
pro tip
unpopular opinion
plot twist
chef's kiss
let me know in the comments
drop a comment
follow for more
here's the thing
here's the kicker
let that sink in
make no mistake
it turns out
the truth is
the reality is
at the end of the day
when it comes to
in a world where
more important than ever
has never been more
cannot be overstated
at an inflection point
changed the game
that changed everything
could potentially
may potentially
might possibly
could possibly
serves as a
stands as a
what most people get wrong
the part everyone misses
what nobody tells you
the part most people skip
as you can see
the key point is
this distinction matters
matters more than it sounds
this is huge
this changes everything
```

## Patterns

```banned-regex
not only .{1,60}? but also
it is not just .{1,60}?, it is
it's not just .{1,60}?, it's
whether you (are|'re) .{1,60}? or
in the (ever[- ]evolving|rapidly changing) .{0,30}(world|landscape|realm)
(n't|not)( think it)? means? what it (looks|seems|sounds) like
,\s+(?:highlighting|underscoring|emphasi[sz]ing|reflecting|symbolizing|reinforcing|solidifying|cementing|cultivating|contributing to|resonating with)\b
\bThe (?:catch|kicker|twist|result|outcome|answer|solution|problem|reason|reality|truth|worst part|best part|takeaway|lesson|upside|downside|difference|bottom line)\?\s+[A-Z]
\bThe (?:catch|kicker|upshot|trade-?off|trick|bottom line|takeaway|moral|payoff|flip side):
```

## Exceptions

```allowed-contexts
eval harness
evaluation harness
test harness
agent harness
harness/
harness.py
_harness
underscore-separated
leading underscore
trailing underscore
double underscore
financial leverage
```
