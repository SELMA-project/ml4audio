# punctuation via sequence tagging -> recovering Casing and `,.?`
* WTF! this exclamation mark `!` is missing !!!
* this repo is work in progress! currently not much more than a first draft!
* everything is based on this [NeMo tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Punctuation_and_Capitalization.ipynb) and [NeMo punctuation_capitalization_train_evaluate](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py)

### some motivating examples
```commandline
query='von der farbetonbeziehung zur farblichtmusik in jörg jewanski natalie sidler hrsg farbe lichtmusik synästhesie und farblichtmusik s 131–209 friedemann kawohl alexander moser – chemiker künstlerfotograf und konstrukteur eines lichtklaviers für alexander skrjabins prométhée in schriften des vereins für geschichte und naturgeschichte der baar 56 2013 71–90 josefhorst lederer die funktion der lucestimme in skrjabins op graz 1980 universal edition studien'
prediction=['von der Farbetonbeziehung zur Farblichtmusik. In Jörg Jewanski Natalie Sidler Hrsg Farbe Lichtmusik. Synästhesie und Farblichtmusik. S. 131–209. Friedemann Kawohl Alexander Moser – Chemiker, Künstlerfotograf und Konstrukteur eines Lichtklaviers für Alexander Skrjabins Prométhée. In Schriften des Vereins für Geschichte und Naturgeschichte der Baar 56, 2013, 71–90. Josefhorst Lederer Die Funktion der Lucestimme in Skrjabins op. Graz 1980. Universal Edition Studien']
original='Von der Farbe-Ton-Beziehung zur Farblichtmusik. In: Jörg Jewanski Natalie Sidler (Hrsg.): Farbe- Licht-Musik: Synästhesie und Farblichtmusik (S. 131–209). Friedemann Kawohl: Alexander Moser – Chemiker, Künstlerfotograf und Konstrukteur eines Lichtklaviers für Alexander Skrjabins Prométhée. In: Schriften des Vereins für Geschichte und Naturgeschichte der Baar 56 (2013), 71–90. Josef-Horst Lederer: Die Funktion der Luce-Stimme in Skrjabins op. Graz 1980, Universal Edition Studien'

query='zur wertungsforschung prometheus von skrjabin in wassily kandinsky franz marc hrsg 19122004 der blaue reiter münchen piper 19122004 s 107–124 sigfried schibli alexander skrjabin und seine musik piper münchenzürich 1983 isbn 3492027598 horst weber zur geschichte der synästhesie oder von den schwierigkeiten die lucestimme in prometheus zu interpretieren graz 1980 universal edition studien zur wertungsforschung sebastian widmaier skrjabin und prometheus'
prediction=['zur Wertungsforschung. Prometheus von Skrjabin. In Wassily Kandinsky. Franz Marc Hrsg 19122004 Der Blaue Reiter. München, Piper 19122004, S. 107–124. Sigfried Schibli Alexander Skrjabin und seine Musik. Piper, Münchenzürich 1983, Isbn 3492027598. Horst Weber Zur Geschichte der Synästhesie oder von den Schwierigkeiten, die Lucestimme in Prometheus zu interpretieren. Graz 1980. Universal Edition Studien zur Wertungsforschung. Sebastian Widmaier Skrjabin und Prometheus.']
original='zur Wertungsforschung; Prometheus von Skrjabin. In: Wassily Kandinsky Franz Marc (Hrsg.) (1912/2004): Der blaue Reiter. München, Piper, 1912/2004, S. 107–124 Sigfried Schibli: Alexander Skrjabin und seine Musik. Piper, München/Z
ürich 1983, ISBN 3-492-02759-8 Horst Weber: Zur Geschichte der Synästhesie. Oder: Von den Schwierigkeiten, die Luce-Stimme in Prometheus zu interpretieren. Graz 1980, Universal Edition Studien zur Wertungsforschung; Sebastian Widmaier: Skrjabin und Prometheus.'

query='hankeverlag weingarten 1986 nienbergen ist ein ortsteil des fleckens bergen an der dumme im landkreis lüchowdannenberg in niedersachsen das rundlingsdorf liegt 3 km südwestlich vom kernort bergen an der dumme und direkt an der östlich verlaufenden grenze zu sachsenanhalt nienbergen liegt an der bahnstrecke stendal–uelzen nördlich in 45 km entfernung liegt das 480 hektar große naturschutzgebiet schnegaer mühlenbachtal nienbergen hieß'
prediction=['Hankeverlag Weingarten 1986, Nienbergen ist ein Ortsteil des Fleckens Bergen an der Dumme im Landkreis Lüchowdannenberg in Niedersachsen. Das Rundlingsdorf liegt 3 km südwestlich vom Kernort Bergen an der Dumme und direkt an der östlich verlaufenden Grenze zu Sachsenanhalt. Nienbergen liegt an der Bahnstrecke Stendal–uelzen Nördlich. In 45 km Entfernung liegt das 480 Hektar große Naturschutzgebiet Schnegaer Mühlenbachtal. Nienbergen hieß']
original='Hanke-Verlag, Weingarten 1986. Nienbergen ist ein Ortsteil des Fleckens Bergen an der Dumme im Landkreis Lüchow-Dannenberg in Niedersachsen. Das Rundlingsdorf liegt 3 km südwestlich vom Kernort Bergen an der Dumme und direkt an der östlich verlaufenden Grenze zu Sachsen-Anhalt. Nienbergen liegt an der Bahnstrecke Stendal–Uelzen. Nördlich in 4,5 km Entfernung liegt das 480 Hektar große Naturschutzgebiet Schnegaer Mühlenbachtal. Nienbergen hieß'
```

# very short description
* punctuation+capitalization via 2-headed sequence-tagging (heads on top of `bert-base-multilingual-uncased`)
* finetuned on wikipedia-data from [Helsinki-NLP/Tatoeba-Challenge](https://raw.githubusercontent.com/Helsinki-NLP/Tatoeba-Challenge/master/data/MonolingualData.md)

# TODO
* more train-data
* proper text-formatting/normalization
* finetuning mono-lingual BERTs might be better (for languages that have a pretrained BERT)

# open questions concerning serving
* what do you prefer?
  1. bake model into docker-image
  2. service-docker-image without models that expects model upload (via post-request)?
* should the service itself detect languages and pull language-specific models from somewhere?