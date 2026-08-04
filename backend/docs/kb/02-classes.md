# Waste Classes

## The five categories
The classifier assigns every item to one of five waste categories: plastic,
metal, paper, glass, and organic. These names are read from the trained model at
load time through the model's own label mapping, so they are never hard-coded in
the source. If the model is retrained with a different label set, the API, the
analytics counters, and the frontend legend all follow automatically without a
code change.

## Plastic and metal
Plastic covers bottles, cups, wrappers, bags, and moulded containers. Metal
covers beverage cans, food tins, foil, and small metal parts. These two classes
are the most common source of confusion in practice, because a crushed can and a
crushed plastic bottle share a similar silhouette under strong overhead light.
Adding more training images of deformed items is the usual fix.

## Paper and glass
Paper covers cardboard, newspaper, office paper, and paper packaging. Glass
covers bottles and jars, whether clear, green, or brown. Glass is difficult
because it is transparent: the belt surface shows through the item, so the model
partly learns the background instead of the object. Diffuse lighting and a
matte belt surface reduce this effect.

## Organic
Organic covers food waste such as peels, cores, shells, and leftovers. It is
the class with the highest visual variety, so it benefits most from additional
training data.

## Uncertain results
A prediction is not forced into a class when the model is unsure. Any frame
whose confidence falls below 0.35 is counted as uncertain instead of being
assigned to the nearest category. Uncertain frames are reported separately so
that a low-quality clip is visible as low confidence rather than as a wrong
class.
