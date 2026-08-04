# Frequently Asked Questions

## Which waste classes can the system recognise
The model recognises five waste classes: plastic, metal, paper, glass, and
organic. The class names are read from the trained model's own label mapping at
load time and are never hard-coded in the application, so a retrained model with
different labels is picked up automatically.

## Why does the system sometimes say uncertain instead of a class
Every prediction carries a confidence score between zero and one. When the top
confidence falls below the threshold of 0.35, the result is reported as uncertain
rather than forced into a class. On a real conveyor belt a wrong confident answer
is more expensive than an admitted uncertainty, because a misrouted item
contaminates a whole batch and has to be sorted again by hand.

## What does the confidence score actually mean
The confidence score is the model's relative certainty across the five classes,
not a probability that the answer is correct in the physical world. A score of
0.9 means the input strongly resembles the training examples of that class. It
does not guarantee correctness on an object type the model has never seen.

## Which classes are hardest to classify
Glass is the hardest because it is transparent, so the camera mostly sees the
background through the object, and specular highlights change with the lighting.
Organic waste is second hardest because it has no stable shape or colour and
needs a broader range of training data. Plastic and metal can also be confused
when a plastic item has a reflective metallic coating.

## Does the system need a GPU to run
No. Inference runs on CPU. The model is loaded once and cached in memory, so the
first request pays the load cost and later requests reuse the same instance.

## What are the upload limits
A single upload may not exceed 150 MB. For video, at most 120 frames are sampled
rather than every frame, which keeps CPU inference time bounded and predictable
regardless of clip length.

## How are video results turned into one answer
Sampled frames are classified individually and then aggregated into a summary for
the whole clip, rather than reported frame by frame. This smooths out single bad
frames caused by motion blur or partial occlusion.

## Why is an API key required
Every request must carry a valid key in the X-API-Key header. A missing or
invalid key is rejected with HTTP 401 before any model work happens. Keys are
never stored in plaintext. The service also refuses to start if the required key
configuration or the model weights file is absent, so a broken deployment fails
immediately at startup instead of silently serving errors later.

## Where is the model trained and why not in this repository
Training runs in Google Colab because the development machines have no local GPU.
The training notebooks and the dataset live outside the application repository,
and only the resulting weights file is deployed. This keeps the runtime
repository small and keeps large binary artifacts out of version control.

## Is the chatbot a large language model
No. It is an extractive retrieval system over the project's own documentation. It
returns passages from the knowledge base with citations and reports when it has no
grounded source, so it cannot invent claims about the system.
