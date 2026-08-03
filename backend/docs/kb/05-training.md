# Training

## Where training happens
Training runs on Google Colab because the development machine has no GPU. The
dataset and the training notebook live outside the application repository, and
only the resulting weights file is copied back into backend/weights/best.pt.
The repository therefore contains no dataset, no checkpoints, and no training
logs.

## Dataset layout
The dataset follows the Ultralytics classification layout: one directory per
split, and inside each split one directory per class whose name is the class
label. The class names in those directory names become the label mapping stored
inside the checkpoint, which is exactly what the API later reads back. Renaming
a class directory and retraining is the supported way to change the label set.

## Producing new weights
A training run produces a best checkpoint and a last checkpoint. Only the best
checkpoint is used in production. Before replacing the existing weights, the
checkpoint is inspected to confirm the number of classes and the label order,
because a silent label-order change would make every stored detection
inconsistent with older rows.

## Evaluating a run
A run is judged on per-class accuracy, not only on overall accuracy, because
the five classes are not equally represented. Glass and organic are the classes
to watch, since transparency and visual variety make them the weakest. Whether
the confidence floor of 0.35 is still appropriate is rechecked after each
retraining.
