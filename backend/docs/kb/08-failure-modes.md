# Failure Modes and Operational Behaviour

## Missing model weights at startup
The classifier weights are expected at backend/weights/best.pt and are not
tracked in version control. If the weights file is absent, the service fails at
startup rather than accepting requests it cannot serve. The fix is to download the
trained weights from the training run and place them at that path.

## Missing knowledge base index
The chat assistant needs the prebuilt index at backend/data/kb_index.npz. If the
index has not been generated, chat requests fail with a service unavailable
response instead of returning empty answers. The fix is to run the build_kb tool,
which reads every Markdown file under backend/docs/kb/ and writes the index.

## Stale knowledge base after editing documents
Editing or adding a Markdown document has no effect on answers until the index is
rebuilt, because retrieval reads the prebuilt vectors rather than the files. A
symptom of a stale index is a question about newly written documentation coming
back as not grounded.

## Unsupported or corrupt upload
A file with an unsupported extension is rejected before inference. A file with a
valid extension but corrupt contents fails at decode time and is reported as a
bad request, not as a model error. A video from which no valid frame can be
sampled produces no aggregated result and is reported as an invalid input.

## Rejected requests versus failed requests
An HTTP 401 means the API key was missing or invalid and no model work was
attempted. A 4xx error generally means the input was unacceptable. A 5xx error
means the input was accepted but the server could not complete the work, for
example missing weights or a missing knowledge base index. Distinguishing the two
is the first step when triaging a demo failure.

## Not grounded is not a failure
A not grounded chat answer means retrieval ran successfully but no section
reached the similarity threshold. The service is healthy; the knowledge base
simply does not cover the question. The remedy is to add documentation, not to
restart the service.

## Questions the knowledge base cannot answer by design
The knowledge base contains documentation, not runtime state. Questions about
what was detected most recently, how many items passed the conveyor today, or the
contents of the results database cannot be answered from documentation and will
come back as not grounded. Live operational data is served by the history and
prediction endpoints instead.
