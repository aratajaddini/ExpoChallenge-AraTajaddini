# Chat and Retrieval Architecture

## How the chat assistant works
The chat assistant is a retrieval system, not a generative language model. It does
not write new sentences. When a question arrives at the chat endpoint, the system
searches the local knowledge base, selects the closest matching passages, and
returns those passages verbatim with a citation for each one. This is called
extractive question answering. Because every sentence in the answer comes from a
source document, the assistant cannot hallucinate facts about the smart waste
robot. If the knowledge base does not cover a question, the assistant says so
instead of guessing.

## Knowledge base source documents
The knowledge base is a set of Markdown files stored under backend/docs/kb/.
Each Markdown heading becomes a separate retrievable section. Editing a Markdown
file does not change the assistant's answers until the index is rebuilt with the
build_kb tool. The index is written to backend/data/kb_index.npz and contains the
embedding vectors, the chunk texts with their source file and section name, and
the name of the embedding model used to build it.

## Chunking and overlap
Documents are split into chunks of about 220 words with an overlap of 40 words
between consecutive chunks. The overlap prevents a fact from being lost when it
falls across a chunk boundary. Each chunk keeps its source file name and its
section heading, which is what makes citations possible.

## Hybrid retrieval with BM25 and embeddings
Retrieval runs two independent searches over every chunk. The first is BM25, a
lexical keyword ranking that rewards exact term matches and is strong for names
like plastic, conveyor, or API key. The second is semantic cosine similarity
between sentence embeddings, which matches meaning even when the wording differs.
The two ranked lists are merged with Reciprocal Rank Fusion, which scores each
chunk by its rank position in both lists rather than by raw score, so the two
scales never need to be normalised. Hybrid retrieval is more robust than either
method alone: BM25 handles rare exact terms, embeddings handle paraphrases.

## Grounding threshold
A result is only returned if its cosine similarity reaches the configured minimum
of 0.35. If the best match falls below this floor, the assistant returns a no
answer response with an empty citation list and marks the reply as not grounded.
The threshold is deliberate: a low similarity score means the knowledge base does
not really cover the question, and returning an unrelated passage would be worse
than admitting the gap. At most three citations are returned per answer, sorted
by descending similarity.

## Reading the grounded badge
Each answer in the interface carries a badge. Grounded means the answer text was
taken from knowledge base sections and at least one citation is attached. Not
grounded means no section scored above the similarity threshold and no citation
is available. The badge is a transparency signal, not an error: it tells the
operator whether the answer can be traced back to a source document.
