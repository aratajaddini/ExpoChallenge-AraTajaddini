# API

## Endpoints
The FastAPI service groups its routes into five routers. The prediction router
accepts image and video uploads and returns classification results. The history
router returns previously stored detections. The feedback router records user
corrections on a prediction. The admin router under /admin/keys creates, lists,
and revokes API keys. The chat router under /chat answers questions about the
project from the knowledge base.

## Authentication
Every protected endpoint requires an API key sent in the X-API-Key request
header. Keys are never stored in plaintext and never committed to the
repository. A missing, unknown, expired, or revoked key returns HTTP 401 with
the detail message "Invalid, expired or revoked API key." and the response
header WWW-Authenticate set to ApiKey.

The API key dependency is resolved before the request body is validated against
its schema. A request that is both unauthenticated and carries a schema-invalid
body therefore returns 401, not 422. The one exception is a body that is not
syntactically valid JSON: that fails during body parsing and returns 422 before
authentication runs.

## Cross-origin access
Browser access is restricted to an explicit allow-list of origins rather than a
wildcard. The default list contains the local development origins
http://localhost:5173 and http://127.0.0.1:5500 and is overridden through the
ALLOWED_ORIGINS environment variable as a comma-separated list.

## Chat responses
A chat answer is grounded only when retrieval finds a knowledge-base chunk close
enough to the question. When nothing is close enough, the endpoint says it does
not know instead of inventing an answer, and returns no citations. Grounded
answers return the source file and section of every chunk used.
