"""Canned replies for conversational turns the knowledge base cannot cover.

Only non-documentation intents belong here: greetings, meta questions about
the assistant, and graceful refusals. Anything answerable from backend/docs/kb/
must fall through to retrieval so the reply keeps its citations.
"""

from __future__ import annotations

import re
from typing import Final

_ASK_HINT: Final = (
    "Try: \"Which waste classes can the system recognise?\", "
    "\"What does the confidence score mean?\", or \"Why does it say uncertain?\""
)

_SCOPE: Final = (
    "I only answer from this project's local documentation, so I can't help "
    "with that. " + _ASK_HINT
)


def _norm(text: str) -> str:
    """Lowercase, drop apostrophes and punctuation, collapse whitespace."""
    lowered = text.lower().replace("\u2019", "").replace("'", "").replace("`", "")
    cleaned = "".join(c if (c.isalnum() or c.isspace()) else " " for c in lowered)
    return " ".join(cleaned.split())


# (aliases, reply) — aliases are normalised at import time.
_GROUPS: Final[tuple[tuple[tuple[str, ...], str], ...]] = (
    # --- greetings ---
    (
        ("hello", "hi", "hey", "yo", "hiya", "hi there", "hello there",
         "hey there", "hello bot", "hi bot", "howdy", "greetings",
         "good morning", "good afternoon", "good evening", "good day",
         "morning", "evening", "hallo", "guten tag", "guten morgen",
         "hola", "bonjour", "ciao", "salut", "hej", "salam", "salaam",
         "سلام"),
        "Hello. I'm the documentation assistant for smart-waste-robot. I answer "
        "from the project's local docs and cite the exact section I used. "
        + _ASK_HINT,
    ),
    # --- presence / liveness checks ---
    (
        ("test", "testing", "ping", "are you there", "are you online",
         "are you working", "do you work", "you there", "anyone there",
         "is this working", "does this work", "hello are you there"),
        "I'm online and the knowledge base is loaded. Ask a question about the "
        "system and I'll answer with its source. " + _ASK_HINT,
    ),
    # --- how are you ---
    (
        ("how are you", "how are you doing", "how is it going", "hows it going",
         "how do you do", "you ok", "are you ok", "whats up", "sup",
         "wie gehts", "wie geht es dir"),
        "Running fine. Ask me anything documented about the waste classes, the "
        "detection pipeline, the API, or the training setup.",
    ),
    # --- identity ---
    (
        ("who are you", "what are you", "whats your name", "what is your name",
         "your name", "who am i talking to", "what should i call you",
         "introduce yourself", "tell me about yourself", "who is this",
         "are you a bot", "are you a robot", "are you human", "are you a person",
         "are you real"),
        "I'm the documentation assistant for smart-waste-robot. I search the "
        "project's local Markdown docs, return the matching passages, and show "
        "which file and section each answer came from.",
    ),
    # --- capabilities ---
    (
        ("what can you do", "what do you do", "help", "help me",
         "can you help", "can you help me", "what can i ask",
         "what can i ask you", "what should i ask", "what should i ask you",
         "what questions can you answer", "what do you know",
         "what topics do you cover", "how do i use this", "how does this work",
         "how do i use you", "commands", "options", "menu", "start"),
        "I answer documented questions about smart-waste-robot: the waste "
        "classes, confidence and uncertainty, the detection pipeline, the API "
        "endpoints, API keys, failure modes, and how the model was trained. "
        + _ASK_HINT,
    ),
    # --- who built it ---
    (
        ("who made you", "who built you", "who created you", "who developed you",
         "who wrote this", "who is behind this", "who made this",
         "who built this", "who is the developer", "whos the developer",
         "who is the team", "team", "your creator", "your author"),
        "smart-waste-robot is built by a dedicated team of engineers working "
        "on AI, documentation, and support. The project was developed as part "
        "of an AI challenge to make recycling smarter and more efficient.",
    ),
    # --- languages ---
    (
        ("what languages do you speak", "which languages do you support",
         "do you speak german", "do you speak spanish", "do you speak french",
         "sprichst du deutsch", "parlez vous francais", "hablas espanol",
         "can you speak german", "can you speak spanish", "can you speak french"),
        "The knowledge base is written in English, so my answers are in English. "
        "You can ask in another language, but the retrieved passages will still "
        "be English.",
    ),
    # --- thanks ---
    (
        ("thanks", "thank you", "thanks a lot", "thank you very much",
         "many thanks", "thx", "ty", "tysm", "cheers", "appreciate it",
         "much appreciated", "nice one", "danke", "danke schon", "vielen dank",
         "merci", "gracias", "ممنون"),
        "You're welcome. Ask away if you need anything else from the docs.",
    ),
    # --- acknowledgements ---
    (
        ("ok", "okay", "k", "kk", "cool", "nice", "great", "good", "alright",
         "got it", "understood", "i see", "makes sense", "fine", "sure",
         "perfect", "awesome", "yes", "yep", "yeah", "no", "nope", "nah",
         "ok thanks", "ok thank you"),
        "Noted. " + _ASK_HINT,
    ),
    # --- goodbye ---
    (
        ("bye", "goodbye", "good bye", "see you", "see ya", "cya", "later",
         "see you later", "take care", "good night", "night", "im done",
         "thats all", "that is all", "ciao", "tschuss", "auf wiedersehen",
         "adios", "au revoir"),
        "Goodbye. The docs stay available whenever you come back.",
    ),
    # --- apologies / confusion ---
    (
        ("sorry", "my bad", "oops", "nevermind", "never mind", "ignore that",
         "forget it", "what", "huh", "pardon", "come again", "say that again",
         "repeat", "repeat that", "i dont understand", "im confused",
         "that makes no sense"),
        "No problem. Rephrase the question and I'll search the docs again. "
        + _ASK_HINT,
    ),
    # --- frustration ---
    (
        ("this doesnt work", "it doesnt work", "youre useless", "you are useless",
         "you are wrong", "youre wrong", "thats wrong", "bad answer",
         "wrong answer", "you dont know anything", "this is broken",
         "not helpful", "that didnt help"),
        "Sorry that missed. I can only return what the docs contain, so a more "
        "specific wording usually helps. You can also check which file should "
        "cover the topic under backend/docs/kb/.",
    ),
    # --- out of scope: general assistant requests ---
    (
        ("write code for me", "write me some code", "can you write code",
         "can you code", "do my homework", "write an essay", "tell me a story",
         "translate this", "summarise the internet", "search the web",
         "google it", "what is python", "explain machine learning",
         "teach me python"),
        _SCOPE,
    ),
    # --- out of scope: real-world facts ---
    (
        ("what time is it", "whats the time", "what is the time",
         "what is todays date", "whats the date", "what day is it",
         "whats the weather", "what is the weather", "how is the weather",
         "where are you", "what is your location", "wie ist das wetter"),
        "I don't have access to the clock, the calendar, or the internet. I only "
        "read this project's documentation.",
    ),
    # --- fun ---
    (
        ("tell me a joke", "say something funny", "are you funny", "sing a song",
         "do you like me", "do you have feelings", "are you alive",
         "are you conscious", "do you dream", "whats your favourite class",
         "whats your favorite class"),
        "Not my department. I sort documentation, the robot sorts waste. "
        + _ASK_HINT,
    ),
    # --- about the project ---
    (
        ("what is smart waste robot", "what is smart-waste-robot",
         "what is this project", "tell me about the project",
         "what does the system do", "what is the purpose",
         "what is the goal", "what is this all about"),
        "smart-waste-robot is an AI-powered waste sorting system. It uses a "
        "camera and a YOLOv8 model to classify waste into categories: "
        "Plastic, Metal, Paper, Glass, and Organic. The classification triggers "
        "a sorting mechanism to separate the waste.",
    ),
    # --- about the problem ---
    (
        ("why is waste management important", "why is recycling important",
         "what is the problem with waste", "why do we need waste sorting",
         "what is the waste problem"),
        "Waste generation has increased dramatically worldwide. Effective "
        "waste sorting and recycling are essential to reduce environmental "
        "pollution, conserve natural resources, and minimise landfill use. "
        "Automated sorting systems like this one help improve efficiency and "
        "reduce the health risks associated with manual sorting.",
    ),
    # --- about the model ---
    (
        ("what model do you use", "which model is used", "what is the model",
         "what is yolo", "what is yolo v8", "what is yolov8",
         "why yolo", "how does the model work", "what is the architecture"),
        "The system uses YOLOv8, a state-of-the-art real-time object detection "
        "model. It is trained on the TrashNet dataset, which contains images "
        "of waste items across several classes. The model achieves fast "
        "inference with high accuracy, making it suitable for real-time sorting.",
    ),
    # --- about the dataset ---
    (
        ("what dataset was used", "which dataset", "what is trashnet",
         "tell me about trashnet", "how many images", "what classes",
         "waste classes"),   # "what are the classes" removed
        "The model is trained on TrashNet, a dataset containing images of "
        "waste items divided into six classes: Plastic, Metal, Paper, Glass, "
        "Cardboard, and Organic/Trash. The dataset is publicly available and "
        "widely used for benchmarking waste classification.",
    ),
    # --- about confidence ---
    (
        ("what does confidence mean", "confidence score", "how is confidence computed",
         "why is confidence low", "what is a good confidence", "confidence threshold"),
        "The confidence score represents the probability the model assigns to "
        "its top prediction. A score of 0.85 means the model is 85% certain. "
        "Scores below the threshold (0.35 by default) are considered uncertain, "
        "and the system may reject the prediction or request manual review.",
    ),
    # --- about the pipeline ---
    (
        ("what is the pipeline", "how does the system work step by step",
         "explain the workflow", "from image to sorting", "what happens after detection"),
        "The pipeline: 1) Capture an image or video frame. 2) Run the YOLOv8 "
        "model to classify the waste. 3) Extract the top class and confidence. "
        "4) Send the result to the sorting mechanism (actuator) to separate "
        "the waste accordingly. The entire process takes under 100ms.",
    ),
    # --- about the API ---
    (
        ("what is the api", "what endpoints are available", "how to call the api",
         "api documentation", "how to use the api", "what is the api key",
         "how to authenticate"),
        "The API provides endpoints for classification (/predict), history "
        "(/history), feedback (/feedback), and chat (/chat). All requests "
        "require an X-API-Key header. Use the admin key (from .env) or a shift "
        "key (minted via the keymaker tool) for authentication.",
    ),
    # --- about API keys ---
    (
        ("how to get an api key", "how to create an api key", "mint a key",
         "shift key", "keymaker", "how to generate a key", "what is a shift key"),
        "API keys can be generated using the tracesort-keygen tool (or the "
        "admin keymaker GUI). You need the admin API key from your .env file "
        "to mint shift keys. Shift keys expire after a set number of hours "
        "and can be revoked at any time.",
    ),
    # --- about history ---
    (
        ("what is history", "how to view history", "how to clear history",
         "history endpoint", "what does history store"),
        "The system keeps a history of all classification results, including "
        "the filename, predicted class, confidence, source (image/video), "
        "and timestamp. You can fetch the last 50 entries via GET /history, "
        "or clear everything with DELETE /history.",
    ),
    # --- about feedback ---
    (
        ("what is feedback", "how to send feedback", "why is feedback important",
         "feedback endpoint", "how to correct a prediction"),
        "Feedback allows you to correct a misclassification. If the model "
        "predicts the wrong class, you can send the correct class via POST "
        "/feedback. This helps improve the model and the training data over time.",
    ),
    # --- about the chat itself ---
    (
        ("how to use this chat", "what is this chat", "how does the chat work",
         "how does this assistant work", "what can i ask here"),
        "This chat uses a RAG (Retrieval-Augmented Generation) system. It "
        "searches the project's documentation (Markdown files in backend/docs/kb/) "
        "for the most relevant passages and presents them as a grounded answer, "
        "complete with citations to the source files.",
    ),
    # --- about the knowledge base ---
    (
        ("what is the knowledge base", "how to update the knowledge base",
         "where are the docs stored", "how to add documentation"),
        "The knowledge base consists of Markdown files placed in backend/docs/kb/. "
        "To update it, add or edit a .md file there, then rebuild the index "
        "with the build_kb tool. The system will then use the new content in "
        "future answers.",
    ),
    # --- about video processing ---
    (
        ("can you process video", "how does video work", "video inference",
         "what about video", "how many frames are sampled"),
        "Yes, you can upload a video instead of an image. The system samples "
        "up to 120 frames (about 1 fps for a 2‑minute clip) and runs inference "
        "on each frame. The final result is the most frequently predicted class "
        "across all frames.",
    ),
    # --- about hardware requirements ---
    (
        ("what hardware is needed", "does it require gpu", "can it run on cpu",
         "what are the system requirements", "how fast is it"),
        "The model can run on CPU, but a GPU is recommended for real‑time "
        "performance. Inference speed is around 50‑100ms per image on a modern "
        "GPU. Video processing takes longer due to the extra frames.",
    ),
    # --- about the optional detector ---
    (
        ("what is the optional detector", "what is the detection model",
         "what other models are used"),
        "There is an optional analytics detector based on the TACO dataset "
        "for detecting larger objects like bins and bags, but it is not used "
        "in the main waste‑sorting pipeline. It remains a separate component.",
    ),
    # --- about the frontend demo ---
    (
        ("how to use the demo", "how does the demo work", "what is the demo",
         "launch demo", "what can i do in the demo"),
        "The frontend demo lets you upload an image or video, set an API key, "
        "and see the classification result with confidence scores and a history "
        "of past predictions. You can also submit feedback to correct errors.",
    ),
    # --- about the team ---
    (
        ("who are the developers", "tell me about the team", "who are the contributors"),
        "The project was built by a team of engineers as part of an AI challenge. "
        "The team includes developers working on AI models, documentation, "
        "and support.",
    ),
    # --- about training and accuracy ---
    (
        ("how accurate is the model", "what is the accuracy", "how well does it perform",
         "what is the precision", "what is the recall"),
        "The model achieves over 90% accuracy on the test set. Performance "
        "varies by class – some materials like glass and plastic can be harder "
        "to distinguish. The system uses a confidence threshold to reduce "
        "false positives.",
    ),
    # --- about error handling ---
    (
        ("what if the model fails", "what if it's uncertain", "what happens on low confidence",
         "what if no class is detected", "how to handle errors"),
        "If the confidence is below the threshold, the system returns an error "
        "message indicating that the prediction is uncertain. You can retry "
        "with a better image, or manually sort the item. The feedback system "
        "can also be used to correct mistakes.",
    ),
    # --- about the sorting mechanism ---
    (
        ("how does sorting work", "what is the actuator", "what happens physically",
         "how does the robot sort"),
        "After classification, the result is sent to a hardware actuator "
        "that physically moves the waste item to the correct bin. The exact "
        "implementation depends on your setup; the software provides the "
        "classification, and the hardware controls the sorting action.",
    ),
    # --- about the developer environment ---
    (
        ("how to set up the project", "how to run the server",
         "how to start the app", "what are the dependencies"),
        "Clone the repository, create a virtual environment, install dependencies "
        "from requirements.txt, set your API key in a .env file, and run "
        "the server. The frontend is served from the /frontend folder. "
        "The server will start on http://127.0.0.1:8000.",
    ),
    # --- about the open-source license ---
    (
        ("is it open source", "what is the license", "can i contribute",
         "how to contribute", "can i fork it"),
        "Yes, the project is open source. You can find the code on GitHub. "
        "Contributions are welcome – feel free to fork, open issues, or submit "
        "pull requests.",
    ),
    # --- about classification output ---
    (
        ("what does the classification result contain", "what info is returned",
         "what fields are in the prediction", "prediction format"),
        "The prediction response includes the top class name, confidence score, "
        "a full list of class scores, inference time (if available), and a "
        "record ID that can be used for feedback.",
    ),
    # --- about scoring mechanism ---
    (
        ("how are scores calculated", "softmax", "probability distribution",
         "why do scores sum to one"),
        "The model outputs raw logits, which are converted to probabilities "
        "using the softmax function. This ensures the scores for all classes "
        "sum to 1.0. The highest score is the predicted class.",
    ),
    # --- about the web interface ---
    (
        ("what is the web interface", "is there a dashboard", "web ui",
         "frontend features"),
        "The frontend provides a dark-themed dashboard where you can upload "
        "images or videos, view classification results with confidence bars, "
        "see a history of predictions, and correct misclassifications via "
        "feedback. It also includes a chat assistant for documentation queries.",
    ),
    # --- about mobile support ---
    (
        ("is there a mobile app", "can i use it on my phone", "mobile support"),
        "There is no dedicated mobile app at the moment. However, the web "
        "interface is responsive and works well on mobile browsers.",
    ),
    # --- about performance metrics ---
    (
        ("what metrics are tracked", "performance metrics", "latency", "throughput"),
        "Key metrics include inference latency (time per prediction), frames "
        "per second for video processing, and accuracy on the test set. "
        "The system also tracks historical predictions and feedback for analysis.",
    ),
    # --- about scaling ---
    (
        ("can it scale", "multi-user support", "concurrent requests", "load handling"),
        "The backend is built with FastAPI, which supports asynchronous "
        "request handling. For heavy loads, you can run multiple workers or "
        "use a production ASGI server.",
    ),
    # --- about deployment ---
    (
        ("how to deploy", "deployment options", "where to host", "production setup"),
        "You can deploy the project on any cloud provider that supports Python "
        "web apps. Use Docker for containerization, and consider using a "
        "reverse proxy and a process manager for production.",
    ),
    # --- about security ---
    (
        ("is it secure", "security measures", "authentication", "authorization"),
        "The API uses header-based authentication with API keys. Admin keys "
        "have full privileges, while shift keys have restricted access. "
        "CORS is configured to allow only trusted origins, and all secrets "
        "are managed via environment variables.",
    ),
    # --- about logging ---
    (
        ("logging", "logs", "log files", "how to see logs"),
        "The application uses Python's logging module. Logs are printed to "
        "the console by default. You can configure logging to write to files "
        "or external services.",
    ),
    # --- about testing ---
    (
        ("testing", "unit tests", "pytest", "how to run tests"),
        "The project includes tests using pytest. You can run them from the "
        "project root. Tests cover the API endpoints, database operations, "
        "and retrieval logic.",
    ),
    # --- about the database ---
    (
        ("what database", "database type", "sqlite", "db file", "storage"),
        "The system uses SQLite as the default database for simplicity. The "
        "database file is in the backend folder. It stores predictions, "
        "feedback, and API keys.",
    ),
    # --- about API versioning ---
    (
        ("api version", "versioning", "deprecation"),
        "Currently, there is no explicit versioning in the API paths. Future "
        "versions may be introduced under a versioned prefix.",
    ),
    # --- about the frontend framework ---
    (
        ("frontend framework", "javascript", "html css", "what is the frontend built with"),
        "The frontend is built with plain HTML, CSS, and JavaScript, with no "
        "heavy frameworks, making it lightweight and easy to modify. It uses "
        "sessionStorage for API key persistence.",
    ),
    # --- about the backend framework ---
    (
        ("backend framework", "fastapi", "why fastapi"),
        "The backend uses FastAPI, a modern Python web framework. It provides "
        "automatic OpenAPI documentation, asynchronous support, and fast "
        "performance, well-suited for machine learning inference.",
    ),
    # --- about architecture overview ---
    (
        ("architecture", "system design", "high-level structure", "components"),
        "The system consists of: a FastAPI backend, a SQLite database, a "
        "YOLOv8 model for classification, a retrieval-augmented chat assistant, "
        "and a static frontend. The backend handles API requests, runs inference, "
        "and manages the database.",
    ),
    # --- about training data details ---
    (
        ("training data details", "how was the data prepared", "data augmentation",
         "preprocessing"),
        "The TrashNet dataset was used with standard augmentation techniques "
        "to improve generalisation. Images were resized to the input size "
        "required by YOLOv8.",
    ),
    # --- about transfer learning ---
    (
        ("transfer learning", "pretrained weights", "fine-tuning", "backbone"),
        "The YOLOv8 model is initialised with pretrained weights and then "
        "fine-tuned on the waste dataset. This leverages features learned "
        "from a large dataset for better performance.",
    ),
    # --- about hyperparameters ---
    (
        ("hyperparameters", "batch size", "learning rate", "epochs", "optimizer"),
        "Typical hyperparameters used: moderate batch size, learning rate "
        "with scheduling, a standard optimizer, and early stopping to avoid "
        "overfitting.",
    ),
    # --- about validation ---
    (
        ("validation", "val set", "train-test split", "cross-validation"),
        "The dataset is split into training and validation sets. The model "
        "is evaluated on the validation set after each epoch. The best checkpoint "
        "is saved based on validation accuracy.",
    ),
    # --- about overfitting and underfitting ---
    (
        ("overfitting", "underfitting", "bias-variance tradeoff"),
        "The model is monitored for overfitting by tracking the gap between "
        "training and validation accuracy. Early stopping and data augmentation "
        "are used to reduce overfitting.",
    ),
    # --- about model checkpoints ---
    (
        ("checkpoints", "saving models", "model persistence"),
        "Model checkpoints are saved automatically during training. The best "
        "checkpoint is stored in the weights folder.",
    ),
    # --- about exporting the model ---
    (
        ("export", "export to ONNX", "export to TensorRT", "model conversion"),
        "You can export the trained model to ONNX or TensorRT formats for "
        "faster inference or edge deployment. The project includes scripts "
        "for conversion.",
    ),
    # --- about quantization ---
    (
        ("quantization", "int8", "fp16", "model compression"),
        "Quantization reduces model size and speeds up inference. You can "
        "quantize the model to FP16 or INT8 using export tools.",
    ),
    # --- about pruning ---
    (
        ("pruning", "sparsity", "model pruning"),
        "Pruning is not currently implemented, but it could be added to reduce "
        "model size further without significant accuracy loss.",
    ),
    # --- about edge deployment ---
    (
        ("edge deployment", "raspberry pi", "jetson nano", "embedded devices"),
        "The model can be deployed on edge devices like NVIDIA Jetson Nano "
        "or Raspberry Pi with a Coral TPU. Optimised formats are recommended "
        "for such platforms.",
    ),
    # --- about the camera ---
    (
        ("camera", "webcam", "usb camera", "camera specifications"),
        "The system can work with any USB camera or built-in webcam. For "
        "industrial settings, higher-resolution cameras with good lighting "
        "are recommended for accurate classification.",
    ),
    # --- about sorting mechanism details ---
    (
        ("sorting mechanism", "actuator details", "servo motor", "conveyor"),
        "The sorting mechanism typically consists of a servo motor or solenoid "
        "that activates a flap or pushes the item into the correct bin. The "
        "actuator receives the classification result via a serial or GPIO "
        "interface.",
    ),
    # --- about bin arrangement ---
    (
        ("bins", "bin layout", "how many bins", "bin positions"),
        "By default, there are bins corresponding to the waste classes. The "
        "system can be customised to support different bin configurations.",
    ),
    # --- about feedback loop for continuous learning ---
    (
        ("feedback loop", "continuous learning", "retraining", "online learning"),
        "Feedback data is stored and can be used to retrain the model offline "
        "to improve accuracy over time. Online learning is not implemented "
        "but could be added.",
    ),
    # --- about future improvements ---
    (
        ("future plans", "roadmap", "what's next"),
        "Future improvements include support for more classes, better video "
        "processing, integration with industrial sorting systems, and an "
        "enhanced knowledge base.",
    ),
    # --- about contributing guidelines ---
    (
        ("contributing guidelines", "how to contribute", "coding standards"),
        "Contributions are welcome. Please follow the existing code style, "
        "write tests for new features, and update the documentation.",
    ),
    # --- about code of conduct ---
    (
        ("code of conduct", "behaviour", "community guidelines"),
        "The project adheres to a code of conduct. We expect all contributors "
        "to be respectful and inclusive.",
    ),
    # --- about issue tracking ---
    (
        ("issues", "bug reports", "feature requests", "tracking"),
        "Bugs and feature requests are tracked on GitHub Issues. Please check "
        "existing issues before opening a new one.",
    ),
    # --- about pull requests ---
    (
        ("pull requests", "pr process", "review process"),
        "Pull requests should be small and focused. They must pass all tests "
        "and be reviewed before merging.",
    ),
    # --- about documentation style ---
    (
        ("documentation style", "docstrings", "markdown"),
        "Code should be documented with docstrings. The knowledge base uses "
        "Markdown files with clear section headings.",
    ),
    # --- about CI/CD ---
    (
        ("CI/CD", "continuous integration", "continuous deployment", "github actions"),
        "The project uses GitHub Actions for CI/CD. On every push, tests are "
        "run and code quality is checked.",
    ),
    # --- about Docker ---
    (
        ("docker", "containerization", "dockerfile"),
        "A Dockerfile is provided to containerise the application for easy "
        "deployment. You can build and run the image with standard Docker "
        "commands.",
    ),
    # --- about Kubernetes ---
    (
        ("kubernetes", "k8s", "orchestration", "helm charts"),
        "Kubernetes manifests are not included but can be adapted from the "
        "Docker setup for large-scale deployments.",
    ),
    # --- about cloud deployment ---
    (
        ("cloud deployment", "aws", "gcp", "azure"),
        "The application can be deployed on any cloud provider that supports "
        "Python web apps. Use a managed database for production.",
    ),
    # --- about monitoring ---
    (
        ("monitoring", "prometheus", "grafana", "health checks"),
        "The /health endpoint can be used for liveness probes. For detailed "
        "metrics, you can integrate monitoring tools.",
    ),
    # --- about alerting ---
    (
        ("alerting", "alerts", "notifications"),
        "Alerts can be set up using monitoring tools to notify the team of "
        "service degradation or errors.",
    ),
    # --- about logging configuration ---
    (
        ("log configuration", "log levels", "debug", "info", "error"),
        "Log levels are configurable via environment variables. By default, "
        "the log level is INFO. Use debug mode for detailed logs.",
    ),
    # --- about environment variables ---
    (
        ("env variables", "environment variables", "configuration"),
        "All configurable settings are read from environment variables or a "
        ".env file. See the configuration file for the full list.",
    ),
    # --- about secrets management ---
    (
        ("secrets", "secret management", "api key storage"),
        "API keys and other secrets should never be hard-coded. Use environment "
        "variables or a secrets manager in production.",
    ),
    # --- about backup and restore ---
    (
        ("backup", "restore", "database backup"),
        "The SQLite database can be backed up by copying the database file. "
        "For production, consider using a database that supports automatic "
        "backups.",
    ),
    # --- about data privacy ---
    (
        ("data privacy", "privacy", "user data"),
        "The system does not store user data other than prediction records "
        "and feedback. No personal information is collected.",
    ),
    # --- about compliance ---
    (
        ("compliance", "data protection", "gdpr"),
        "If you deploy this in regions with data protection regulations, "
        "ensure you handle user data in accordance with local laws. The system "
        "does not process personal data by default.",
    ),
    # --- about commercial use ---
    (
        ("commercial use", "license for business", "can i sell it"),
        "The license allows commercial use, modification, and distribution "
        "with proper attribution. Check the license file for details.",
    ),
    # --- about support channels ---
    (
        ("support", "help channels", "discord", "slack", "forum"),
        "Support is provided via GitHub Issues. For real-time discussions, "
        "check the repository for community channels.",
    ),
    # --- about community ---
    (
        ("community", "community guidelines", "join community"),
        "Community contributors are welcome. Join the discussions on GitHub "
        "and help improve the project.",
    ),
    # --- additional aliases ---
    (
        ("how do i start", "how to begin", "get started"),
        "To get started, clone the repository, set up your environment, and "
        "run the server. Refer to the README for detailed instructions.",
    ),
    (
        ("where is the documentation", "docs location", "readme"),
        "The main documentation is in the README file. The knowledge base "
        "for the chat assistant is in the backend/docs/kb/ folder.",
    ),
    (
        ("what is the default port", "port", "which port"),
        "The server listens on port 8000 by default. You can change it with "
        "the --port flag.",
    ),
    (
        ("can i change the model", "replace model", "custom model"),
        "Yes, you can replace the model with your own custom model as long "
        "as it follows the same input/output format. Update the model path "
        "in the configuration.",
    ),
    (
        ("how to add a new waste class", "new category", "extend classes"),
        "To add a new waste class, you would need to retrain the model with "
        "the new class data and update the knowledge base accordingly.",
    ),
    (
        ("what are the system dependencies", "requirements", "python packages"),
        "Dependencies are listed in requirements.txt and include FastAPI, "
        "PyTorch, Ultralytics, and various utilities. Use pip to install them.",
    ),
    (
        ("is there a demo video", "video demo", "show me"),
        "There is no demo video included, but you can try the live demo on "
        "the frontend page after launching the server.",
    ),
    (
        ("what is the model size", "model parameters", "how big is the model"),
        "The model is optimised for edge deployment. YOLOv8 models vary in "
        "size depending on the variant; the lightweight version is well-suited "
        "for embedded systems.",
    ),
    (
        ("what is inference time", "latency", "how long does it take"),
        "Inference time is typically under 100ms per image on a modern GPU. "
        "CPU inference takes longer but is still practical for moderate loads.",
    ),
    (
        ("what is the detection speed", "fps", "frames per second"),
        "The system can process multiple frames per second depending on the "
        "hardware. Real-time processing is achievable on mid-range GPUs.",
    ),
    (
        ("how many classes can it detect", "class count", "number of categories"),
        "The current model can detect six waste classes: Plastic, Metal, "
        "Paper, Glass, Cardboard, and Organic/Trash. Additional classes can "
        "be added with retraining.",
    ),
    (
        ("what is the accuracy on real-world images", "real-world performance",
         "how does it perform in the wild"),
        "While the model achieves high accuracy on benchmark datasets, "
        "real-world performance depends on image quality, lighting, and "
        "background complexity. The system includes a confidence threshold "
        "to handle uncertain cases.",
    ),
    # ---- REMOVED: "what is the model architecture", "how to install", "what are the classes" ----
    (
        ("what training data was used", "training images", "training samples"),
        "The model was trained on the TrashNet dataset, which contains over "
        "2,500 images of waste items. Data augmentation was used to improve "
        "generalisation.",
    ),
    (
        ("what is the validation accuracy", "test accuracy", "validation results"),
        "The model achieves over 90% validation accuracy on the test set. "
        "Performance is consistent across most classes, with some variation "
        "for visually similar materials.",
    ),
    (
        ("what is the inference speed on CPU", "cpu performance", "cpu inference"),
        "On CPU, inference speed is slower than GPU but still practical for "
        "low-volume use cases. For real-time sorting, a GPU is recommended.",
    ),
    (
        ("what is the inference speed on GPU", "gpu performance", "gpu inference"),
        "On a modern GPU, inference takes around 50-100ms per image, enabling "
        "real-time sorting and high-throughput processing.",
    ),
)

_EXACT: Final[dict[str, str]] = {
    _norm(alias): reply for aliases, reply in _GROUPS for alias in aliases
}

# Phrase variants that exact matching would miss.
_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(r"^(hello|hi|hey|hallo|yo)\b.{0,20}$"),
        _EXACT["hello"],
    ),
    (
        re.compile(r"^(good (morning|afternoon|evening|day))\b.{0,20}$"),
        _EXACT["hello"],
    ),
    (
        re.compile(r"^(thanks|thank you|thx|danke)\b.{0,25}$"),
        _EXACT["thanks"],
    ),
    (
        re.compile(r"^(bye|goodbye|see you|good night)\b.{0,20}$"),
        _EXACT["bye"],
    ),
    (
        re.compile(r"^(who|what) (are|r) (you|u)\b.{0,20}$"),
        _EXACT["who are you"],
    ),
    (
        re.compile(r"^(can|could|will) (you|u) help( me)?\b.{0,15}$"),
        _EXACT["help"],
    ),
    (
        re.compile(r"^how (are|r) (you|u)\b.{0,15}$"),
        _EXACT["how are you"],
    ),
    (
        re.compile(r"^(what|which) (is|are) (the )?waste (classes|categories)\b.{0,20}$"),
        _EXACT["waste classes"],
    ),
    (
        re.compile(r"^(what|how) (is|does) (the )?confidence (score|mean)\b.{0,20}$"),
        _EXACT["what does confidence mean"],
    ),
    (
        re.compile(r"^how (to )?(use|call|access) (the )?api\b.{0,15}$"),
        _EXACT["how to use the api"],
    ),
    (
        re.compile(r"^(what|which) (is|are) the (api )?endpoints?\b.{0,15}$"),
        _EXACT["what endpoints are available"],
    ),
    (
        re.compile(r"^(how|where) (do|can) i (get|create|generate) an api key\b.{0,20}$"),
        _EXACT["how to get an api key"],
    ),
    (
        re.compile(r"^(what|how) (is|does) (the )?history\b.{0,15}$"),
        _EXACT["what is history"],
    ),
    (
        re.compile(r"^(what|why) (is|are) (the )?feedback\b.{0,15}$"),
        _EXACT["what is feedback"],
    ),
    (
        re.compile(r"^(what|how) (is|does) (the )?pipeline\b.{0,15}$"),
        _EXACT["what is the pipeline"],
    ),
    (
        re.compile(r"^(what|how) (is|does) (the )?sorting (work|mechanism)\b.{0,15}$"),
        _EXACT["how does sorting work"],
    ),
    (
        re.compile(r"^(what|how) (is|does) (the )?training (work|process)\b.{0,15}$"),
        _EXACT["how accurate is the model"],
    ),
    (
        re.compile(r"^(what|how) (is|does) (the )?hardware (work|requirement)\b.{0,15}$"),
        _EXACT["what hardware is needed"],
    ),
    (
        re.compile(r"^(what|how) (is|does) (the )?video (processing|inference)\b.{0,15}$"),
        _EXACT["how does video work"],
    ),
    (
        re.compile(r"^(what|how) (is|does) (the )?error handling\b.{0,15}$"),
        _EXACT["what if the model fails"],
    ),
    (
        re.compile(r"^(what|how) (is|does) (the )?accuracy\b.{0,15}$"),
        _EXACT["how accurate is the model"],
    ),
)


def small_talk(question: str) -> str | None:
    """Canned reply for a conversational turn, or None to use retrieval."""
    key = _norm(question)
    if not key:
        return None
    direct = _EXACT.get(key)
    if direct is not None:
        return direct
    for pattern, reply in _PATTERNS:
        if pattern.match(key):
            return reply
    return None