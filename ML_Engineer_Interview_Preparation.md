# ML Engineer Internship Interview Preparation

## How to Use This Preparation Pack

This file is the syllabus checklist and study-workspace plan for the ML Engineer Internship interview preparation.

### How I want to study

I am new to machine learning (ML) and large language models (LLMs), so I want to learn in a sequential, practical manner rather than memorising definitions.

- Use AI as a learning assistant and coding partner, but do not blindly copy code it generates.
- Generate code in small, understandable steps.
- Explain the purpose of each library, imported module, class, function, method, and important parameter used.
- Add useful comments directly inside code so I can understand what each part does.
- Explain unfamiliar Python syntax when it appears.
- After generating code, walk through it in execution order: input → processing → output.
- Run the code where possible, inspect the output, and practise changing one thing at a time.
- Keep explanations practical and beginner-friendly. I do not need excessive theory for every topic, but I do need to understand the code I use.
- Practise rewriting small examples without AI assistance so I can verify that I understand them.
- When an error occurs, first learn to read the traceback and investigate it systematically instead of immediately asking AI to fix it.

### Suggested folder and file structure

Create a workspace with one folder for each syllabus topic. Keep small runnable examples separate from notes.

```text
ml-engineer-interview-prep/
├── README.md
├── 01_llm_training_and_inference/
│   ├── notes.md
│   ├── 01_tokenizer/
│   │   └── main.py
│   ├── 02_load_pretrained_model/
│   │   └── main.py
│   ├── 03_inference_parameters/
│   │   └── main.py
│   └── 04_training_basics/
│       └── main.py
├── 02_llm_parameters/
│   ├── notes.md
│   └── examples/
├── 03_agent_harness/
│   ├── notes.md
│   └── examples/
├── 04_prompt_engineering/
│   ├── notes.md
│   └── examples/
├── 05_ml_system_architecture/
│   ├── notes.md
│   └── diagrams/
├── 06_api_development/
│   ├── notes.md
│   ├── flask/
│   └── fastapi/
├── 07_docker/
│   ├── notes.md
│   ├── Dockerfile
│   └── compose.yaml
├── 08_streamlit_or_gradio/
│   ├── notes.md
│   └── app.py
├── 09_debugging/
│   ├── notes.md
│   └── broken_examples/
├── 10_leetcode/
│   ├── arrays/
│   ├── dictionaries/
│   └── strings/
└── vocabulary/
    ├── README.md
    ├── python_syntax.md
    ├── ml_terms.md
    ├── llm_terms.md
    ├── api_and_deployment_terms.md
    └── index.md
```

This is a suggested structure, not a requirement. Add or remove subfolders as your learning progresses.

### How to practise each code example

For every concept or library, use this sequence:

1. **Goal:** Write one sentence describing what the code should do.
2. **Minimum example:** Start with the smallest useful runnable code.
3. **Syntax explanation:** Explain unfamiliar Python syntax and punctuation.
4. **Library/function explanation:** Explain what each imported library, class, function, method and important argument does.
5. **Commented code:** Add useful comments beside the relevant lines.
6. **Execution walkthrough:** Trace the values through the code and describe the output.
7. **Experiment:** Change one input or parameter and observe the result.
8. **Debugging practice:** Introduce or inspect a small error and learn to diagnose it.
9. **Rebuild:** Close the reference and recreate the example from memory.
10. **Interview explanation:** Explain the code and the design choices aloud in simple terms.

For AI-generated code, ask for one small section at a time. Ask the AI to explain every unfamiliar function and parameter, identify which parts are Python built-ins versus third-party library features, and avoid introducing unnecessary abstractions or libraries.

### Vocabulary / dictionary Markdown files

Maintain searchable Markdown vocabulary files. Each entry should include:

- **Term**
- **Technical definition**
- **Plain-English definition**
- **Example or analogy**
- **Small code snippet**, when useful
- **Explanation of the snippet**
- **Related terms**

Use headings consistently so terms can be found with the editor's search feature. Keep `vocabulary/index.md` as a list of terms with links to their detailed entries. Avoid copying a definition without being able to explain it in your own words.

---

# Interview Syllabus Checklist

The following is the complete 10-topic checklist assembled from the syllabus checklist discussed for this interview. Subtopics below are a study checklist; they are not a claim that the employer supplied a separate official weighting for each item.

## 1. Basic LLM Training and Fast Inference

- [ ] LLM fundamentals and transformer architecture
- [ ] Tokenization and embeddings
- [ ] Pretraining vs fine-tuning vs inference
- [ ] Loading pretrained models
- [ ] Training loop, loss and optimizer
- [ ] Dataset preparation and preprocessing
- [ ] LoRA and QLoRA basics
- [ ] Fast inference frameworks (vLLM, Hugging Face Transformers)
- [ ] Latency, throughput and GPU memory

### Practical examples to create

- [ ] Tokenizer: convert text to token IDs and inspect the result.
- [ ] Pretrained model: load a small model and generate output.
- [ ] Inference parameters: change one generation parameter at a time.
- [ ] Training basics: understand a minimal training loop and inspect loss.
- [ ] Add comments explaining each import, function, argument and important line.

## 2. LLM Parameters — Training and Inference

- [ ] Learning rate
- [ ] Batch size
- [ ] Epochs
- [ ] Optimizers and loss functions
- [ ] Temperature
- [ ] Top-k and top-p sampling
- [ ] Max new tokens
- [ ] Context window
- [ ] Repetition penalty
- [ ] Quantization
- [ ] Parameter tuning and trade-offs

### Practical examples to create

- [ ] Create a small configuration file and explain each parameter.
- [ ] Compare outputs with different generation settings.
- [ ] Record how output length, latency, memory and response quality change.

## 3. Agent Harness Design Without Frameworks

- [ ] LLM API integration
- [ ] Agent execution loop
- [ ] Tool/function calling
- [ ] Tool registration and dispatch
- [ ] Tool argument validation
- [ ] Context and conversation history management
- [ ] Context-window limits and summarization
- [ ] Memory management
- [ ] Stop conditions and maximum steps
- [ ] Error handling and retries
- [ ] Tool safety and permissions

### Practical examples to create

- [ ] Build a simple Python loop that can call a calculator function.
- [ ] Validate tool names and arguments before execution.
- [ ] Add a maximum-step limit and a clear stop condition.
- [ ] Record tool calls and results so the execution can be debugged.

## 4. Prompt Engineering

- [ ] Role, task and instruction design
- [ ] Zero-shot prompting
- [ ] Few-shot prompting
- [ ] System vs user prompts
- [ ] Structured output (JSON)
- [ ] Output constraints and schemas
- [ ] Prompt templates
- [ ] Chain-of-thought alternatives and task decomposition
- [ ] Prompt injection awareness
- [ ] Prompt testing and evaluation

### Practical examples to create

- [ ] Write a prompt that classifies text into a defined set of labels.
- [ ] Request structured JSON and validate the returned data.
- [ ] Test normal, ambiguous, missing-input and adversarial cases.
- [ ] Compare prompt versions using a small repeatable test set.

## 5. Basic ML System Architecture

- [ ] End-to-end ML pipeline
- [ ] Data collection and preprocessing
- [ ] Feature engineering
- [ ] Model training and evaluation
- [ ] Model serialization and loading
- [ ] Training vs inference pipelines
- [ ] API and model-serving architecture
- [ ] Database and model storage
- [ ] Batch vs real-time inference
- [ ] Logging and monitoring
- [ ] Model versioning
- [ ] Scalability, latency and reliability

### Practical examples to create

- [ ] Draw the path from user input to model output.
- [ ] Save and reload a small trained model.
- [ ] Explain preprocessing, inference, postprocessing and monitoring.
- [ ] Describe how the design changes when requests increase or latency rises.

## 6. Flask + Gunicorn and FastAPI + Uvicorn

- [ ] Flask fundamentals
- [ ] FastAPI fundamentals
- [ ] Routes and HTTP methods
- [ ] GET vs POST
- [ ] Request and response handling
- [ ] Input validation and Pydantic
- [ ] Error handling and HTTP status codes
- [ ] REST API basics
- [ ] WSGI vs ASGI
- [ ] Gunicorn basics
- [ ] Uvicorn basics
- [ ] Workers and concurrency
- [ ] Environment variables
- [ ] API testing and documentation

### Practical examples to create

- [ ] Build a health-check endpoint.
- [ ] Build a POST endpoint that accepts input and returns a result.
- [ ] Add request validation and useful error responses.
- [ ] Test the endpoint and inspect the generated API documentation.
- [ ] Explain the role of the web framework and the server separately.

## 7. Docker and Docker Compose

- [ ] Containers vs virtual machines
- [ ] Images vs containers
- [ ] Dockerfile instructions
- [ ] Base images
- [ ] Dependencies and requirements files
- [ ] Build and run commands
- [ ] Port mapping
- [ ] Environment variables
- [ ] Volumes and bind mounts
- [ ] Container networking
- [ ] Docker Compose YAML structure
- [ ] Multi-container applications
- [ ] Logs and debugging
- [ ] Health checks
- [ ] Multi-stage builds
- [ ] Image optimization and security basics

### Practical examples to create

- [ ] Containerize the API from Topic 6.
- [ ] Build and run the image locally.
- [ ] Configure ports and environment variables.
- [ ] Inspect logs and diagnose a deliberately broken configuration.
- [ ] Use Compose when multiple services are needed.

## 8. Streamlit / Gradio — Choose at Least One

- [ ] Installation and setup
- [ ] Text input and output components
- [ ] Buttons and event handling
- [ ] Forms and file uploads
- [ ] Displaying model predictions
- [ ] Connecting to APIs
- [ ] Loading and using ML models
- [ ] Error handling and loading states
- [ ] Session state / state management
- [ ] Layouts and columns
- [ ] Basic UI customization
- [ ] Running and sharing applications

### Practical examples to create

- [ ] Build a minimal UI that accepts input and displays model output.
- [ ] Connect the UI to the API from Topic 6.
- [ ] Display loading and error states.
- [ ] Explain the callback or function that runs when the user interacts.

## 9. Debugging Without ChatGPT or Claude

- [ ] Reading Python tracebacks
- [ ] Syntax and runtime errors
- [ ] Import and dependency errors
- [ ] Virtual environments
- [ ] Package version conflicts
- [ ] File paths and working directories
- [ ] Type errors and data validation
- [ ] API and HTTP errors
- [ ] Network and timeout errors
- [ ] Environment variables and configuration
- [ ] Docker build and runtime errors
- [ ] Logging and print debugging
- [ ] Minimal reproducible examples
- [ ] Root-cause analysis and systematic testing

### Practical examples to create

- [ ] Diagnose an import error without asking AI to fix it.
- [ ] Read a traceback and locate the relevant line.
- [ ] Check the Python interpreter and installed package versions.
- [ ] Reproduce a bug with the smallest possible example.
- [ ] Write down the root cause and how the fix was verified.

## 10. LeetCode — Arrays and Dictionaries

### Arrays

- [ ] Traversal and iteration
- [ ] Searching
- [ ] Minimum and maximum
- [ ] Two pointers
- [ ] Sliding window
- [ ] Prefix sums
- [ ] In-place modifications
- [ ] Sorting basics
- [ ] Duplicate detection

### Dictionaries / Hash Maps

- [ ] Key-value operations
- [ ] Frequency counting
- [ ] Lookup optimization
- [ ] Complement lookup
- [ ] Grouping elements
- [ ] Counting occurrences
- [ ] Dictionary and set usage

### Additional essentials

- [ ] Strings
- [ ] Loops and conditionals
- [ ] List operations
- [ ] Time and space complexity
- [ ] Brute force vs optimized solutions
- [ ] Edge cases and dry runs

### Practical examples to create

- [ ] Solve Two Sum using brute force and a dictionary.
- [ ] Solve Contains Duplicate using a set.
- [ ] Solve Valid Anagram using frequency counting.
- [ ] Explain time and space complexity for each solution.
- [ ] Test empty inputs, duplicates and boundary cases.

---

# Final Instructions and Learning Rules

**This is the complete syllabus checklist assembled for my interview preparation from the topics discussed.** Keep the checklist as the master list and track progress against it. If the employer supplies an updated or more detailed official syllabus, compare it with this checklist rather than assuming this file overrides the employer's latest instructions.

The interview preparation must focus not only on knowing the topic names, but also on demonstrating practical understanding.

The interviewers are particularly keen to assess:

1. **Python syntax:** I should understand the syntax used in the code I write, including variables, functions, classes, imports, loops, conditionals, exceptions, comprehensions, type hints and file handling where relevant.
2. **Libraries and functions:** I should know what the libraries I import are for, what the functions and methods I call do, what their important arguments mean, and what they return.
3. **Hands-on implementation:** I should create runnable code snippets for the concepts I learn instead of studying only definitions.
4. **Code generated with AI:** I may use AI to generate code, but I must learn from it. I should request small, sequential code sections with comments and explanations, understand the code before moving on, test it, and practise recreating it independently.
5. **Code comments and explanations:** Important lines should have clear, useful comments. Each example should also have a short explanation of its purpose, inputs, processing and output.
6. **Vocabulary files:** Maintain searchable Markdown dictionaries containing each technical term's definition, a plain-English explanation, an example or analogy, a small code snippet where useful, and related terms.
7. **Debugging independence:** I should practise reading tracebacks, inspecting documentation and logs, checking environments, isolating failures and verifying fixes without depending on AI.
8. **Interview readiness:** For every important topic, I should be able to explain what it is, why it is used, how it works at a basic level, demonstrate a small implementation and describe at least one limitation or trade-off.

## Rule for every AI-assisted coding session

Use this prompt as the default instruction when asking AI to help with a topic:

> Teach me this concept as a beginner preparing for an ML Engineer Internship interview. Start with the simplest useful example and proceed sequentially. Generate only a small section of code at a time. Add clear comments to important lines. Explain every unfamiliar Python syntax element, imported library, class, function, method and important parameter. Distinguish Python built-ins from third-party library features. Explain the input, execution flow and output in plain English. Tell me how to run the code and how to test it. Do not use unnecessary libraries or abstractions. After I understand the example, give me a small modification task to solve myself. Do not move to the next step until I can explain the current one.

**Final goal:** I should not merely have code that works. I should understand the code I use well enough to explain it, modify it, debug it and write a basic version independently during the interview.
