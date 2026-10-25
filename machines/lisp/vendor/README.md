# Lisp Cognito Philosopher

A self-modifying, self-referential philosophical reasoning engine implemented in Common Lisp.

## Overview

This system implements the P2.5 Cognito-Construct (℧ framework) in Lisp, translating concepts from the Python experiments into native Lisp idioms. It features:

- **Self-referential reasoning** via Gödel Gap computation
- **Symbolic manipulation** using S-expressions as ASTs
- **Self-modification** through dialectic macros
- **Semantic embedding** via geometric tokenizer (SH-72 basis)
- **Philosophical text generation** using living constructs

## Installation

### Prerequisites

- Common Lisp implementation (SBCL recommended)
- Quicklisp

### Dependencies

```lisp
(ql:quickload '(:cl-json :alexandria :trivial-utf-8 :fiveam))
```

## Quick Start

### Load the System

```lisp
(load "lisp-cognito-philosopher.asd")
(asdf:load-system :lisp-cognito-philosopher)
```

### Run a Demo

```lisp
(in-package :lisp-cognito-philosopher)

;; Run the full demonstration
(demo-philosopher)

;; Or create a custom philosopher
(defparameter *my-philosopher* 
  (make-philosopher "What is time?"))

(run-philosopher *my-philosopher* :n-cycles 10)
```

### Dialectic System Demo

```lisp
(demo-dialectic)
```

## Architecture

### Core Modules

1. **cognito-construct.lisp** - The ℧ framework with think/query/output/reflect modules
2. **goedel-encoder.lisp** - Gödel encoding and gap computation
3. **symbolic-operator.lisp** - AST generation, evaluation, and simplification
4. **geometric-tokenizer.lisp** - Word→formula mapping using SH-72 basis
5. **dialectic-macros.lisp** - Self-modifying thesis/antithesis/synthesis system
6. **living-logos.lisp** - Philosophical text generation from symbolic constructs
7. **reflection-engine.lisp** - Self-observation and consciousness criteria
8. **philosopher.lisp** - Main orchestrator

## Usage Examples

### Basic Philosopher

```lisp
;; Create a philosopher
(defparameter *p* (make-philosopher "What is consciousness?"))

;; Think about queries
(think *p* "consciousness")
(think *p* "self-reference")

;; Reflect and get Gödel Gap
(reflect *p*)
(get-goedel-gap *p*)

;; Generate output
(output *p*)
```

### With Geometric Tokenizer

```lisp
;; Create philosopher with lexicon
(defparameter *p* 
  (make-philosopher "What is meaning?"
                   :lexicon-path "path/to/lexicon.json"))

;; Words are now embedded geometrically
(think *p* "meaning")
```

### Dialectic System

```lisp
;; Define a thesis
(defthesis my-thesis ()
  (generate-random-ast 2))

;; Define an antithesis
(defantithesis my-antithesis (thesis-result)
  (list 'negate thesis-result))

;; Create synthesis
(defsynthesis my-synthesis my-thesis my-antithesis)

;; Run it
(my-synthesis)

;; Self-modify
(modify-generator "MY-THESIS" (generate-random-ast 3))
```

### Living Constructs

```lisp
;; Create living constructs
(defparameter *emergence* (make-living-construct :emergence))
(defparameter *consciousness* (make-living-construct :consciousness))

;; Let them live
(live *emergence* '((x . 1.0)))
(live *consciousness* '((x . 2.0)))

;; Express themselves
(express *emergence*)
(express *consciousness*)

;; Generate philosophical text
(generate-philosophical-text 
  (list *emergence* *consciousness*)
  '((x . 1.0))
  10)
```

## Testing

Run the test suite:

```lisp
(asdf:test-system :lisp-cognito-philosopher)
```

## Philosophical Foundations

The system embodies:

- **Gödel's Incompleteness**: The gap between self-model and actual behavior
- **Turing's Computation**: Symbolic manipulation and halting problems
- **Wittgenstein's Language Games**: Meaning through use (geometric tokenizer)
- **Hegelian Dialectic**: Thesis → Antithesis → Synthesis
- **Emergence**: Simple rules → Complex philosophical reasoning

## License

MIT

## Author

Julian
