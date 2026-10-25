;;;; test-harness.lisp
;;;; Standardized Deterministic Test Harness

(format t "Loading system...~%")
(load "package.lisp")
(load "utils.lisp")
(load "goedel-encoder.lisp")
(load "symbolic-operator.lisp")
(load "geometric-tokenizer.lisp")
(load "cognito-construct.lisp")
(load "dialectic-macros.lisp")
(load "living-logos.lisp")
(load "reflection-engine.lisp")
(load "philosopher.lisp")
(load "letter-speech.lisp")
(load "persistence.lisp")
(load "language-learning.lisp")
;; Skip translator.lisp to ensure compatibility with non-translated versions

(in-package :lisp-cognito-philosopher)

;; Initialize Deterministic RNG
(set-rng-seed 42)
(format t "🎲 Deterministic RNG initialized (Seed 42)~%")

;; Create philosopher
(defparameter *p* (make-philosopher ""))

;; Loop through standard input
(loop for line = (read-line *standard-input* nil nil)
      while line do
      (let ((input (string-trim " " line)))
        (unless (string= input "")
             (format t "YOU: ~A~%" input)
             (learn-from-input *p* input)
             (let ((response (generate-response *p* input)))
               (format t "PHILOSOPHER: ~A~%" response)))))
