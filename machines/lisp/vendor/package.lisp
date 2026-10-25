;;;; package.lisp
;;;; Package definitions for Lisp Cognito Philosopher

(defpackage #:lisp-cognito-philosopher
  (:nicknames #:lcp)
  (:use #:cl)
  (:export
   ;; Main entry point
   #:make-philosopher
   #:run-philosopher
   
   ;; Cognito Construct
   #:cognito-construct
   #:make-cognito-construct
   #:construct
   #:think
   #:query
   #:output
   #:reflect
   #:get-goedel-gap
   
   ;; Gödel Encoder
   #:goedel-encode
   #:compute-structural-signature
   #:predict-output-signature
   #:compute-goedel-gap
   
   ;; Symbolic Operator
   #:generate-random-ast
   #:evaluate-ast
   #:simplify-ast
   #:ast-to-string
   
   ;; Geometric Tokenizer
   #:make-geometric-tokenizer
   #:load-lexicon
   #:word-to-formula
   #:embed-word
   #:resonance
   #:get-gematria
   
   ;; Dialectic Macros
   #:defthesis
   #:defantithesis
   #:defsynthesis
   #:modify-generator
   
   ;; Living Logos
   #:living-construct
   #:make-living-construct
   #:live
   #:express
   #:mutate
   #:get-stability
   
   ;; Reflection Engine
   #:observe-self
   #:predict-next-state
   #:predict-gap
   #:self-correct
   #:check-consciousness-criteria
   
   ;; Constants
   #:+pi+
   #:+e+
   #:+phi+))

(in-package #:lisp-cognito-philosopher)

;;; Mathematical constants
(defconstant +pi+ pi)
(defconstant +e+ (exp 1.0d0))
(defconstant +phi+ (/ (+ 1 (sqrt 5.0d0)) 2)) ; Golden ratio
