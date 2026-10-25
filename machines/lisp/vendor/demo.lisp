#!/usr/bin/env sbcl --script
;;;; demo.lisp
;;;; Quick demo script for the Lisp Cognito Philosopher

(load "lisp-cognito-philosopher.asd")
(asdf:load-system :lisp-cognito-philosopher)

(in-package :lisp-cognito-philosopher)

(format t "~%~%")
(format t "╔════════════════════════════════════════════════════════════╗~%")
(format t "║    Lisp Cognito Philosopher - Quick Demo                  ║~%")
(format t "╚════════════════════════════════════════════════════════════╝~%")
(format t "~%")

;; Run the demonstration
(demo-philosopher)

(format t "~%~%Press Enter to see dialectic demonstration...")
(read-line)

(demo-dialectic)

(format t "~%~%Demo complete!~%")
