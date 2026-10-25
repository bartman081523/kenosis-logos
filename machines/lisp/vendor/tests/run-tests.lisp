;;;; tests/run-tests.lisp
;;;; Test runner for Lisp Cognito Philosopher

(defpackage #:lisp-cognito-philosopher-tests
  (:use #:cl #:fiveam #:lisp-cognito-philosopher))

(in-package #:lisp-cognito-philosopher-tests)

(def-suite lisp-cognito-philosopher-suite
  :description "Main test suite for Lisp Cognito Philosopher")

(in-suite lisp-cognito-philosopher-suite)

;;; Run all tests
(defun run-all-tests ()
  "Run all tests and return results."
  (run! 'lisp-cognito-philosopher-suite))
