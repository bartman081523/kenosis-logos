;;;; run-bridge.lisp — Einstieg der SBCL-Brücke für machines/lisp_bridge.py.
;;;;
;;;; Aufruf (immer, kein Netz, kein Quicklisp — Vendor braucht nur (:use #:cl)):
;;;;   echo '(:op "ping")' | timeout 45 sbcl --non-interactive \
;;;;        --script machines/lisp/run-bridge.lisp
;;;;
;;;; Lädt die 10 asd-Komponenten des Vendor in asd-Reihenfolge (nur die
;;;; Lademeile — demo/interactive-repl/language-learning/letter-speech/
;;;; persistence/run-demo/simple-test/test-harness/test-load bleiben vendor-
;;;; Provenienz ohne Bridge-Rolle), dann das Protokoll, dann EINEN Auftrag.

(defparameter *bridge-loader-truename*
  (or *load-truename* (make-pathname :name "run-bridge")))

(defparameter *bridge-vendor-dir*
  (make-pathname :name nil :type nil
                 :directory (append (pathname-directory *bridge-loader-truename*)
                                    (list "vendor"))))

(defun bridge-load (filename)
  (let ((path (merge-pathnames filename *bridge-vendor-dir*)))
    (load path)))

(mapc #'bridge-load
      '("package.lisp"
        "utils.lisp"
        "goedel-encoder.lisp"
        "symbolic-operator.lisp"
        "geometric-tokenizer.lisp"
        "cognito-construct.lisp"
        "dialectic-macros.lisp"
        "living-logos.lisp"
        "reflection-engine.lisp"
        "philosopher.lisp"))

(load (merge-pathnames "bridge-protocol.lisp"
                       (or *load-truename* *bridge-loader-truename*)))

(in-package #:lisp-cognito-philosopher)

(run-bridge-main)